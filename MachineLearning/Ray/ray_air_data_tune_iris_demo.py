import pandas as pd
import ray
import os
import psutil
import pyarrow.compute as pc
from ray import train, tune
from ray.air import session, Checkpoint
from ray.air.config import ScalingConfig, DatasetConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.data.preprocessors import Chain, BatchMapper, Concatenator, StandardScaler
from ray.tune.tuner import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pickle

# more verbose
#ray.data.DataContext.get_current().execution_options.verbose_progress = True
num_cpus = psutil.cpu_count(logical=False)
num_partitions = min(num_cpus - 2, 32)
print(f'Num CPUs: {num_cpus}')
print(f'Num partitions: {num_partitions}')

if not ray.is_initialized():
    # init a driver
    ray.init( num_cpus=num_cpus, num_gpus=1, include_dashboard=True)

# load data
ds = ray.data.read_csv("iris.csv",)

# model
class NeuralNetwork(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        probs = F.log_softmax(x, dim=1)
        return probs

# preprocessor
def preprocessor_fn(df: pd.DataFrame, classes: dict) -> pd.DataFrame:
    df["variety"] = df["variety"].map(classes)
    return df

# get labels
classes = {label: i for i, label in enumerate(ds.unique("variety"))}

# map relabeling
ds = ds.map_batches(preprocessor_fn, fn_kwargs={"classes": classes}, batch_format="pandas")

# Create a preprocessor to scale some columns and concatenate the result.
preprocessor = Chain(StandardScaler(columns=["sepal.length", "sepal.width", "petal.length", "petal.width"]),
                     Concatenator(exclude=["variety"], dtype=np.float32)
)

def train_loop_per_worker(config: dict):

    # set mixed
    train.torch.accelerate(amp=True)

    # unpack config
    batch_size = config["batch_size"]
    lr = config["lr"]
    num_epochs = config["num_epochs"]
    num_classes = config['num_classes']
    num_hidden = config['num_hidden']
    num_features = config['num_features']

    # get train/valid shard
    train_shard = session.get_dataset_shard("train")
    valid_shard = session.get_dataset_shard("valid")

    # init model
    model = NeuralNetwork(num_features=num_features,
                          num_hidden=num_hidden,
                          num_classes=num_classes)

    # prepare model
    model = train.torch.prepare_model(model=model,
                                      move_to_device=True,
                                      parallel_strategy=None,  # ddp / fsdp
                                      )

    # loss fn
    loss_fn = nn.NLLLoss()

    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = train.torch.prepare_optimizer(optimizer)

    for epoch in range(num_epochs):
        # set train
        model.train()

        # torch dataloader replacement
        for batches in train_shard.iter_torch_batches(batch_size=batch_size,
                                                      device='cuda',
                                                      dtypes=torch.float32,
                                                      prefetch_batches=8,  # num_workers
                                                      drop_last=True):
            
            inputs, labels = batches["concat_out"], batches["variety"].to(torch.int64)
            output = model(inputs)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            train.torch.backward(loss)  # mixed prec loss
            optimizer.step()

        session.report({"loss": loss.item(), "epoch": epoch + 1, "split": "train", 'acc': np.nan})
        
        # eval loop
        model.eval()
        total_loss = 0
        correct = 0
        total_size = 0
        steps = 0
        for batches in valid_shard.iter_torch_batches(batch_size=batch_size*2,
                                                      device='cuda',
                                                      dtypes=torch.float32,
                                                      prefetch_batches=8,  # num_workers
                                                      drop_last=False):
            
            inputs, labels = batches["concat_out"], batches["variety"].to(torch.int64)

            # forward
            output = model(inputs)
            _, predicted = torch.max(output, 1)

            # loss
            loss = loss_fn(output, labels)
            total_loss += loss.detach().float().item()
            steps += 1
            total_size += labels.size(0)
            correct += (predicted == labels).sum().item()


        session.report({"loss": (total_loss / steps),
                        "acc": correct / total_size,
                        "epoch": epoch + 1,
                        "split": "val"},
                        checkpoint=Checkpoint.from_dict(dict(epoch=epoch, model=model.state_dict())), )


# split
train_df, test_df = ds.random_shuffle().train_test_split(test_size=0.2)

# keep the 2 checkpoints with the smallest val_loss
checkpoint_config = CheckpointConfig(
    num_to_keep=2,
    checkpoint_score_attribute="loss",
    checkpoint_score_order="min"
)

# trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={"lr": 0.0048, "batch_size": 8, "num_epochs": 10, "num_classes": 3, "num_hidden": 256, "num_features": 4},
    datasets={"train": train_df, "valid": test_df},
    scaling_config=ScalingConfig(num_workers=1, use_gpu=True),  # if use_gpu=True, num_workers = num GPUs
    preprocessor=preprocessor,
    run_config=RunConfig(checkpoint_config=checkpoint_config),
    dataset_config={"train": DatasetConfig(fit=True,  # fit() on train only; transform all
                                           split=True,  # split data acrosss workers if num_workers > 1
                                           global_shuffle=False,  # local shuffle
                                           max_object_store_memory_fraction=0.2  # stream mode; 20% of available object store memory
                                           ),
                    },
)

# fit
result = trainer.fit()

# returns the last saved checkpoint
result.checkpoint

# returns the N best saved checkpoints, as configured in ``RunConfig.CheckpointConfig``
result.best_checkpoints

# returns the final metrics as reported
result.metrics

# returns the Exception if training failed.
result.error

# Returns a pandas dataframe of all reported results
result.metrics_dataframe
result.metrics_dataframe.query("split == 'val'").tail(3)

# write to pickle / save to cloud storage; 2nd best checkpoint
ckpt = Checkpoint.from_directory(result.best_checkpoints[1][0].to_directory());
my_dict = ckpt.to_dict();

# pickle
with open('best_ckpt.pickle', 'wb') as handle:
    pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# tune
config = {
        "lr": tune.loguniform(1e-4, 5e-1),
        "batch_size": tune.choice([2, 4, 8, 16, 32]),
        "num_hidden": tune.choice([2 ** i for i in range(2, 9)]),
    }

scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=30,
        grace_period=3,
        reduction_factor=2,
        brackets=1)

tuner = Tuner(trainer,
              param_space={"train_loop_config": config},
              tune_config=TuneConfig(num_samples=50,  # n trials
                                        metric="acc",
                                        mode="max",
                                        scheduler=scheduler,
                                        ),
)

# fit tune
result_grid = tuner.fit()

# get results
result_grid.get_best_result()
