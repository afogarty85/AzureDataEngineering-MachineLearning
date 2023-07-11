import pandas as pd
import ray
import os
from typing import Dict, Optional, Union
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
np.set_printoptions(suppress=True)

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

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=50000, n_features=8, n_informative=6, n_redundant=2,
                                      n_classes=10,  random_state=1)

# stack
arr = np.column_stack((X, y))
df = pd.DataFrame(arr, columns=[f'feature_{i}' for i in range(0, 9)])
df = df.rename(columns={'feature_8': 'variety'})

# convert to ray data
ds = ray.data.from_pandas([df])

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


# Create a preprocessor to scale some columns and concatenate the result.
preprocessor = Chain(StandardScaler(columns=[f'feature_{i}' for i in range(0, 8)]),
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
train_set, valid_test_residual_set = ds.random_shuffle().train_test_split(test_size=0.3)
test_set, valid_set = valid_test_residual_set.train_test_split(test_size=0.5)
calibration_set, valid_set = valid_set.train_test_split(test_size=0.5)

# apply preprocessor
train_set = preprocessor.fit_transform(train_set)
valid_set = preprocessor.transform(valid_set)
test_set = preprocessor.transform(test_set)
calibration_set = preprocessor.transform(calibration_set)


# trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={"lr": 0.0029, "batch_size": 1024, "num_epochs": 10, "num_classes": 10, "num_hidden": 256, "num_features": 8},
    datasets={"train": train_set, "valid": valid_set},
    scaling_config=ScalingConfig(num_workers=1, use_gpu=True),  # if use_gpu=True, num_workers = num GPUs
    run_config=RunConfig(checkpoint_config=CheckpointConfig(
                        num_to_keep=1,
                        checkpoint_score_attribute="loss",
                        checkpoint_score_order="min")),
    dataset_config={"train": DatasetConfig(fit=True,  # fit() on train only; transform all
                                           split=True,  # split data acrosss workers if num_workers > 1
                                           global_shuffle=False,  # local shuffle
                                           max_object_store_memory_fraction=0.2  # stream mode; 20% of available object store memory
                                           ),
                    },
)


# tune
config = {
        "lr": tune.loguniform(1e-4, 5e-1),
        "batch_size": tune.choice([4, 8, 16, 32, 64, 128, 256, 512, 1024]),
        "num_hidden": tune.choice([2 ** i for i in range(2, 12)]),
    }

scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=30,
        grace_period=3,
        reduction_factor=2,
        brackets=1)

tuner = Tuner(trainer,
              param_space={"train_loop_config": config},
              tune_config=TuneConfig(num_samples=15,  # n trials
                                        metric="acc",
                                        mode="max",
                                        scheduler=scheduler,
                                        ),
)

# fit tune
result_grid = tuner.fit()

# get results
result_grid.get_best_result()

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

# returns a pandas dataframe of all reported results
result.metrics_dataframe.query("split == 'val'").tail(3)

# write to pickle / save to cloud storage; best checkpoint
ckpt = Checkpoint.from_directory(result.best_checkpoints[0][0].to_directory())
with open('best_ckpt.pickle', 'wb') as handle:
    pickle.dump(ckpt.to_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)



# custom batch predictor
model = NeuralNetwork(num_features=8,
                          num_hidden=256,
                          num_classes=10)

# load state
with open(r'best_ckpt.pickle', 'rb') as fp:
    ckpt = pickle.load(fp)
model.load_state_dict(ckpt['model'])

class TorchPredictor:
    def __init__(self, model):
        self.model = model.cuda()
        self.model.eval()

    def __call__(self, batch):
        # transform to tensor / attach to GPU
        tensor = torch.as_tensor(batch["concat_out"], dtype=torch.float32, device="cuda")

        # like no_grad
        with torch.inference_mode():
            # forward and back to cpu
            softmax_scores = self.model(tensor)
            return {
                    "softmax": softmax_scores.cpu().numpy(),
                    "labels": batch['variety']
                    }

# map against calibration_set
out = calibration_set.map_batches(TorchPredictor(model=model),
                             num_gpus=1,
                             batch_size=64,
                             compute=ray.data.ActorPoolStrategy(size=1),
                             )
# materialize results
out = out.take_all()

# unpack softmax preds
smx = np.array([x['softmax'] for x in out])

# unpack labels
labels = np.array([x['labels'] for x in out]).astype(int)


# adapative prediction set
n = smx.shape[0] // 2
# 1-alpha is the desired coverage
alpha = 0.1 
# Set RAPS regularization parameters (larger lam_reg and smaller k_reg leads to smaller sets)
lam_reg = 0.0001
k_reg = 3
# set this to False in order to see the coverage upper bound hold
disallow_zero_sets = False 
# Set this to True in order to see the coverage upper bound hold
rand = True 
reg_vec = np.array(k_reg*[0,] + (smx.shape[1]-k_reg)*[lam_reg,])[None,:]

# Split the softmax scores into calibration and validation sets (save the shuffling)
idx = np.array([1] * n + [0] * (smx.shape[0]-n)) > 0
np.random.shuffle(idx)
cal_smx, val_smx = smx[idx,:], smx[~idx,:]
cal_labels, val_labels = labels[idx], labels[~idx]

# step 1: scores; larger values = worse fit
cal_pi = cal_smx.argsort(1)[:, ::-1] # sort from most -> least likely; e.g., np.exp(x) to get probs
cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
cal_srt_reg = cal_srt + reg_vec  # add regularization
cal_L = np.where(cal_pi == cal_labels[:, None])[1]  # find where true matches prob
# add scores of all labels until true class is reached
# the amount of estimated probability needed to get true label in predicition set
conformal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n), cal_L]

# step 2: threshold
qhat = np.quantile(conformal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')

# validate
n_val = val_smx.shape[0]
val_pi = val_smx.argsort(1)[:,::-1]
val_srt = np.take_along_axis(val_smx, val_pi, axis=1)
val_srt_reg = val_srt + reg_vec
val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
prediction_sets = np.take_along_axis(indicators, val_pi.argsort(axis=1),axis=1)

empirical_coverage = prediction_sets[np.arange(n_val), val_labels.astype(int)].mean()
print(f"The empirical coverage is: {empirical_coverage}")
print(f"The quantile is: {qhat}")

# map against test set
test_out = test_set.map_batches(TorchPredictor(model=model),
                             num_gpus=1,
                             batch_size=64,
                             compute=ray.data.ActorPoolStrategy(size=1),
                             )
# materialize results
test_out = test_out.take_all()

# unpack softmax preds
softmax_preds = np.array([x['softmax'] for x in test_out])

# init true vals
true_vals = np.array([f'class_{i}' for i in range(0, 10)])

# get prediction sets
for i in range(0, 15):
    # get softmax for x_i
    _smx = softmax_preds[i]
    # sort labels from most to least likely
    _pi = np.argsort(_smx)[::-1]
    # sort softmax scores from most to least likely
    _srt = np.take_along_axis(_smx, _pi, axis=0)
    # add reg
    _srt_reg = _srt + reg_vec.squeeze()
    # sum the probability mass 
    _srt_reg_cumsum = _srt_reg.cumsum()
    # return all labels that are lt/eq to qhat
    _ind = (_srt_reg_cumsum - np.random.rand()*_srt_reg) <= qhat if rand else _srt_reg_cumsum - _srt_reg <= qhat
    # sort / create prediction set based on qhat threshold and most -> least likely
    prediction_set = np.take_along_axis(_ind, _pi.argsort(), axis=0)
    print(f"The prediction set is: {true_vals[prediction_set]}")

