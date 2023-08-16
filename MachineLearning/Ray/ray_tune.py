import pandas as pd
import ray
import os
import psutil
import pyarrow.compute as pc
from ray import train, tune
from ray.air import session, Checkpoint
from ray.air.config import ScalingConfig, DatasetConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer, TorchCheckpoint
from ray.data.preprocessors import Chain, BatchMapper, Concatenator, StandardScaler
from ray.tune.tuner import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import datetime
import pickle
import torchmetrics
from ray.train import DataConfig
from azureml.core import Run
from sklearn.metrics import f1_score
np.set_printoptions(suppress=True)


if not ray.is_initialized():
    # init a driver
    ray.init()
    print('ray initialized')


# init Azure ML run context data
aml_context = Run.get_context()

# set data context
ctx = ray.data.DataContext.get_current()
ctx.use_push_based_shuffle = True
ctx.use_streaming_executor = True
ctx.execution_options.locality_with_output = True
ctx.execution_options.verbose_progress = True

# cluster available resources which can change dynamically
print(f"Ray cluster resources_ {ray.cluster_resources()}")


# model
class ConcatenatedEmbeddings(torch.nn.Module):
    """Map multiple categorical variables to concatenated embeddings.

    Args:
        embedding_table_shapes: A dictionary mapping column names to
            (cardinality, embedding_size) tuples.
        dropout: A float.
        sparse_columns: A list of sparse columns

    Inputs:
        x: An int64 Tensor with shape [batch_size, num_variables].

    Outputs:
        A Float Tensor with shape [batch_size, embedding_size_after_concat].
    """

    def __init__(self, embedding_table_shapes, dropout=0.0):
        super().__init__()

        self.embedding_layers = torch.nn.ModuleList(
            [
                torch.nn.Embedding(cat_size, emb_size,)
                for col, (cat_size, emb_size) in embedding_table_shapes.items()
            ]
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        if len(x.shape) <= 1:
            x = x.unsqueeze(0)
        x = [layer(x[:, i]) for i, layer in enumerate(self.embedding_layers)]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return x


class MLP(torch.nn.Module):
    """
    Generic Base Pytorch Model, that contains support for Categorical and Continuous values.

    Parameters
    ----------
    embedding_tables_shapes: dict
        A dictionary representing the <column>: <max cardinality of column> for all
        categorical columns.
    num_continuous: int
        Number of continuous columns in data.
    emb_dropout: float, 0 - 1
        Sets the embedding dropout rate.
    layer_hidden_dims: list
        Hidden layer dimensions.
    layer_dropout_rates: list
        A list of the layer dropout rates expressed as floats, 0-1, for each layer
    """

    def __init__(
        self,
        embedding_table_shapes,
        num_continuous,
        emb_dropout,
        layer_hidden_dims,
        layer_dropout_rates,
        num_classes,
    ):
        super().__init__()
        mh_shapes = None
        if isinstance(embedding_table_shapes, tuple):
            embedding_table_shapes, mh_shapes = embedding_table_shapes
        if embedding_table_shapes:
            self.initial_cat_layer = ConcatenatedEmbeddings(embedding_table_shapes, dropout=emb_dropout)
        self.initial_cont_layer = torch.nn.BatchNorm1d(num_continuous)

        embedding_size = sum(emb_size for _, emb_size in embedding_table_shapes.values())
        if mh_shapes is not None:
            embedding_size = embedding_size + sum(emb_size for _, emb_size in mh_shapes.values())
        layer_input_sizes = [embedding_size + num_continuous] + layer_hidden_dims[:-1]
        layer_output_sizes = layer_hidden_dims
        self.layers = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm1d(output_size),
                torch.nn.Dropout(dropout_rate),
            )
            for input_size, output_size, dropout_rate in zip(
                layer_input_sizes, layer_output_sizes, layer_dropout_rates
            )
        )

        self.output_layer = torch.nn.Linear(layer_output_sizes[-1], num_classes)

    def forward(self, x_cat, x_cont):
        concat_list = []
        if x_cat.dim() == 1:
            x_cat = x_cat.unsqueeze(1)
        # must use is not None for tensor, and len logic for empty list
        if x_cat is not None and len(x_cat) > 0:
            x_cat = self.initial_cat_layer(x_cat)
            concat_list.append(x_cat)
        if x_cont is not None and len(x_cont) > 0:
            x_cont = self.initial_cont_layer(x_cont)
            concat_list.append(x_cont)
        # if no layers in concat_list this breaks by design
        if len(concat_list) > 1:
            x = torch.cat(concat_list, 1)
        else:
            x = concat_list[0]
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


# load data from input -- run.input_datasets['RandomizedTest'] is a locally mounted path... /mnt/..; faster than abfs route
print(f'Getting data...')
valid_set = ray.data.read_parquet(aml_context.input_datasets['RTE'],
                                  use_threads=True,
                                  filter=pc.field('test_split').isin(['valid'])) \
                                .drop_columns(['test_split'])


# cat cols
cat_cols = ['INPUT__REG__m_cmd_merge_mode_0_',
            'INPUT__REG__m_cmd_merge_mode_1_',
            'INPUT__REG__m_hazard_type',
            'INPUT__YML__testname',
            'INPUT__REG__m_ddr_speed_variant']

# set model params
# (max_val, n_features) -- max_val needs to be count(distinct(val)) +1 we can encounter
EMBEDDING_TABLE_SHAPES = {
    'INPUT__REG__m_cmd_merge_mode_0_': (5, 25),
    'INPUT__REG__m_cmd_merge_mode_1_': (5, 25),
    'INPUT__REG__m_hazard_type': (6, 25),
    'INPUT__YML__testname': (523, 200),
    'INPUT__REG__m_ddr_speed_variant': (6, 25),
}

NUM_CLASSES = 64
EMBEDDING_DROPOUT_RATE = 0.01
DROPOUT_RATES = [0.01, 0.01]
HIDDEN_DIMS = [2000, 500]
NUM_CONTINUOUS = 223
num_epochs = 15
batch_size = 512
NUM_CONTINUOUS = 223

# Create a preprocessor to concatenate
preprocessor = Chain(Concatenator(include=cat_cols, dtype=np.int32, output_column_name=['x_cat']),
                     Concatenator(exclude=["SYNDROME", "x_cat"], dtype=np.float32, output_column_name=['x_cont'])
                     )


# apply preprocessor
valid_set = preprocessor.fit_transform(valid_set)


# time function
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# train loop
def train_loop_per_worker(config: dict):

    # set mixed
    train.torch.accelerate(amp=True)

    # unpack config
    batch_size = config["batch_size"]
    lr = config["lr"]
    num_epochs = config["num_epochs"]
    DROPOUT_RATES = config['dropout_rates']
    weight_decay = config['l2']

    # init model
    model = MLP(embedding_table_shapes=EMBEDDING_TABLE_SHAPES,
                num_continuous=NUM_CONTINUOUS,
                emb_dropout=EMBEDDING_DROPOUT_RATE,
                layer_hidden_dims=HIDDEN_DIMS,
                layer_dropout_rates=DROPOUT_RATES,
                num_classes=NUM_CLASSES
                )

    # prepare model
    model = train.torch.prepare_model(model=model,
                                      move_to_device=True,
                                      parallel_strategy=None,  # ddp / fsdp / None
                                      )

    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = train.torch.prepare_optimizer(optimizer)

    class_weights = np.array([1.72375779e-02, 5.57314565e-01, 1.21002182e+00, 1.62034115e+00,
                        2.36676911e+00, 3.62426610e+00, 3.84672974e+00, 4.45735977e+00,
                        5.32213965e+00, 7.40028374e+00, 7.82806709e+00, 1.42187537e+01,
                        1.49731796e+01, 1.63186752e+01, 1.72847743e+01, 1.88530725e+01,
                        2.04518930e+01, 2.33746636e+01, 2.35859651e+01, 2.38349488e+01,
                        2.43925968e+01, 2.61111173e+01, 2.65006491e+01, 2.66611297e+01,
                        3.63216200e+01, 3.86353830e+01, 3.89607498e+01, 3.89369453e+01,
                        3.94485204e+01, 4.08506905e+01, 4.31440650e+01, 4.69279075e+01,
                        5.01135105e+01, 5.11975083e+01, 5.19074699e+01, 6.61903806e+01,
                        6.82569469e+01, 7.22725167e+01, 7.43618418e+01, 7.45837655e+01,
                        8.92269221e+01, 1.01733006e+02, 1.06680223e+02, 1.07250250e+02,
                        1.08816014e+02, 1.18349224e+02, 1.30170247e+02, 1.42075796e+02,
                        1.55966957e+02, 1.57178686e+02, 1.65807463e+02, 1.77059383e+02,
                        1.77651679e+02, 1.93188627e+02, 1.93540653e+02, 1.95905621e+02,
                        2.02842678e+02, 2.07718704e+02, 2.09425233e+02, 2.15079644e+02,
                        3.58526573e+02, 3.59487229e+02, 3.65623055e+02, 3.91931725e+02])
    # create weights
    class_weighting = torch.tensor(class_weights, dtype=torch.float16).cuda() / batch_size

    # loss fn
    loss_fn = nn.CrossEntropyLoss(weight=class_weighting)

    # get shards
    valid_shard = session.get_dataset_shard("valid")

    for epoch in range(1, num_epochs + 1):

        # set train
        model.train()
        pred_storage = []
        label_storage = []
        total_loss = 0.0
        completed_steps = 0
        mean_f1 = torchmetrics.MeanMetric().cuda()
        sum_steps = torchmetrics.SumMetric().cuda()
        
        dataloader = valid_shard.iter_torch_batches(batch_size=batch_size,
                                                    prefetch_batches=4,
                                                    local_shuffle_buffer_size=5000,
                                                    device="cuda",
                                                    dtypes={"x_cont": torch.float32,
                                                            "x_cat": torch.long,
                                                            "SYNDROME": torch.long,},
                                                    drop_last=True,)

        # torch dataloader replacement
        for i, batch in enumerate(dataloader):

            # forward
            outputs = model(x_cat=batch["x_cat"].cuda(), x_cont=batch["x_cont"].cuda())

            # y
            labels = batch["SYNDROME"].cuda()

            # loss
            loss = loss_fn(outputs, labels)

            # update
            train.torch.backward(loss)  # mixed prec loss
            optimizer.step()
            optimizer.zero_grad()

            # metrics
            total_loss += loss.detach().float().item()
            completed_steps += 1
            sum_steps.update(1)
            
            # preds
            pred = outputs.argmax(dim=1, keepdim=True)

            label_storage.append(labels.detach().cpu().numpy())
            pred_storage.append(pred.detach().cpu().numpy())
        
        # compute f1 per worker
        local_f1 = f1_score(np.concatenate(label_storage, axis=0), np.concatenate(pred_storage, axis=0), average='weighted')

        # save f1 in aggregator
        mean_f1.update(local_f1)

        # compute
        mean_f1_ddp = mean_f1.compute().item()
        sum_steps_ddp = sum_steps.compute().item()

        print(f"Epoch {epoch} | Loss {(total_loss / completed_steps)} | F1: {mean_f1_ddp} | Total Steps: {sum_steps_ddp} ")
        session.report({"loss": (total_loss / completed_steps),
                        "epoch": epoch,
                        "split": "train",
                        'F1': mean_f1_ddp},
                       checkpoint=Checkpoint.from_dict(dict(epoch=epoch, model=model.state_dict())),)


# keep the 1 checkpoint
checkpoint_config = CheckpointConfig(
    num_to_keep=1,
    checkpoint_score_attribute="F1",
    checkpoint_score_order="max"
)

# trainer -- has 24 cores total; training should use 20 cores + 1 for trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={"lr": 0.001, "batch_size": 1024, "num_epochs": 1, "dropout_rates": [0.001, 0.01], "l2": 0.01},
    datasets={"valid": valid_set},
    scaling_config=ScalingConfig(num_workers=1,
                                 use_gpu=True,  # if use_gpu=True, num_workers = num GPUs
                                 _max_cpu_fraction_per_node=0.8,
                                resources_per_worker={"GPU": 1},
                                trainer_resources={"CPU": 1}
                                 ),  
    run_config=RunConfig(checkpoint_config=checkpoint_config),
    dataset_config=DataConfig(datasets_to_split=["valid"]),
)


# tune
search_space = {
    "lr": tune.loguniform(1e-4, 5e-1),
    "l2": tune.choice([0.01, 0.05, 0.1, 0.001]),
    "dropout_rates": tune.choice([[0.001, 0.01],
                                  [0.01, 0.01],
                                  [0.05, 0.01],
                                  [0.05, 0.05],
                                  [0.01, 0.01],]),
    "batch_size": tune.choice([128, 256, 512, 1024]),
    "num_hidden": tune.choice(
        [
            [250, 250],
            [500, 250],
            [1000, 500],
            [2000, 1000],
            [750, 500],
        ]), }

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='F1',
    mode='max',
    max_t=100,
    grace_period=10,
    reduction_factor=3,
    brackets=1,
)


stopping_criteria = {"training_iteration": 1 }
tuner = Tuner(trainable=trainer,
              param_space={"train_loop_config": search_space},
              tune_config=TuneConfig(num_samples=25,  # n trials
                                     scheduler=asha_scheduler,
                                     ),
               run_config=RunConfig(storage_path="./results",
                                    stop=stopping_criteria,
                                    verbose=1,
                                    name="asynchyperband_test")
              )

# fit tune
result_grid = tuner.fit()

# get results
result_grid.get_best_result()

# returns a pandas dataframe of all reported results
col_set = ['loss', 'F1'] + result_grid.get_dataframe().filter(regex='config/train_loop').columns.tolist()

print(result_grid.get_dataframe().sort_values('F1', ascending=False).head()[col_set].head(5).to_dict())

# write
with open(aml_context.output_datasets['output1'] + '/best_df.pickle', 'wb') as handle:
    pickle.dump(result_grid.get_dataframe(), handle, protocol=pickle.HIGHEST_PROTOCOL)
# tensorboard --logdir=./results/my_experiment
