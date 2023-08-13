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
from sklearn.metrics import f1_score
from ray.train import DataConfig
np.set_printoptions(suppress=True)


if not ray.is_initialized():
    # init a driver
    ray.init(num_gpus=1)
    print('ray initialized')


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
train_set = ray.data.read_parquet('data/RTEDistinct',
                                  use_threads=True,
                                  filter=pc.field('test_split').isin(['valid']),
                                  ) \
    .drop_columns(['test_split'])


# num cols
continuous_features = [c for c in train_set.columns() if c not in ['test_split', 'SYNDROME', 'INPUT__YML__testname']]

# set model params
# (max_val, n_features) -- max_val needs to be count(distinct(val)) +1 we can encounter
EMBEDDING_TABLE_SHAPES = {'INPUT__YML__testname': (523, 50)}
NUM_CLASSES = 51
EMBEDDING_DROPOUT_RATE = 0.04
DROPOUT_RATES = [0.001, 0.01]
HIDDEN_DIMS = [500, 500]
NUM_CONTINUOUS = len(continuous_features)

# Create a preprocessor to concatenate
preprocessor = Concatenator(
    exclude=["SYNDROME", "INPUT__YML__testname"],
    dtype=np.float32, output_column_name=['x_cont'])

# apply preprocessor
train_set = preprocessor.fit_transform(train_set)


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
    EMBEDDING_TABLE_SHAPES = config['EMBEDDING_TABLE_SHAPES']

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    optimizer = train.torch.prepare_optimizer(optimizer)

    # loss fn
    loss_fn = nn.CrossEntropyLoss()

    # get shards
    train_shard = session.get_dataset_shard("train")

    for epoch in range(1, num_epochs + 1):

        # set train
        model.train()

        total_loss = 0.0
        completed_steps = 0
        n_correct = 0
        nb_examples = 0

        # torch dataloader replacement
        for i, batch in enumerate(train_shard.iter_torch_batches(
            batch_size=batch_size,
            prefetch_batches=5,
            local_shuffle_buffer_size=10000,
            device="cuda",
            dtypes={
                "x_cont": torch.float32,
                "INPUT__YML__testname": torch.long,
                "SYNDROME": torch.long,
            },
            drop_last=True,
        )):

            # forward
            outputs = model(x_cat=batch["INPUT__YML__testname"].cuda(), x_cont=batch["x_cont"].cuda())

            # loss
            loss = loss_fn(outputs, batch["SYNDROME"].cuda())

            # update
            train.torch.backward(loss)  # mixed prec loss
            optimizer.step()
            optimizer.zero_grad()

            # metrics
            total_loss += loss.detach().float().item()
            completed_steps += 1
            nb_examples += batch["SYNDROME"].size(0)

            # basic acc
            pred = outputs.argmax(dim=1, keepdim=True)
            n_correct += pred.eq(batch["SYNDROME"].view_as(pred)).sum().item()

        print(f"Epoch {epoch} | Loss {(total_loss / completed_steps)} | Acc: {n_correct / nb_examples} ")
        session.report({"loss": (total_loss / completed_steps),
                        "epoch": epoch, "split": "train", 'acc': n_correct / nb_examples},
                       checkpoint=Checkpoint.from_dict(dict(epoch=epoch, model=model.state_dict())),)


# keep the 1 checkpoint
checkpoint_config = CheckpointConfig(
    num_to_keep=1,
    checkpoint_score_attribute="acc",
    checkpoint_score_order="max"
)

# trainer -- has 24 cores total; training should use 20 cores + 1 for trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={"lr": 0.001, "batch_size": 1024, "num_epochs": 7, "dropout_rates": [0.001, 0.01], "EMBEDDING_TABLE_SHAPES": {'INPUT__YML__testname': (523, 50)}},
    datasets={"train": train_set},
    scaling_config=ScalingConfig(num_workers=1, use_gpu=True, ),  # if use_gpu=True, num_workers = num GPUs
    run_config=RunConfig(checkpoint_config=checkpoint_config),
    dataset_config=DataConfig(datasets_to_split=["train"]),
)


# tune
config = {
    "lr": tune.loguniform(1e-4, 5e-1),
    "EMBEDDING_TABLE_SHAPES": tune.choice(
        [{'INPUT__YML__testname': (523, 25)},
         {'INPUT__YML__testname': (523, 50)},
         {'INPUT__YML__testname': (523, 100)},
         {'INPUT__YML__testname': (523, 200)},
         {'INPUT__YML__testname': (523, 500)}]),
    "dropout_rates": tune.choice([[0.001, 0.01],
                                  [0.01, 0.01],
                                  [0.05, 0.01],
                                  [0.05, 0.05],
                                  [0.01, 0.01],]),
    "batch_size": tune.choice([64, 128, 256, 512]),
    "num_hidden": tune.choice(
        [
         [1000, 500],
         [2000, 500],
         [10000, 1000],
         [5000, 500],
         [2500, 1500],
        ]), }

scheduler = ASHAScheduler(
    time_attr='training_iteration',
    max_t=100,
    grace_period=3,
    reduction_factor=2,
    brackets=1)


resource_group = tune.PlacementGroupFactory([{"CPU": 8, "GPU": 1}])
trainable_with_resources = tune.with_resources(train_loop_per_worker, resource_group)

tuner = Tuner(trainer,
              param_space={"train_loop_config": config},
              tune_config=TuneConfig(num_samples=25,  # n trials
                                     metric="loss",
                                     mode="min",
                                     scheduler=scheduler,
                                     ),
              )

# fit tune
result_grid = tuner.fit()

# get results
result_grid.get_best_result()

# returns a pandas dataframe of all reported results
col_set = ['loss', 'acc'] + result_grid.get_dataframe().filter(regex='config/train_loop').columns.tolist()
result_grid.get_dataframe().query("split == 'train'").sort_values('loss', ascending=True).head()[col_set]

