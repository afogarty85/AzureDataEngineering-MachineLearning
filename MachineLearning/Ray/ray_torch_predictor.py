import numpy as np
from azureml.core import Run
import pickle
import ray
import pyarrow.compute as pc
from ray import train, tune
from ray.air import session, Checkpoint
from ray.air.config import ScalingConfig, DatasetConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer, TorchConfig, TorchCheckpoint
from ray.train import DataConfig
from ray.data.preprocessors import Chain, BatchMapper, Concatenator, StandardScaler, MinMaxScaler
from ray.tune.tuner import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import psutil
import pandas as pd
from sklearn.metrics import f1_score
import time, datetime
import os
import torchmetrics
from ray.experimental import tqdm_ray


if not ray.is_initialized():
    # init a driver
    ray.init(num_gpus=1)
    print('ray initialized')



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




# cat cols
EMBEDDING_TABLE_SHAPES = {
                          'INPUT__REG__m_cmd_merge_mode_0_': (5, 50),
                          'INPUT__REG__m_cmd_merge_mode_1_': (5, 50),
                          'INPUT__REG__m_hazard_type': (6, 50),
                          'INPUT__YML__testname': (523, 200),
                          'INPUT__REG__m_ddr_speed_variant': (6, 50),
                          }

# set model params
NUM_CLASSES = 67
EMBEDDING_DROPOUT_RATE = 0.01
DROPOUT_RATES = [0.001, 0.01]
HIDDEN_DIMS = [3500, 2000]
NUM_CONTINUOUS = 223


# init model
model = MLP(embedding_table_shapes=EMBEDDING_TABLE_SHAPES,
        num_continuous=NUM_CONTINUOUS,
        emb_dropout=EMBEDDING_DROPOUT_RATE,
        layer_hidden_dims=HIDDEN_DIMS,
        layer_dropout_rates=DROPOUT_RATES,
        num_classes=NUM_CLASSES
        )

class TorchPredictor:
    def __init__(self, model, state_dict):
        self.model = model.cuda()
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def __call__(self, batch):
        # transform to tensor / attach to GPU
        x_cont = torch.as_tensor(batch["x_cont"], dtype=torch.float32, device="cuda")
        x_cat = torch.as_tensor(batch["x_cat"], dtype=torch.int64, device="cuda")
        labels = torch.as_tensor(batch["SYNDROME"], dtype=torch.int64, device="cuda")

        # like no_grad
        with torch.inference_mode():
            # forward and back to cpu
            logits = self.model(x_cat=x_cat, x_cont=x_cont)
            _, predicted_labels = torch.max(logits, dim=1)

            yield {
                    "y_pred": predicted_labels.cpu().numpy(),
                    "y_true": labels.cpu().numpy(),
                    }



cat_cols = ['INPUT__REG__m_cmd_merge_mode_0_',
            'INPUT__REG__m_cmd_merge_mode_1_',
            'INPUT__REG__m_hazard_type',
            'INPUT__YML__testname',
            'INPUT__REG__m_ddr_speed_variant']


valid_set = ray.data.read_parquet('/mnt/c/Users/afogarty/Desktop/newCE/data/test',
                            use_threads=True,
                            filter=pc.field('test_split').isin(['test']),   
                            ).drop_columns(['test_split'])


# load state
with open(r'best_ckpt_latest.pickle', 'rb') as fp:
    best_ckpt = pickle.load(fp)

# Create a preprocessor to concatenate
preprocessor = Chain(Concatenator(include=cat_cols, dtype=np.int32, output_column_name=['x_cat']),
                     Concatenator(exclude=["SYNDROME", "x_cat"], dtype=np.float32, output_column_name=['x_cont'])
                     )

# apply preprocessor
valid_set = preprocessor.fit_transform(valid_set)


# map against calibration_set
out = valid_set.map_batches(TorchPredictor(model=model,
                             state_dict=best_ckpt['model'].state_dict()),
                             num_gpus=1,
                             batch_size=1024,
                             )

# score batch
score_store = 0
batch_count = 0

pred_store = []
true_store = []

for batch in out.iter_batches(batch_size=1024):
    pred_store.append(batch['y_pred'])
    true_store.append(batch['y_true'])

np.mean(f1_score(y_true=np.concatenate(pred_store, axis=0), y_pred=np.concatenate(true_store, axis=0), average=None))