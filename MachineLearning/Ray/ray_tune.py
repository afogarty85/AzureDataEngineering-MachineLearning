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
import torchmetrics
import ray.experimental.tqdm_ray
from ray.train import DataConfig
np.set_printoptions(suppress=True)



os.environ['RAY_DATA_DISABLE_PROGRESS_BARS'] = "1"

if not ray.is_initialized():
    # init a driver
    ray.init(num_gpus=1)
    print('ray initialized')


# set data context


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



ctx = ray.data.DataContext.get_current()
ctx.use_streaming_executor = True
ctx.execution_options.locality_with_output = True
ctx.execution_options.verbose_progress = False

# load data from input -- run.input_datasets['RandomizedTest'] is a locally mounted path... /mnt/..; faster than abfs route
print(f'Getting data...')
valid_set = ray.data.read_parquet('data/RTE/part-00019-e47453b1-4802-4d45-ad60-50804504db95.c000.snappy.parquet',
                                  use_threads=True,
                                  filter=pc.field('test_split').isin(['valid']),
                                  ) \
                                .drop_columns(['test_split', 'CYCLES'])


# cat cols
cat_cols = ['INPUT__REG__m_cmd_merge_mode_0_',
            'INPUT__REG__m_cmd_merge_mode_1_',
            'INPUT__REG__m_hazard_type',
            'INPUT__YML__testname',
            'INPUT__REG__m_ddr_speed_variant']

# # num cols
# continuous_features = [c for c in train_set.columns() if c not in ['test_split', 'SYNDROME', 'INPUT__YML__testname'] + cat_cols]

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


# feature cols
continuous_cols = [c for c in valid_set.columns() if c not in ['SYNDROME', 'test_split', 'CYCLES']]

# Create a preprocessor to concatenate
preprocessor = Chain(
                    Concatenator(include=cat_cols, dtype=np.int32, output_column_name=['x_cat']),
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
                                      parallel_strategy='ddp',  # ddp / fsdp / None
                                      )

    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = train.torch.prepare_optimizer(optimizer)

    class_weights = np.array([4.39810732e-05, 1.42196850e-03, 3.08732809e-03, 4.13424341e-03,
                            6.03872808e-03, 9.24718738e-03, 9.81479553e-03, 1.13727966e-02,
                            1.35792520e-02, 1.88815635e-02, 1.99730376e-02, 3.62786495e-02,
                            3.82035408e-02, 4.16365254e-02, 4.41014931e-02, 4.81029510e-02,
                            5.21822850e-02, 5.96396314e-02, 6.01787598e-02, 6.08140329e-02,
                            6.22368522e-02, 6.66215967e-02, 6.76154734e-02, 6.80249341e-02,
                            9.26733350e-02, 9.85768199e-02, 9.94069817e-02, 9.93462455e-02,
                            1.00651511e-01, 1.04229099e-01, 1.10080563e-01, 1.19734904e-01,
                            1.27862858e-01, 1.30628640e-01, 1.32440082e-01, 1.68882426e-01,
                            1.74155197e-01, 1.84400783e-01, 1.89731622e-01, 1.90297852e-01,
                            2.27659351e-01, 2.59568185e-01, 2.72190835e-01, 2.73645237e-01,
                            2.77640229e-01, 3.01963879e-01, 3.32124802e-01, 3.62501393e-01,
                            3.97944200e-01, 4.01035886e-01, 4.23051906e-01, 4.51760783e-01,
                            4.53272005e-01, 4.92913983e-01, 4.93812163e-01, 4.99846296e-01,
                            5.17545954e-01, 5.29986962e-01, 5.34341111e-01, 5.48768140e-01,
                            9.14767932e-01, 9.17219010e-01, 9.32874355e-01, 1.00000000e+00])
    # create weights
    class_weighting = torch.tensor(class_weights, dtype=torch.float16).cuda()

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
        nb_examples = 0
        mean_f1 = torchmetrics.MeanMetric().cuda()
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
            nb_examples += labels.size(0)

            # report progress
            if i % 25 == 0 and i != 0:
                print(f"Step: {i} | Loss: {(total_loss / completed_steps)} ")

            # preds
            pred = outputs.argmax(dim=1, keepdim=True)

            label_storage.append(labels.detach().cpu().numpy())
            pred_storage.append(pred.detach().cpu().numpy())
        
        # compute f1 per worker
        local_f1 = f1_score(np.concatenate(label_storage, axis=0), np.concatenate(pred_storage, axis=0), average='weighted')

        # save f1 in aggregator
        mean_f1(local_f1)

        # compute
        mean_f1_ddp = mean_f1.compute().item()

        print(f"Epoch {epoch} | Loss {(total_loss / completed_steps)} | F1: {mean_f1_ddp} ")
        session.report({"loss": (total_loss / completed_steps),
                        "epoch": epoch, "split": "train", 'F1': mean_f1_ddp},
                       checkpoint=Checkpoint.from_dict(dict(epoch=epoch, model=model.state_dict())),)


# keep the 1 checkpoint
checkpoint_config = CheckpointConfig(
    num_to_keep=1,
    checkpoint_score_attribute="F1",
    checkpoint_score_order="max"
)


# data opts
options = DataConfig.default_ingest_options()
options.resource_limits.object_store_memory = 4e+11
options.locality_with_output = False
options.use_push_based_shuffle = True
options.use_streaming_executor = True
options.verbose_progress = True


# trainer -- has 24 cores total; training should use 20 cores + 1 for trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={"num_hidden": (250, 250)},
    datasets={"valid": valid_set},
    scaling_config=ScalingConfig(num_workers=1,
                                 use_gpu=True,  # if use_gpu=True, num_workers = num GPUs
                                 _max_cpu_fraction_per_node=0.8,
                                resources_per_worker={"GPU": 1},
                                # trainer_resources={"CPU": 1}
                                 ),  
    run_config=RunConfig(checkpoint_config=checkpoint_config),
    dataset_config=DataConfig(datasets_to_split=["valid"], execution_options=options),
)



asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='F1',
    mode='max',
    max_t=100,
    grace_period=10,
    reduction_factor=3,
    brackets=1,
)


from ray.tune.search.hyperopt import HyperOptSearch

ConcurrencyLimiter
search_space = {"num_hidden": tune.choice([ [250, 250], ])}
current_best_params = [{'num_hidden': ([1,]),}]

hyperopt_search = HyperOptSearch(
    metric="mean_loss", mode="min",)


stopping_criteria = {"training_iteration": 1 }


tuner = Tuner(
              trainable=tune.with_resources(train_loop_per_worker, resources={"cpu": 2, "gpu": 1}),
              param_space=search_space,
              tune_config=TuneConfig(num_samples=1,
                                     max_concurrent_trials=4,
                                     scheduler=asha_scheduler,
                                     search_alg=hyperopt_search,
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
result_grid.get_dataframe().sort_values('F1', ascending=False).head()[col_set].head(3).to_dict()

# tensorboard --logdir=./results/my_experiment
