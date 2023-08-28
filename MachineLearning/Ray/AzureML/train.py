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

os.environ['RAY_DATA_DISABLE_PROGRESS_BARS'] = "1"


# repartition for parallel
num_cpus = psutil.cpu_count(logical=False)
num_partitions = min(num_cpus - 2, 32)
print(f'Num partitions: {num_partitions}')
print(f'Num CPUs: {num_cpus}')


# set data context
ctx = ray.data.DataContext.get_current()
ctx.use_push_based_shuffle = True
ctx.use_streaming_executor = True
ctx.execution_options.locality_with_output = True
ctx.execution_options.verbose_progress = False

if not ray.is_initialized():
    # init a driver
    ray.init()
    print('ray initialized')


# cluster available resources which can change dynamically
print(f"Ray cluster resources_ {ray.cluster_resources()}")

# init Azure ML run context data
aml_context = Run.get_context()


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


shap_cols = [
       'INPUT__REG__bus_addr_cntl_reg_14_14___EN_ADDR_RANGE_CHECK',
       'INPUT__REG__m_weight_prefetch',
       'INPUT__YML__uvm_test_top_m_ddrss_base_config_m_enable_memory_preloading',
       'INPUT__YML____m_mc_address_config_0__m_addr_table_size',
       'INPUT__YML____m_ddrss_base_tb_config_m_csr_frontdoor_access_enabled',
       'INPUT__REG__m_weight_rw', 'INPUT__REG__m_refresh_enabled',
       'INPUT__REG__m_weight_wr_fc',
       'INPUT__YML__uvm_test_top_m_ddrss_base_tb_config_m_perf_config_m_num_actions',
       'INPUT__YML____m_ddrss_part_config_m_ddr_speed_variant',
       'INPUT__YML___m_mc_base_config_0__m_error_injection_config_m_ca_parity_error_injection_single_type_cmd_type',
       'INPUT__REG__fecq_cmd_dispatch_limit_config_reg_5_0___FECQ_BECQ_CMD_THRESH',
       'INPUT__REG__fecq_cmd_dispatch_limit_config_reg_14_8___FECQ_BEWDB_CMD_THRESH',
       'INPUT__REG__m_weight_rd_fc',
       'INPUT__YML__uvm_test_top_m_ddrss_base_tb_config_m_random_enable',
       'INPUT__REG__m_ddr_size',
       'INPUT__REG__ue_config_reg_2_0___UE_MAX_RETRY',
       'INPUT__YML___part_config_m_dfi_freq_ratio'
       ]

cat_cols = ['INPUT__REG__m_cmd_merge_mode_0_',
            'INPUT__REG__m_cmd_merge_mode_1_',
            'INPUT__REG__m_hazard_type',
            'INPUT__YML__testname',
            'INPUT__REG__m_ddr_speed_variant']

filter_cols = shap_cols + cat_cols + ['SYNDROME', 'test_split']

# load data from input -- run.input_datasets['RandomizedTest'] is a locally mounted path... /mnt/..; faster than abfs route
print(f'Getting data...')
train_set = ray.data.read_parquet(aml_context.input_datasets['RTE'],
                            use_threads=True,
                            #columns=filter_cols,
                            filter=pc.field('test_split').isin(['train'])) \
                            .drop_columns(['test_split'])

# limit size of train set
#_, train_set = train_set.train_test_split(0.3)

valid_set = ray.data.read_parquet(aml_context.input_datasets['valid'],
                            use_threads=True,
                            #columns=filter_cols,
                            filter=pc.field('test_split').isin(['valid']),   
                            ).drop_columns(['test_split'])


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
DROPOUT_RATES = [0.01, 0.01]
HIDDEN_DIMS = [2000, 1000]
NUM_CONTINUOUS = 223

# cat cols
cat_cols = ['INPUT__REG__m_cmd_merge_mode_0_',
            'INPUT__REG__m_cmd_merge_mode_1_',
            'INPUT__REG__m_hazard_type',
            'INPUT__YML__testname',
            'INPUT__REG__m_ddr_speed_variant']

# Create a preprocessor to concatenate
preprocessor = Chain(Concatenator(include=cat_cols, dtype=np.int32, output_column_name=['x_cat']),
                     Concatenator(exclude=["SYNDROME", "x_cat"], dtype=np.float32, output_column_name=['x_cont'])
                     )

# apply preprocessor
train_set = preprocessor.fit_transform(train_set)
valid_set = preprocessor.transform(valid_set)

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    optimizer = train.torch.prepare_optimizer(optimizer)

    class_weights = np.array([  0.01646575,   0.53236018,   1.15584174,   1.54778856,
                                    2.26079437,   3.46198553,   3.67448811,   4.2577765 ,
                                    5.08383489,   7.06892775,   7.47755662,  13.582093  ,
                                    14.3027387 ,  15.5879883 ,  16.5108291 ,  18.0089051 ,
                                    19.5361366 ,  22.3280369 ,  22.5298771 ,  22.7677123 ,
                                    23.300391  ,  24.9419628 ,  25.3140529 ,  25.4673478 ,
                                    36.9054405 ,  37.1935    ,  37.2162386 ,  37.6821687 ,
                                    39.0215551 ,  41.2122412 ,  44.8266579 ,  46.3294657 ,
                                    47.8696219 ,  48.9050825 ,  49.5832548 ,  63.2266322 ,
                                    65.2006657 ,  69.0364339 ,  71.0322071 ,  71.2441939 ,
                                    85.2316868 ,  97.177797  , 101.903497  , 102.448     ,
                                103.943655  , 113.050005  , 124.341728  , 135.714193  ,
                                148.983361  , 150.140835  , 158.383248  , 169.131351  ,
                                169.697126  , 184.53839   , 184.874653  , 187.133727  ,
                                193.76017   , 198.417867  , 200.047984  , 205.449212  ,
                                342.473145  , 343.390785  , 349.251873  , 374.382543  ,
                                402.21078   , 405.763049  , 437.239012  ])
       
    # create weights
    class_weights = torch.tensor(class_weights, device=device, dtype=torch.float16) / batch_size

    # loss fn, weighted
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # get shards
    train_shard = session.get_dataset_shard("train")
    valid_shard = session.get_dataset_shard("valid")

    for epoch in range(1, num_epochs + 1):

        # set train
        model.train()
        total_loss = 0.0
        completed_steps = 0
        sum_steps = torchmetrics.SumMetric().to(device)
        sum_loss = torchmetrics.SumMetric().to(device)
        cat_metric_preds = torchmetrics.CatMetric().to(device)
        cat_metric_labels = torchmetrics.CatMetric().to(device)
        t0 = time.time()

        # torch dataloader replacement
        for i, batch in enumerate(train_shard.iter_torch_batches(
            batch_size=batch_size,
            prefetch_batches=4,
            local_shuffle_buffer_size=10000,
            device="cuda",
            dtypes={
                    "x_cont": torch.float32,
                    "x_cat": torch.long,
                    "SYNDROME": torch.long,
                },
            drop_last=True,
        )):

            # forward
            outputs = model(x_cat=batch["x_cat"].to(device), x_cont=batch["x_cont"].to(device))

            # loss
            labels = batch['SYNDROME'].to(device)
            loss = loss_fn(outputs, labels)

            # update
            train.torch.backward(loss)  # mixed prec loss
            optimizer.step()
            optimizer.zero_grad()

            # metrics
            total_loss += loss.detach().float().item()
            completed_steps += 1

            # update metrics
            sum_loss.update(loss)
            sum_steps.update(1)            

            # report progress
            if i % 100 == 0 and i != 0:
                print(f"Training: Epoch: {epoch} | Step {i} | Loss: {(total_loss / completed_steps)} | Time: {time.time() - t0:.2f}")
                # reset
                t0 = time.time()

        # compute metrics
        ddp_steps = sum_steps.compute().item()
        ddp_loss = sum_loss.compute().item()
        avg_loss = (ddp_loss / ddp_steps)

        print(f"Train: Epoch {epoch} Complete! | Loss: {avg_loss} | Total Steps: {ddp_steps}")

        # reset  
        sum_steps.reset()
        sum_loss.reset()
        cat_metric_preds.reset()
        cat_metric_labels.reset()

        # evaluating
        model.eval()

        # torch dataloader replacement
        for i, batch in enumerate(valid_shard.iter_torch_batches(
            batch_size=batch_size,
            prefetch_batches=4,
            local_shuffle_buffer_size=10000,
            device="cuda",
            dtypes={
                    "x_cont": torch.float32,
                    "x_cat": torch.long,
                    "SYNDROME": torch.long,
                },
            drop_last=True,
        )):

            with torch.no_grad():
                
                # forward
                outputs = model(x_cat=batch["x_cat"].to(device), x_cont=batch["x_cont"].to(device))

                # loss
                labels = batch['SYNDROME'].to(device)
                loss = loss_fn(outputs, labels)

            # metrics
            total_loss += loss.detach().float().item()
            completed_steps += 1

            # update metrics
            sum_loss.update(loss)
            sum_steps.update(1)            

            # preds
            preds = outputs.argmax(dim=1, keepdim=True)

            # update metrics
            cat_metric_preds.update(preds)
            cat_metric_labels.update(labels)

            # report progress
            if i % 100 == 0 and i != 0:
                print(f"Validating: Epoch: {epoch} | Step {i} | Loss: {(total_loss / completed_steps)} | Time: {time.time() - t0:.2f}")
                # reset
                t0 = time.time()

        # compute metrics
        ddp_steps = sum_steps.compute().item()
        ddp_loss = sum_loss.compute().item()
        ddp_preds = cat_metric_preds.compute()
        ddp_labels = cat_metric_labels.compute()
        f1 = torchmetrics.classification.MulticlassF1Score(num_classes=67, average=None).to(device)
        ddp_f1 = torch.mean(f1(ddp_preds.view(-1), ddp_labels, )).item()
        avg_loss = (ddp_loss / ddp_steps)
        
        print(f"Validating: Epoch {epoch} Complete! | Loss: {avg_loss} | Total Steps: {ddp_steps} | Valid F1: {ddp_f1} | LR: {lr}")

        # epoch report
        session.report({"loss": avg_loss,
                        "total_steps": ddp_steps,
                        "epoch": epoch,
                        "split": 'valid',
                        'F1': ddp_f1,
                        "batch_size": batch_size,
                        "lr": lr,
                        },
                    #    checkpoint=Checkpoint.from_dict(
                    #        dict(epoch=epoch,
                    #             model=model.state_dict())
                    #             ),
                        checkpoint=TorchCheckpoint.from_model(model),
                    )


# keep the 1 checkpoint
checkpoint_config = CheckpointConfig(
    num_to_keep=1,
    checkpoint_score_attribute="F1",
    checkpoint_score_order="max"
)


# data opts
options = DataConfig.default_ingest_options()
options.resource_limits.object_store_memory = 35e+10
options.locality_with_output = True
options.use_push_based_shuffle = True
options.use_streaming_executor = True
options.verbose_progress = False

# trainer -- has 24 cores total; training should use 20 cores + 1 for trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={"lr": 0.00053, "batch_size": 2048, "num_epochs": 4},
    datasets={"train": train_set, "valid": valid_set},
    scaling_config=ScalingConfig(num_workers=4,
                                 use_gpu=True,  # if use_gpu=True, num_workers = num GPUs
                                 _max_cpu_fraction_per_node=0.8,
                                resources_per_worker={"GPU": 1},
                                trainer_resources={"CPU": 1}
                                 ), 
    run_config=RunConfig(checkpoint_config=checkpoint_config),
    dataset_config=DataConfig(datasets_to_split=["train" ], execution_options=options)
)


print(f"Starting to train with a dataset of size: {train_set.count()}!")

# fit
result = trainer.fit()


# get best state dict
best_state_dict = result.get_best_checkpoint(metric='F1', mode='max').to_dict()

# write
with open(aml_context.output_datasets['output1'] + '/best_ckpt_latest.pickle', 'wb') as handle:
    pickle.dump(best_state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Training complete!")



print("Evaluating...")

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


# map against calibration_set
out = valid_set.map_batches(TorchPredictor(model=model,
                             state_dict=best_state_dict['model'].state_dict()),
                             compute=ray.data.ActorPoolStrategy(size=4),
                             num_gpus=1,
                             batch_size=1024,
                             )

# score
pred_store = []
true_store = []

for batch in out.iter_batches(batch_size=1024):
    pred_store.append(batch['y_pred'])
    true_store.append(batch['y_true'])

total_f1 = np.mean(f1_score(y_true=np.concatenate(pred_store, axis=0), y_pred=np.concatenate(true_store, axis=0), average=None))

print(f"F1 Score: {total_f1}")

