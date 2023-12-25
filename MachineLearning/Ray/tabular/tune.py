import numpy as np
from azureml.core import Run
import pickle
import ray
import ray.train
from ray.train import DataConfig, ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer
import pyarrow.compute as pc
from accelerate.utils import set_seed
import argparse
from pathlib import Path
from accelerate import Accelerator
import math
import tqdm
from ray.data.preprocessors import Chain, Concatenator
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import time, datetime
import os
import tempfile
from datetime import timedelta
import evaluate as hf_evaluate
from accelerate import InitProcessGroupKwargs
from ray.air.config import (
    CheckpointConfig,
    DatasetConfig,
    RunConfig,
    ScalingConfig,
)
import mlflow
from ray.tune import Tuner
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
np.set_printoptions(suppress=True)



# for NVLINK
#os.environ['NCCL_P2P_LEVEL'] = 'NVL'

# for no NVLINK
#os.environ['NCCL_P2P_DISABLE'] = '1'




def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mx",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )

    parser.add_argument(
        "--batch_size_per_device",
        type=int,
        default=1024,
        help="Batch size to use per device.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Grad accumulation steps",
    )
    parser.add_argument(
        "--eval_batch_size_per_device",
        type=int,
        default=1024,
        help="Batch size to use per device (For evaluation).",
    )
    parser.add_argument(
        "--num_devices", type=int, default=1, help="Number of devices to use."
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers to use."
    )    
    parser.add_argument(
        "--seed", type=int, default=1, help="Seed."
    ) 
    parser.add_argument(
        "--num_continuous", type=int, default=150, help="Num continuous features."
    )  
    parser.add_argument(
        "--num_classes", type=int, default=3, help="Num labels."
    )   
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--output_dir", type=str, default='/mnt/c/Users/afogarty/Desktop/RandomizedTest/checkpoints', help="Output dir."
    )
    parser.add_argument(
        "--label_col", type=str, default='', help="label col"
    )    
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate to use.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")

    args = parser.parse_args()

    return args


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



def evaluate(model, valid_ds, loss_fn, accelerator, eval_steps, config, ds_kwargs):
    model.eval()
    losses = []

    print(f"Found these config: {config} " )

    # load loader
    eval_loader = valid_ds.iter_torch_batches(batch_size=config['eval_batch_size_per_device'], **ds_kwargs)

    # Load metrics
    metric = hf_evaluate.load("f1")

    for step, batch in tqdm.tqdm(enumerate(eval_loader), total=eval_steps):
        with torch.inference_mode():
            # forward
            outputs = model(x_cat=batch["x_cat"], x_cont=batch["x_cont"])

        # preds
        predictions = outputs.argmax(dim=-1)

        # loss
        labels = batch['labels']
        loss = loss_fn(outputs, labels)

        # collect losses and preds
        losses.append(accelerator.gather_for_metrics(loss.repeat(config['eval_batch_size_per_device'])))

        predictions, references = accelerator.gather_for_metrics((predictions, labels))
        metric.add_batch(predictions=predictions, references=references,)

    # cat
    losses = torch.cat(losses)

    # compute f1
    eval_metric = metric.compute(average=None)

    accelerator.print(
                        f"[ F1 Results {eval_metric}  ] "
                    )

    try:
        eval_loss = torch.mean(losses).item()
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity, eval_loss, eval_metric


# train loop
def train_loop_per_worker(config):
    print("training_function called")

    # Train has a bug somewhere that causes ACCELERATE_TORCH_DEVICE to not be set
    # properly on multi-gpu nodes
    cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = cuda_visible_device[local_rank]
    os.environ["ACCELERATE_TORCH_DEVICE"] = f"cuda:{device_id}"
        
    # init Azure ML run context data
    aml_context = Run.get_context()

    # Initialize accelerator
    accelerator = Accelerator(
        deepspeed_plugin=None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mx,
    )
    
    # set seed
    set_seed(config['seed'])

    # train_ds is the local shard for this model
    train_ds = ray.train.get_dataset_shard("train")
    valid_ds = ray.train.get_dataset_shard("valid")


    def collate_fn(batch: dict[str, np.ndarray]) -> dict:
        return {
            "x_cont": torch.stack(
            [torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items() if k.startswith('INPUT_') and k not in cat_cols],
            axis=1
        ).to(accelerator.device),

        "x_cat":  torch.stack(
            [torch.as_tensor(v, dtype=torch.int32) for k, v in batch.items() if k in cat_cols],
            axis=1
        ).to(accelerator.device),

        "labels": torch.stack(
            [torch.as_tensor(v, dtype=torch.int64) for k, v in batch.items() if k == 'SYNDROME_SI'],
            axis=0
        ).to(accelerator.device).squeeze(0)
        }

    # cat cols
    EMBEDDING_TABLE_SHAPES = {
                            'INPUT__YML__testname_SI': (578, 75),
                            'INPUT__REG__m_ddr_speed_variant_SI': (6, 25),
                            'INPUT__REG__m_hazard_type_SI': (7, 25),
                            'INPUT__REG__m_cmd_merge_mode_0__SI': (5, 25),
                            'INPUT__REG__m_cmd_merge_mode_1__SI': (5, 25),
                            'INPUT__YML___m_ddrmctop_reg_block_trace_pa_fltr_cmd_0_reg_OPCODE_config_SI': (3, 25),
                            'INPUT__YML___m_ddrmctop_reg_block_trace_pa_fltr_cmd_1_reg_OPCODE_config_SI': (3, 25),
                            'INPUT__YML___m_ddrmctop_reg_block_trace_pa_fltr_cmd_2_reg_OPCODE_config_SI': (3, 25),
                            'INPUT__YML___m_ddrmctop_reg_block_trace_pa_fltr_cmd_3_reg_OPCODE_config_SI': (3, 25),
                            }

    # set model params
    NUM_CLASSES = args.num_classes
    EMBEDDING_DROPOUT_RATE = 0.01
    DROPOUT_RATES = [0.05, 0.05, 0.00]
    HIDDEN_DIMS = list(config['HIDDEN_DIMS'])
    NUM_CONTINUOUS = args.num_continuous

    # cat cols
    cat_cols = ['INPUT__YML__testname_SI',
                'INPUT__REG__m_ddr_speed_variant_SI',
                'INPUT__REG__m_hazard_type_SI',
                'INPUT__REG__m_cmd_merge_mode_0__SI',
                'INPUT__REG__m_cmd_merge_mode_1__SI',
                'INPUT__YML___m_ddrmctop_reg_block_trace_pa_fltr_cmd_0_reg_OPCODE_config_SI',
                'INPUT__YML___m_ddrmctop_reg_block_trace_pa_fltr_cmd_1_reg_OPCODE_config_SI',
                'INPUT__YML___m_ddrmctop_reg_block_trace_pa_fltr_cmd_2_reg_OPCODE_config_SI',
                'INPUT__YML___m_ddrmctop_reg_block_trace_pa_fltr_cmd_3_reg_OPCODE_config_SI']

    # init model
    model = MLP(embedding_table_shapes=EMBEDDING_TABLE_SHAPES,
                num_continuous=NUM_CONTINUOUS,
                emb_dropout=EMBEDDING_DROPOUT_RATE,
                layer_hidden_dims=HIDDEN_DIMS,
                layer_dropout_rates=DROPOUT_RATES,
                num_classes=NUM_CLASSES
        )

    class_weights = np.array([1.37060329e-02, 1.02219155e+00, 1.30289178e+00, 5.36079614e+00,
       8.27647889e+00, 4.66976689e+00, 4.34910918e+00, 1.47928851e+01,
       1.04163098e+01, 1.30806462e+01, 2.14153581e+01, 1.35959626e+01,
       2.45488581e+01, 2.52821730e+01, 1.37548091e+01, 1.71960018e+01,
       3.23873632e+01, 2.70199151e+01, 3.42332613e+01, 2.33320805e+01,
       4.62815528e+01, 3.60149549e+01, 4.17275284e+01, 8.23564271e+01,
       5.85614500e+01, 9.17475373e+01, 2.10937144e+01, 7.05623599e+01,
       9.75429441e+01, 7.67224020e+01, 1.06070969e+02, 9.03475717e+01,
       6.01505562e+01, 1.83001502e+02, 1.90162602e+02, 7.42686058e+01,
       1.14866880e+02, 1.97625062e+02, 2.79258400e+02, 3.83770140e+02,
       1.94747608e+02, 3.65473661e+02, 8.32792316e+01, 2.77331951e+02,
       5.79062941e+02, 3.10100805e+02, 4.56400057e+02, 6.19197521e+02,
       6.41484141e+02, 5.71255456e+02, 7.24807424e+02, 4.26547509e+02,
       1.14667694e+02, 8.90232773e+02, 8.90793161e+02, 9.44038703e+02,
       1.57262848e+02, 1.11654885e+03, 1.15143533e+03, 1.18598225e+03,
       1.28787224e+03, 1.32327849e+03, 2.63935022e+02, 1.44399389e+03,
       1.50480010e+03, 6.03305771e+02, 1.78136205e+03, 3.11287729e+03,
       3.19006766e+03, 3.24567435e+03, 3.30479686e+03, 3.32654917e+03,
       3.33124768e+03, 3.49583502e+03, 9.82718066e+04, 7.07557008e+06,
       2.08382826e+00])
                            
    # create weights
    class_weights = torch.tensor(class_weights, device=accelerator.device, dtype=torch.float32)
    
    # train steps
    num_train_steps_per_epoch = config['train_ds_len'] // ((accelerator.num_processes * config['batch_size_per_device']))
    num_eval_steps_per_epoch = config['eval_ds_len'] // ((accelerator.num_processes * config['batch_size_per_device']))

    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # loss fn, weighted
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # prepare
    model, optimizer = accelerator.prepare(model, optimizer)


    # data loaders
    train_dataloader = train_ds.iter_torch_batches(
        batch_size=config['batch_size_per_device'], collate_fn=collate_fn, prefetch_batches=4, drop_last=True, local_shuffle_buffer_size=config['batch_size_per_device']*2)

    # Now we train the model
    if accelerator.is_main_process:
        print("Starting training ...")
        print("Number of batches on main process", num_train_steps_per_epoch )

    for epoch in range(1, config['num_epochs'] + 1):
        fwd_time_sum, bwd_time_sum, optim_step_time_sum = 0, 0, 0
        s_epoch = time.time()
        model.train()
        loss_sum = torch.tensor(0.0).to(accelerator.device)

        for step, batch in tqdm.tqdm(enumerate(train_dataloader), total=num_train_steps_per_epoch):

            with accelerator.accumulate(model):
                s_fwd = time.time()

                # forward
                outputs = model(x_cat=batch["x_cat"], x_cont=batch["x_cont"])

                # loss
                labels = batch['labels']
                loss = loss_fn(outputs, labels)
                loss_sum += loss.item()
                e_fwd = time.time()
                fwd_time = e_fwd - s_fwd
                fwd_time_sum += fwd_time
                s_bwd = time.time()
                accelerator.backward(loss)
                e_bwd = time.time()
                bwd_time = e_bwd - s_bwd
                bwd_time_sum += bwd_time

                s_opt_step = time.time()
                optimizer.step()
                optimizer.zero_grad()
                e_opt_step = time.time()
                optim_step_time_sum += e_opt_step - s_opt_step

            if accelerator.is_main_process:
                accelerator.print(
                    f"[epoch {epoch} step {step}] "
                    f"loss: {loss.item()} step-time: {e_opt_step - s_fwd}"
                )


            # wait
            accelerator.wait_for_everyone()
            aggregated_loss = torch.mean(accelerator.gather(loss[None])).item()

        e_epoch = time.time()
        accelerator.print("Train time per epoch: ", e_epoch - s_epoch)

        eval_s_epoch = time.time()
        print("Running evaluation ...")
        print("Waiting ...")
        accelerator.wait_for_everyone()
        perplex, eloss, ef1 = evaluate(
            model=model,
            valid_ds=valid_ds,
            loss_fn=loss_fn,
            accelerator=accelerator,
            eval_steps=num_eval_steps_per_epoch,
            config=config,
            ds_kwargs={"collate_fn": collate_fn}
        )

        avg_eval_f1 = np.mean(ef1['f1'])
        accelerator.print("Eval result loss", eloss)
        accelerator.print("Eval perplex", perplex)
        accelerator.print("Eval F1", avg_eval_f1)
        
        eval_e_epoch = time.time()
        accelerator.print("Eval time per epoch: ", eval_e_epoch - eval_s_epoch)
        accelerator.print("avg fwd time: ", fwd_time_sum / (step + 1))
        accelerator.print("avg bwd time: ", bwd_time_sum / (step + 1))
        accelerator.print("avg opt step time: ", optim_step_time_sum / (step + 1))

        metrics = {
            "epoch": epoch,
            "avg_train_loss_epoch": loss_sum.item() / (step + 1),
            "eval_f1": avg_eval_f1,
            "eval_loss": eloss
        }


        ray.train.report(metrics)


if __name__ == '__main__':

    # get args
    args = parse_args()

    if not ray.is_initialized():
        # init a driver
        ray.init(
                 runtime_env={
            "env_vars": {
                "RAY_AIR_LOCAL_CACHE_DIR": args.output_dir,
                "RAY_memory_usage_threshold": ".95"
            },
        }
    )
        print('ray initialized')


    # cluster available resources which can change dynamically
    print(f"Ray cluster resources_ {ray.cluster_resources()}")

    # update the config with args so that we have access to them.
    config = vars(args)

    # load data from input
    print(f'Getting data...')
    aml_context = Run.get_context()
    data_path = aml_context.input_datasets['train_files']

    ctx = ray.data.context.DatasetContext.get_current()
    ctx.use_streaming_executor = True
    ctx.execution_options.verbose_progress = True

    train_set = ray.data.read_parquet(data_path, ray_remote_args={"num_cpus": 0.25},
                                filter=pc.field('test_split').isin(['train'])) \
                                .drop_columns(['test_split', 'randomizedTestKey', 'condition_name_SI'])

    valid_set = ray.data.read_parquet(data_path, ray_remote_args={"num_cpus": 0.25},
                                filter=pc.field('test_split').isin(['valid'])) \
                                .drop_columns(['test_split', 'randomizedTestKey', 'condition_name_SI'])
                                
                                

    # light shuffle
    train_set = train_set.randomize_block_order()
    valid_set = valid_set.randomize_block_order()



    # keep the 1 checkpoint
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="eval_loss",
        checkpoint_score_order="min"
    )

    # Dataset config
    options = ray.data.ExecutionOptions(verbose_progress=True, locality_with_output=False, resource_limits=ray.data.ExecutionResources(object_store_memory=230e9))
    dataset_config = DataConfig(datasets_to_split=["train", "valid"], execution_options=options)

    strategy = "PACK"  # default is PACK; was SPREAD
    config['train_ds_len'] = train_set.count()
    config['eval_ds_len'] = valid_set.count()


  
    # Hyperparameters to start with
    search_alg = HyperOptSearch(points_to_evaluate=None)
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)  # trade off b/w optimization and search space

    # Parameter space
    param_space = {
        "train_loop_config": {
            "lr": tune.choice([1e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 1e-6]),
            "weight_decay": tune.loguniform(5e-4, 1e-1),
            "batch_size_per_device": tune.choice([
                                                  1024,
                                                  1024*2,
                                                  1024*3,
                                                  1024*4,
                                                  8192*1,
                                                  8192*2,
                                                  8192*3,
                                                  8192*4
                                                  ]),
            "HIDDEN_DIMS": tune.choice(
                [
                    (10000, 5000, 500),
                    (2000, 2000, 2000),
                    (20000, 1000, 500),
                    (20000, 10000, 2500),
                    (5000, 1000, 250),
                    # (10000, 500),
                    # (1000, 100),
                    # (5000, 500),
                    # (5000, 5000),
                    # (10000, 10000),
                    # (20000, 10000),
                    # (2000, 1500),
                ]),
        }
    }

    # Scheduler
    scheduler = AsyncHyperBandScheduler(
        time_attr='epoch',
        max_t=2,  # max epoch (<time_attr>) per trial
        grace_period=1,  # min epoch (<time_attr>) per trial
    )

    # Tune config
    tune_config = tune.TuneConfig(
        metric="eval_f1",
        mode="max",
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=16,
    )


    # trainer -- 32 GPUs, 160 Cores
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config,
        datasets={"train": train_set, "valid": valid_set},
        dataset_config=dataset_config,
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=True, placement_strategy=strategy, resources_per_worker={"GPU": args.num_devices,}),
        run_config=RunConfig(checkpoint_config=checkpoint_config),
        )


    # Tuner
    tuner = Tuner(
        trainable=trainer,
        run_config=None,
        param_space=param_space,
        tune_config=tune_config,
    )

    # Tune
    results = tuner.fit()
    best_trial = results.get_best_result(metric="eval_f1", mode="max")
    d = {
            "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
            "params": best_trial.config["train_loop_config"],
        #"metrics": utils.dict_to_list(best_trial.metrics_dataframe.to_dict(), keys=["epoch", "train_loss", "val_loss"]),
        }
    
    print(d)
    print("Training complete!")


