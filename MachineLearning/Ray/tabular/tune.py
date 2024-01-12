import numpy as np
from azureml.core import Run
import pickle
import ray
import ray.train
from ray.train import DataConfig, ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer
import pyarrow.compute as pc
from accelerate.utils import set_seed
from ray.tune.search.hyperopt import HyperOptSearch
import argparse
from pathlib import Path
from accelerate import Accelerator
import math
from ray.tune.schedulers import ASHAScheduler
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
import gc
import mlflow
import evaluate as hf_evaluate
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
from ray.air.config import (
    CheckpointConfig,
    DatasetConfig,
    RunConfig,
    ScalingConfig,
)
from ray.data.preprocessors import OneHotEncoder
from itertools import zip_longest  
from collections import OrderedDict
import mlflow
from ray.tune import Tuner
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from torch.nn.utils import clip_grad_norm_  
from typing import Any, Dict, List
from tmodel import *
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
np.set_printoptions(suppress=True)

# for NVLINK
#os.environ['NCCL_P2P_LEVEL'] = 'NVL'

# for no NVLINK
#os.environ['NCCL_P2P_DISABLE'] = '1'

# init Azure ML run context data
aml_context = Run.get_context()
data_path = aml_context.input_datasets['train_files']
output_path = aml_context.output_datasets['output_dir']

os.environ['RAY_AIR_NEW_OUTPUT'] = '1'





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
        "--eval_batch_size_per_device",
        type=int,
        default=4096,
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
        "--max_concurrent", type=int, default=4, help="Num continuous features."
    )      
    parser.add_argument(
        "--num_classes", type=int, default=3, help="Num labels."
    ) 
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Num trials."
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
    args = parser.parse_args()

    return args







def evaluate(model, valid_ds, loss_fn, eval_steps, config, accelerator, ds_kwargs):
    model.eval()
    losses = []

    print(f"Found these config: {config} " )

    # load loader
    eval_loader = valid_ds.iter_torch_batches(batch_size=config['eval_batch_size_per_device'], **ds_kwargs)

    pred_store = []
    true_store = []
    prob_store = []

    for step, batch in tqdm.tqdm(enumerate(eval_loader), total=eval_steps):
        with torch.inference_mode():
            # forward
            outputs = model(x_cat=batch["x_cat"],
                                x_cont=batch["x_cont"],
                                x_binary=batch["x_binary"],
                                x_ohe=batch['x_ohe'],
                                x_invariant=batch['x_invariant'],
                                cat_masks=batch["cat_masks"],
                                cont_masks=batch['cont_masks'],
                                binary_masks=batch['binary_masks'],
                                invariant_masks=batch['invariant_masks'])
            
            assert not torch.isnan(outputs).any(), "NaNs in outputs"  

        # preds
        predictions = outputs.argmax(dim=-1)
        probabilities = F.softmax(outputs, dim=-1) 
        positive_class_probabilities = probabilities[:, 1]  

        # loss
        labels = batch['labels']

        # Since your network outputs logits for two classes, you need to one-hot encode labels  
        labels_one_hot = F.one_hot(labels, num_classes=2).float()  
    
        # Apply FocalLoss  
        loss = loss_fn(outputs, labels_one_hot)   

        # collect losses and preds
        losses.append(accelerator.gather_for_metrics(loss.repeat(config['eval_batch_size_per_device'])))

        predictions, references = accelerator.gather_for_metrics((predictions, labels))
        positive_class_probabilities, _ = accelerator.gather_for_metrics((positive_class_probabilities, labels))


        pred_store.append(predictions)
        true_store.append(references)
        prob_store.append(positive_class_probabilities)


    # cat
    losses = torch.cat(losses)

    # compute precision
    my_recall = recall_score(torch.cat(true_store).cpu(), torch.cat(pred_store).cpu())
    my_precision = precision_score(torch.cat(true_store).cpu(), torch.cat(pred_store).cpu())
    my_f1 = f1_score(torch.cat(true_store).cpu(), torch.cat(pred_store).cpu())
    my_ap = average_precision_score(torch.cat(true_store).cpu(), torch.cat(prob_store).cpu())

    accelerator.print(
                        f"[ Recall {my_recall}  | Precision {my_precision} | F1 {my_f1} | AP {my_ap} ] "
                    )
    try:
        eval_loss = torch.mean(losses).item()
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity, eval_loss, my_recall, my_precision, my_f1, my_ap


# train loop
def train_loop_per_worker(config):
    print("training_function called")

    # Train has a bug somewhere that causes ACCELERATE_TORCH_DEVICE to not be set
    # properly on multi-gpu nodes
    cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = cuda_visible_device[local_rank]
    os.environ["ACCELERATE_TORCH_DEVICE"] = f"cuda:{device_id}"
        
    # Initialize accelerator
    accelerator = Accelerator(
        deepspeed_plugin=None,
        gradient_accumulation_steps=1,
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
                [torch.as_tensor(v, dtype=torch.float32)
                 for k, v in batch.items() if k in cont_cols],
                axis=1
            ).to(accelerator.device),

            "x_cat":  torch.stack(
                [torch.as_tensor(batch[k], dtype=torch.int64) for k in cat_cols],
                axis=1
            ).to(accelerator.device),

            "x_binary":  torch.stack(
                [torch.as_tensor(batch[k], dtype=torch.int64) for k in binary_cols],
                axis=1
            ).to(accelerator.device),

            "x_invariant":  torch.stack(
                [torch.as_tensor(batch[k], dtype=torch.int64) for k in invariant_cols],
                axis=1
            ).to(accelerator.device),

            # x.startswith('INPUT__YML__testname')
            "x_ohe": torch.stack(
                [torch.as_tensor(batch[k], dtype=torch.int64) for k in ohe_cols],
                axis=1
            ).to(accelerator.device),

            "cat_masks": torch.stack(
                [torch.as_tensor(batch[k], dtype=torch.int64) for k in ['cat_masks']],
                axis=1
            ).to(accelerator.device).squeeze(1),

            "cont_masks": torch.stack(
                [torch.as_tensor(batch[k], dtype=torch.int64) for k in ['cont_masks']],
                axis=1
            ).to(accelerator.device).squeeze(1),

            "binary_masks": torch.stack(
                [torch.as_tensor(batch[k], dtype=torch.int64) for k in ['binary_masks']],
                axis=1
            ).to(accelerator.device).squeeze(1),

            "invariant_masks": torch.stack(
                [torch.as_tensor(batch[k], dtype=torch.int64) for k in ['invariant_masks']],
                axis=1
            ).to(accelerator.device).squeeze(1),


            "labels": torch.stack(
                [torch.as_tensor(batch[k], dtype=torch.int64) for k in ['SYNDROME_SI']],
                axis=1
            ).to(accelerator.device).squeeze(1),
        }


    EMBEDDING_TABLE_SHAPES = {'INPUT__REG__m_ddr_speed_variant_SI': (6, 25),
                            'INPUT__REG__m_hazard_type_SI': (7, 25),
                            'INPUT__REG__m_cmd_merge_mode_0__SI': (5, 25),
                            'INPUT__REG__m_cmd_merge_mode_1__SI': (5, 25)}
    # reorder
    EMBEDDING_TABLE_SHAPES = OrderedDict((k, EMBEDDING_TABLE_SHAPES[k]) for k in cat_cols)  

    # update by config
    EMBEDDING_TABLE_SHAPES = OrderedDict({k: (v[0], config['EMBEDDING_SIZE']) for k, v in EMBEDDING_TABLE_SHAPES.items()})
    
    
    # set model params
    NUM_CLASSES = 2
    NUM_CONTINUOUS = 138
    NUM_BINARY = 607

    emb_dropout = config['emb_dropout']  
    transformer_config = config['transformer_config']  

    model = TabularTransformer(embedding_table_shapes=EMBEDDING_TABLE_SHAPES,
                           num_continuous=NUM_CONTINUOUS,
                           num_binary=NUM_BINARY,
                           num_ohe=config['ohe_cols'],
                           num_invariant=2,
                           emb_dropout=emb_dropout,
                           transformer_config=transformer_config,
                           num_classes=NUM_CLASSES,
                           )
   
    # train steps
    num_train_steps_per_epoch = config['train_ds_len'] // ((accelerator.num_processes * config['batch_size_per_device']))
    num_eval_steps_per_epoch = config['eval_ds_len'] // ((accelerator.num_processes * config['batch_size_per_device']))

    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # prepare
    model, optimizer = accelerator.prepare(model, optimizer)

    # data loaders
    train_dataloader = train_ds.iter_torch_batches(batch_size=config['batch_size_per_device'], collate_fn=collate_fn, prefetch_batches=4, drop_last=True, local_shuffle_buffer_size=config['batch_size_per_device']*4)

    # focal loss
    alpha_tensor = torch.tensor([1 - config['ALPHA'], config['ALPHA']]).float().cuda()  
    loss_fn = FocalLoss(alpha=alpha_tensor, gamma=config['GAMMA'], reduction='mean')  
    
    warmup_steps = config['warmup']
    
    # warmup
    initial_lr = config['lr'] * 0.1
    
    # Define a lambda function for the warmup  
    lambda1 = lambda epoch: epoch / warmup_steps if epoch < warmup_steps else 1  
    
    # Create a learning rate scheduler  
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  

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
                outputs = model(x_cat=batch["x_cat"],
                                x_cont=batch["x_cont"],
                                x_binary=batch["x_binary"],
                                x_ohe=batch['x_ohe'],
                                x_invariant=batch['x_invariant'],
                                cat_masks=batch["cat_masks"],
                                cont_masks=batch['cont_masks'],
                                binary_masks=batch['binary_masks'],
                                invariant_masks=batch['invariant_masks'])

                # loss
                labels = batch['labels']

                # Since your network outputs logits for two classes, you need to one-hot encode labels  
                labels_one_hot = F.one_hot(labels, num_classes=2).float()  
            
                # Apply FocalLoss  
                loss = loss_fn(outputs, labels_one_hot)  

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

                # clip
                clip_grad_norm_(accelerator.unwrap_model(model).parameters(), 1.0)

                optimizer.step()
                scheduler.step()
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

        perplex, eloss, eval_recall, eval_precision, eval_f1, eval_ap = evaluate(
            model=model,
            valid_ds=valid_ds,
            loss_fn=loss_fn,
            accelerator=accelerator,
            eval_steps=num_eval_steps_per_epoch,
            config=config,
            ds_kwargs={"collate_fn": collate_fn}
        )

        if accelerator.is_main_process:
            mlflow.log_metric('Recall', eval_recall)
            mlflow.log_metric('Precision', eval_recall)

        eval_e_epoch = time.time()
        accelerator.print("Eval time per epoch: ", eval_e_epoch - eval_s_epoch)

        metrics = {
            "epoch": epoch,
            "iteration": step,
            "avg_train_loss_epoch": loss_sum.item() / (step + 1),
            "eval_recall": eval_recall,
            "eval_precision": eval_precision,
            "eval_f1": eval_f1,
            "eval_ap": eval_ap,
            "eval_loss": eloss,
            "perplexity": perplex,
            "num_iterations": step + 1,
        }

        # only report after a few passes
        if epoch >= 3:
            ray.train.report(metrics)
        else:
            ray.train.report({
            "epoch": epoch,
            "iteration": step,
            "avg_train_loss_epoch": loss_sum.item() / (step + 1),
            "eval_recall": 0,
            "eval_precision": 0,
            "eval_f1": 0,
            "eval_ap": 0,
            "eval_loss": 0,
            "perplexity": 0,
            "num_iterations": step + 1,
        })


if __name__ == '__main__':

    # get args
    args = parse_args()

    args.output_dir = output_path

    if not ray.is_initialized():
        # init a driver
        ray.init(log_to_driver=False, runtime_env={
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

    ctx = ray.data.context.DatasetContext.get_current()
    ctx.use_streaming_executor = True
    ctx.execution_options.verbose_progress = False

    train_set = ray.data.read_parquet(data_path, ray_remote_args={"num_cpus": 0.25},
                                filter=pc.field('test_split').isin(['train'])) \
                                .drop_columns(['test_split', 'randomizedTestKey'])

    valid_set = ray.data.read_parquet(data_path, ray_remote_args={"num_cpus": 0.25},
                                filter=pc.field('test_split').isin(['test'])) \
                                .drop_columns(['test_split', 'randomizedTestKey'])
                                

    # light shuffle
    train_set = train_set.randomize_block_order()
    valid_set = valid_set.randomize_block_order()

    # OHE
    encoder = OneHotEncoder(columns=['INPUT__YML__testname'])
    train_set = encoder.fit_transform(train_set)
    valid_set = encoder.transform(valid_set)

    ohe_cols = list(encoder.stats_['unique_values(INPUT__YML__testname)'].keys())
    ohe_cols = list(map(lambda x: 'INPUT__YML__testname_' + x, ohe_cols))
    config['ohe_cols'] = len(ohe_cols)

    # Dataset config
    options = ray.data.ExecutionOptions(locality_with_output=False, resource_limits=ray.data.ExecutionResources(object_store_memory=230e9))
    dataset_config = DataConfig(datasets_to_split=["train", "valid"], execution_options=options)

    strategy = "STRICT_PACK"  # SPREAD causes major device conflicts
    config['train_ds_len'] = train_set.count()
    config['eval_ds_len'] = valid_set.count()

  
    # Parameter space
    param_space = {
        "train_loop_config": {

            "lr": tune.loguniform(1e-6, 1e-2),
            
            "EMBEDDING_SIZE": tune.choice([20, 25, 30, 50, 75, 100, 125, 150, 200]),

            'emb_dropout': tune.uniform(0.0, 0.5),

            'transformer_config': {  
                                    'dim_model': tune.choice([64, 128, 256, 512]),  
                                    'num_heads': tune.choice([1, 2, 4, 8]),  
                                    'dim_feedforward': tune.choice([128, 256, 512, 1024]),  
                                    'dropout': tune.uniform(0.0, 0.5),  
                                    'n_layers': tune.choice([1, 2, 3, 4, 5]),  
                                },

            "weight_decay": tune.loguniform(1e-4, 0.2),

            "warmup": tune.choice([50, 75, 100, 125, 150, 175, 200]),

            "batch_size_per_device": tune.choice([
                                                    64, 
                                                    96,
                                                    128,
                                                    160,
                                                    192,
                                                    224,
                                                    256,
                                                    288,
                                                    320,
                                                    352,
                                                    384,
                                                    416,
                                                    448,
                                                    480,
                                                    512,
                                                    544,
                                                    576,
                                                    608,
                                                    640,
                                                    672,
                                                    704,
                                                    736,
                                                    768,
                                                    800,
                                                    832,
                                                    864,
                                                    896,
                                                    928,
                                                    960,
                                                    992,
                                                    1024,]),            
            "GAMMA": tune.uniform(0.5, 5.0),
            "ALPHA": tune.uniform(0.5, 0.9),
        }
    }

    # Scheduler
    scheduler = ASHAScheduler(
        time_attr='epoch',
        max_t=config['num_epochs'],
        grace_period=3,
        reduction_factor=2,
        metric='eval_f1',
        mode='max',
    )

    search_alg = HyperOptSearch(metric='eval_f1', mode='max', points_to_evaluate=None)
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=args.max_concurrent)

    # Tune config
    tune_config = tune.TuneConfig(
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=args.num_samples,
    )

    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="eval_f1",
        checkpoint_score_order="max"
    )

    run_config = RunConfig(callbacks=None, checkpoint_config=checkpoint_config, storage_path=args.output_dir)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config,
        datasets={"train": train_set, "valid": valid_set},
        dataset_config=dataset_config,
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=True, placement_strategy=strategy, resources_per_worker={"GPU": args.num_devices,}),
        run_config=run_config,
        )


    # Tuner
    tuner = Tuner(
        trainable=trainer,
        param_space=param_space,
        tune_config=tune_config,
    )

    # Tune
    results = tuner.fit()
    best_trial = results.get_best_result(metric="eval_f1", mode="max")
    print("Best Result:", best_trial)

    # get checkpoint
    checkpoint = best_trial.checkpoint
    print(f"Best Checkpoint: {checkpoint}")



    d = {
            "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
            "params": best_trial.config["train_loop_config"],
        }
    
    print("Best Params", d)
    print("Training complete!")
    print(best_trial.metrics_dataframe.to_dict())


