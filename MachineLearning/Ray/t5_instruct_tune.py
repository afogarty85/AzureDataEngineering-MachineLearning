import argparse
import math
import os
import tempfile
from typing import Tuple
from azureml.core import Run
from transformers.optimization import Adafactor, get_scheduler
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
import torch
import torch.nn as nn
import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import ray
import ray.util.scheduling_strategies
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint
import mlflow
from ray.train import (
    Checkpoint,
    CheckpointConfig,
    DataConfig,
    RunConfig,
    ScalingConfig,
)
import ray.train as train
from tempfile import TemporaryDirectory
from torch.optim.lr_scheduler import OneCycleLR  


# init Azure ML run context data
aml_context = Run.get_context()



def parse_args():

    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mx",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )

    parser.add_argument(
        "--batch_size_per_device",
        type=int,
        default=16,
        help="Batch size to use per device.",
    )

    parser.add_argument(
        "--eval_batch_size_per_device",
        type=int,
        default=64,
        help="Batch size to use per device (For evaluation).",
    )
    parser.add_argument("--padding", default="max-length", type=str)
    parser.add_argument("--input_col", default="input", type=str)
    parser.add_argument("--target_col", default="label", type=str)
    parser.add_argument("--train_path", default="Loc of train file", type=str)
    parser.add_argument(
        "--num_devices", type=int, default=1, help="Number of devices to use."
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps."
    )
    parser.add_argument("--no_grad_ckpt", action="store_true",
                        help="If passed, will not use gradient checkpointing.", )
    parser.add_argument("--no_adafactor", action="store_true",
                        help="If passed, will use AdamW", )    
    parser.add_argument("--output_dir", type=str, help="Path to output directory.")

    parser.add_argument(
        "--model_name", default="google/flan-t5-small", type=str
    )
    parser.add_argument(
        "--experiment_name", default="", type=str
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--seed", type=int, default=4, help="Seed."
    )    
    parser.add_argument(
        "--num-checkpoints-to-keep",
        type=int,
        help=(
            "Number of checkpoints to keep, if None, all checkpoints will be kept, "
            "if set to n>=1, the top n checkpoint with min. evaluation perplexity "
            "will be kept."
        ),
        default=None,
    )
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate to use.")
    parser.add_argument(
        "--ds_config",
        type=str,
        default="./deepspeed_configs/t5_xl.json",
        help="Deepspeed config json to use.",
    )


    args = parser.parse_args()

    return args


def evaluate(*, model, eval_ds, accelerator, bsize, eval_steps, ds_kwargs) -> Tuple[float, float]:
    model.eval()
    losses = []

    eval_dataloader = eval_ds.iter_torch_batches(batch_size=bsize, prefetch_batches=4, **ds_kwargs)
    for step, batch in tqdm.tqdm(enumerate(eval_dataloader), total=eval_steps):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss[None]))

    losses = torch.stack(losses)
    try:
        eval_loss = torch.mean(losses).item()
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity, eval_loss




def training_function(kwargs: dict):
    print("training_function called")

    cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = cuda_visible_device[local_rank]
    os.environ["ACCELERATE_TORCH_DEVICE"] = f"cuda:{device_id}"

    config = kwargs["config"]
    args = argparse.Namespace(**kwargs["args"])

    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size_per_device"])
    gradient_accumulation_steps = int(config["gradient_accumulation_steps"])

    # Get deepspeed config to setup the batch size per device
    ds_plugin = config["ds_plugin"]
    ds_plugin.hf_ds_config.config["train_micro_batch_size_per_gpu"] = batch_size

    # Initialize accelerator
    accelerator = Accelerator(
        deepspeed_plugin=ds_plugin,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=args.mx,
    )

        
    def collate_fn(batch):
        model_inputs = tokenizer(
            list(batch[args.input_col]),
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                list(batch[args.target_col]),
                max_length=410,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels

        model_inputs = {k: v.to(accelerator.device) for k, v in model_inputs.items()}
        return model_inputs

    set_seed(seed)

    # train_ds is the local shard for this model
    train_ds = train.get_dataset_shard("train")
    valid_ds = train.get_dataset_shard("test")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                                torch_dtype='auto',
                                                use_cache=False,
                                                )

    # set dropout
    model.config.dropout_rate = 0.05


    num_train_steps_per_epoch = config['train_ds_len'] // ((accelerator.num_processes * args.batch_size_per_device))
    num_eval_steps_per_epoch = config['eval_ds_len'] // ((accelerator.num_processes * args.eval_batch_size_per_device))
    
    total_training_steps = (num_train_steps_per_epoch * num_epochs // gradient_accumulation_steps )

    print("Model initialized with pretrained weights. Training starting...")
    if not args.no_grad_ckpt:
        print("Enabling gradient checkpointing....")
        model.gradient_checkpointing_enable()

    if not args.no_adafactor:
        print("Using Adafactor...")
        optimizer = Adafactor(model.parameters(),
                              scale_parameter=False,
                                relative_step=False,
                                lr=config['lr'])

        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"][
            "offload_optimizer"]['device'] = 'none'
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"][
            "offload_optimizer"]["pin_memory"] = False


    total_training_steps = (num_train_steps_per_epoch * num_epochs // gradient_accumulation_steps )
    NUM_WARMUP_STEPS = 300

    lr_scheduler = get_scheduler(name='constant',
                                 optimizer=optimizer,
                                 num_warmup_steps=NUM_WARMUP_STEPS * args.num_devices,
                                 num_training_steps=total_training_steps * args.num_devices,
                                 )


    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    train_dataloader = train_ds.iter_torch_batches(
        batch_size=batch_size,
        prefetch_batches=4,
        drop_last=True,
        local_shuffle_buffer_size=batch_size*10,
        collate_fn=collate_fn,
    )
    # Now we train the model
    if accelerator.is_main_process:
        print("Starting training ...")
        print("Number of batches on main process", num_train_steps_per_epoch)

    for epoch in range(num_epochs):
        model.train()
        loss_sum = torch.tensor(0.0).to(accelerator.device)
        running_loss_sum = 0.0

        for step, batch in tqdm.tqdm(enumerate(train_dataloader), total=num_train_steps_per_epoch):

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                
                outputs = model(**batch)

                loss = outputs.loss
                accelerator.backward(loss)
                loss_sum += loss.item()
                running_loss_sum += loss.detach().item()

                optimizer.step()
                lr_scheduler.step()
                

            # Calculate and print average loss every n_steps  
            if step % 100 == 0 and step > 0:  
                average_loss = running_loss_sum / step  
                if accelerator.is_main_process:  
                    accelerator.print(  
                        f"[epoch {epoch} step {step} "  
                        f"average loss: {average_loss:.4f}]"  
                    )  
                # Reset cumulative loss sum after reporting  
            if accelerator.is_main_process:
                mlflow.log_metric('train_loss', running_loss_sum)                
                running_loss_sum = 0.0
            
            aggregated_loss = torch.mean(accelerator.gather(loss[None])).item()
            
        print("Running evaluation ...")
        perplex, eloss = evaluate(
            model=model,
            eval_ds=valid_ds,
            accelerator=accelerator,
            bsize=config["eval_batch_size_per_device"],
            eval_steps=num_eval_steps_per_epoch,
            ds_kwargs={"collate_fn": collate_fn},
        )
        accelerator.print("Eval result loss", eloss)
        accelerator.print("Eval perplex", perplex)

        if accelerator.is_main_process:
            mlflow.log_metric('eval_perplexity', perplex)
            mlflow.log_metric('eval_loss', eloss)

        metrics = {
            "epoch": epoch,
            "iteration": step,
            "train_loss_batch": aggregated_loss,
            "avg_train_loss_epoch": loss_sum.item() / (step + 1),
            "eval_loss": eloss,
            "perplexity": perplex,
            "learning_rate": lr_scheduler.get_lr()[0],
            }


        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            accelerator.print(f"Saving the model locally at {temp_checkpoint_dir}")
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                print("Saving tokenizer and config.")
                tokenizer.save_pretrained(temp_checkpoint_dir)

            accelerator.wait_for_everyone()

            # Checkpointing strategy 2: Aggregate model on the rank 0 worker then upload
            aggregate_on_rank_0 = True
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                temp_checkpoint_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                safe_serialization=True,
                state_dict=accelerator.get_state_dict(model),
            )
            accelerator.wait_for_everyone()

            # Create the checkpoint object to report to Ray Train and upload to storage.
            # If we aggregated the model on rank 0, we only need to report
            # the checkpoint from the rank 0 worker, since all other checkpoint
            # directories are empty (`save_pretrained` was a noop for other workers).
            if aggregate_on_rank_0:
                checkpoint = (
                    Checkpoint.from_directory(temp_checkpoint_dir)
                    if accelerator.is_main_process
                    else None
                )
            else:
                # Distributed checkpointing should upload shards from each worker.
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            # Note: After `train.report`, in the case of remote storage,
            # the checkpoint directory will be uploaded to the remote storage.
            train.report(metrics, checkpoint=checkpoint)




def main():


    args = parse_args()

    # init Azure ML run context data
    aml_context = Run.get_context()

    # get aml paths
    input_path = aml_context.input_datasets['train_files']
    export_path = aml_context.output_datasets['output_dir']

    args.output_dir = export_path + '/' + args.experiment_name + '/checkpoints'
    args.train_path = input_path + args.train_path

    if not args.output_dir:
        raise ValueError("--output_dir must be specified")

    # update the config with args so that we have access to them.
    config = vars(args)

    os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = args.output_dir

    ray.init(
        runtime_env={
            "env_vars": {
                "RAY_AIR_LOCAL_CACHE_DIR": os.environ["RAY_AIR_LOCAL_CACHE_DIR"],
            },
            "working_dir": ".",
        }
    )

    # Add deepspeed plugin to the config
    ds_plugin = DeepSpeedPlugin(hf_ds_config=config.get("ds_config"))
    config.update(ds_plugin=ds_plugin)

    ctx = ray.data.context.DatasetContext.get_current()
    ctx.use_streaming_executor = True
    ctx.execution_options.verbose_progress = True

    ds = ray.data.read_parquet(args.train_path, ray_remote_args={"num_cpus": 0.25})
    
    # split
    train, test = ds.train_test_split(test_size=0.1, seed=4)

    # light shuffle
    train = train.randomize_block_order()
    test = test.randomize_block_order()

    ray_datasets = {
        "train": train,
        "test": test,
    }

    config['train_ds_len'] = ray_datasets['train'].count()
    config['eval_ds_len'] = ray_datasets['test'].count()

    trainer = TorchTrainer(
        training_function,
        train_loop_config={
            "config": config,
            "args": vars(args),
        },
        run_config=RunConfig(
            storage_path=args.output_dir,
            checkpoint_config=CheckpointConfig(
                num_to_keep=None,
                checkpoint_score_attribute="perplexity",
                checkpoint_score_order="min",
            ),
        ),
        scaling_config=ScalingConfig(
            num_workers=args.num_devices,
            use_gpu=True,
            resources_per_worker={"GPU": 1},
        ),
        datasets=ray_datasets,
        dataset_config=DataConfig(datasets_to_split=["train", "test"]),
    )

    result: train.Result = trainer.fit()
    best_checkpoint, best_checkpoint_metrics = result.best_checkpoints[-1]

    print("Results are stored at:")
    print(result.path)
    print("Best checkpoint is stored at:")
    print(best_checkpoint)
    print(f"With perplexity: {best_checkpoint_metrics['perplexity']}")


if __name__ == "__main__":
    main()