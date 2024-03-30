from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse
import torch
import ray
import numpy as np
from ray import train
import pandas as pd
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, Trainer, default_data_collator
from datasets import load_dataset, interleave_datasets
from azureml.core import Run
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import RunConfig, ScalingConfig, CheckpointConfig, DataConfig, Checkpoint
import os
import transformers
import tempfile
from transformers.trainer_callback import TrainerCallback
import deepspeed
import shutil
import glob
import bitsandbytes as bnb
from peft import LoraConfig

print(f"Torch Version: {torch.__version__}")
print(f"Torch CUDA Version: {torch.version.cuda}")
torch.backends.cuda.matmul.allow_tf32 = True


class RayTrainReportCallback(TrainerCallback):  
    '''  
    Override callback to reduce disk usage.  
    '''  
    CHECKPOINT_NAME = "checkpoint"  
  
    def on_save(self, args, state, control, **kwargs):  
        """Event called after a checkpoint save."""  
        # Aggregate all the logged metrics  
        metrics = {}  
        for log in state.log_history:  
            metrics.update(log)  
  
        # Copy ckpt files and construct a Ray Train Checkpoint  
        source_ckpt_path = transformers.trainer.get_last_checkpoint(args.output_dir)  
        with tempfile.TemporaryDirectory() as tmpdir:  
            target_ckpt_path = os.path.join(tmpdir, self.CHECKPOINT_NAME)  
            shutil.copytree(source_ckpt_path, target_ckpt_path)  
            checkpoint = Checkpoint.from_directory(tmpdir)  
              
            # Report latest metrics and checkpoint to Ray Train  
            ray.train.report(metrics=metrics, checkpoint=checkpoint)  
  
        # Clean up large files in the output directory, if necessary  
        self.cleanup_large_files(args.output_dir)  
    
    def cleanup_large_files(self, dir_path):  
        """Remove files larger than 100MB in the given directory."""  
        for file in glob.glob(os.path.join(dir_path, '**/*'), recursive=True):  
            try:  
                if os.path.isfile(file) and os.path.getsize(file) > 100 * 1024 * 1024:  # size > 100MB  
                    os.remove(file)  
                elif os.path.isdir(file) and not os.path.islink(file):  # Avoid attempting to remove symlinks as directories  
                    # Optionally, check the combined size of directory contents before removal  
                    shutil.rmtree(file)  
            except FileNotFoundError:  
                print(f"File not found: {file}")  
            except OSError as e:  
                print(f"Error removing file {file}: {e}")  


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
        default=16,
        help="Batch size to use per device.",
    )

    parser.add_argument(
        "--eval_batch_size_per_device",
        type=int,
        default=64,
        help="Batch size to use per device (For evaluation).",
    )
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="If passed, will run the script in test mode.",
    )

    parser.add_argument(
        "--num-devices", type=int, default=4, help="Number of devices to use."
    )
    parser.add_argument(
        "--grad_accum", type=int, default=1, help="Gradient accumulation steps."
    )
    parser.add_argument("--train_path", type=str, help="Path to training jsonl file")
    parser.add_argument("--test_path", type=str, help="Path to testing jsonl file")
    parser.add_argument(
        "--special_token_path", type=str, help="Path to token json file"
    )
    parser.add_argument(
        "--no-grad-ckpt",
        action="store_true",
        help="If passed, will not use gradient checkpointing.",
    )
    parser.add_argument("--output_dir", type=str, help="Path to output directory.")
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf", type=str)
    parser.add_argument("--lr_scheduler_type", default="cosine", type=str)
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train for." )
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
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate to use.")
    parser.add_argument("--ctx_len", type=int, default=1005, help="Token length.")
    parser.add_argument("--ds-config", type=str, default="./zero_3_llama_2_7b.json", help="Deepspeed config json to use.",)
    args = parser.parse_args()

    return args


def train_func(kwargs: dict):
    os.environ["OMP_NUM_THREADS"] = str(
            train.get_context().get_trial_resources().bundles[-1].get("CPU", 1)
        )

    config = kwargs["config"]

    # tf32
    torch.backends.cuda.matmul.allow_tf32 = True
   
    # deepspeed
    deepspeed = {
    "bf16": {
        "enabled": "auto"
    },
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}

    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'],
        token='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',
        add_eos_token=False,
        add_bos_token=False,
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = 'right'


    ds = load_dataset("parquet", data_files=train_path, split='train', num_proc=96).train_test_split(test_size=0.1, seed=4)


    peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=256,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
            logging_steps=1,
            save_strategy='steps',
            save_steps=200,
            save_only_model=True,
            num_train_epochs=config['num_epochs'],
            per_device_train_batch_size=config['batch_size_per_device'],
            per_device_eval_batch_size=config['eval_batch_size_per_device'],
            evaluation_strategy="steps",
            eval_steps=200,
            gradient_checkpointing=True,
            gradient_accumulation_steps=1,
            learning_rate=config['lr'],
            weight_decay=0.001,
            push_to_hub=False,
            optim="paged_adamw_32bit",
            report_to='none',
            lr_scheduler_type="cosine_with_restarts",
            output_dir="PEFT",
            fp16=False,
            bf16=True,
            tf32=True,
            deepspeed=deepspeed,           
        )

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        attn_implementation="flash_attention_2",
        token='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',
        torch_dtype='auto',
        use_cache=False,
        )

    model.resize_token_embeddings(len(tokenizer))

    def formatting_func(example):
        output_texts = []
 
        for i in range(len(example['prompt'])):
            text = example['prompt'][i]
            output_texts.append(text)
        return output_texts

    def packing_func(example):
        return example['prompt']

    response_template = "\n[/INST]"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    tokenizer.decode(response_template_ids)

    # SFT
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        max_seq_length=config['ctx_len'],
        neftune_noise_alpha=5,
        formatting_func=packing_func,
        peft_config=peft_config,
        #data_collator=collator,
        packing=True,
        dataset_num_proc=96,  # only if packing is False
        args=training_args,
    )

    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)

    print("Starting training")
    trainer.train()
    trainer.save_model(training_args.output_dir)



# Init
# ====================================================

# args
args = parse_args()
config = vars(args)

# init Azure ML run context data
aml_context = Run.get_context()
output_path = aml_context.output_datasets['output_dir']
train_path = aml_context.input_datasets['train_files'] + args.train_path

# ray
ray.init(
    runtime_env={
        "env_vars": {
            "RAY_AIR_LOCAL_CACHE_DIR": output_path,
        },
    }
)


# trainer
my_trainer = TorchTrainer(
    train_func,
    train_loop_config={
            "config": config,
        },
    run_config=RunConfig(storage_path=output_path, name="PEFT", checkpoint_config=CheckpointConfig(num_to_keep=7, checkpoint_frequency=0)),
    scaling_config=ScalingConfig(num_workers=config['num_devices'], use_gpu=True, resources_per_worker={"GPU": 1},),

)

result = my_trainer.fit()


