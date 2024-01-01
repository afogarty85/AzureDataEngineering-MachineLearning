from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import transformers
import torch
import ray
import numpy as np
from transformers import get_linear_schedule_with_warmup  
import bitsandbytes as bnb
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, Trainer, default_data_collator
from datasets import load_dataset, interleave_datasets
from azureml.core import Run
from ray.train.huggingface.transformers import prepare_trainer
from transformers.trainer_callback import TrainerCallback
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import RunConfig, ScalingConfig, CheckpointConfig, DataConfig, Checkpoint
import os
import deepspeed
import shutil
from tempfile import TemporaryDirectory
import glob  
print(f"Torch Version: {torch.__version__}")
print(f"Torch CUDA Version: {torch.version.cuda}")



# init Azure ML run context data
aml_context = Run.get_context()
data_path = aml_context.input_datasets['data_files']
output_path = aml_context.output_datasets['output_dir']


class RayTrainReportCallback(TrainerCallback):
    '''
    override callback to reduce disk usage.
    '''
    CHECKPOINT_NAME = "checkpoint"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def on_save(self, args, state, control, **kwargs):
        """Event called after a checkpoint save."""

        dir_path = '/tmp/ray'
        with TemporaryDirectory() as tmpdir:
            # Aggregate all the logged metrics
            metrics = {}
            for log in state.log_history:
                metrics.update(log)

            # Copy ckpt files and construct a Ray Train Checkpoint
            source_ckpt_path = transformers.trainer.get_last_checkpoint(args.output_dir)
            target_ckpt_path = os.path.join(tmpdir, self.CHECKPOINT_NAME)
            shutil.copytree(source_ckpt_path, target_ckpt_path)
            checkpoint = Checkpoint.from_directory(tmpdir)

            # Report latest metrics and checkpoint to Ray Train
            ray.train.report(metrics=metrics, checkpoint=checkpoint)

            for file in glob.glob(os.path.join(dir_path, '**/*'), recursive=True):    
                if os.path.getsize(file) > 100 * 1024 * 1024:  # size > 100MB  
                    os.remove(file)      

            for file in glob.glob(os.path.join(tmpdir, '**/*'), recursive=True):    
                if os.path.getsize(file) > 100 * 1024 * 1024:  # size > 100MB  
                    os.remove(file)      

            for file in glob.glob(os.path.join(source_ckpt_path, '**/*'), recursive=True):    
                if os.path.getsize(file) > 100 * 1024 * 1024:  # size > 100MB  
                    os.remove(file)      

            for file in glob.glob(os.path.join(target_ckpt_path, '**/*'), recursive=True):    
                if os.path.getsize(file) > 100 * 1024 * 1024:  # size > 100MB  
                    os.remove(file)      


def collate_fn(examples):
    return examples

def train_func(config):

    # tf32
    torch.backends.cuda.matmul.allow_tf32 = True
   
    # deepspeed
    deepspeed = {   
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
        "round_robin_gradients": True
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 10,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
    }

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2",
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # train_ds = ray.train.get_dataset_shard("train")
    # valid_ds = ray.train.get_dataset_shard("valid")

    # train_ds_iterable = train_ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn, prefetch_batches=1,)
    # valid_ds_iterable = valid_ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn, prefetch_batches=1,)
    flan_ds = load_dataset("json", data_dir=data_path, split='train', num_proc=180).shuffle(seed=4).remove_columns(['_template_idx', '_task_source', '_task_name', '_template_type'])
    # .train_test_split(test_size=0.1, seed=4)
        
    def formatting_func(example):
        """
        Function to pack 'inputs' and 'targets' into a single string in the format 
        'Question: {inputs}\nAnswer: {targets}<EOS>'

        Arguments:
        example -- a dictionary containing 'inputs' and 'targets' keys.

        Returns:
        A string formatted as 'Question: {inputs}\nAnswer: {targets}<EOS>'
        """
        text = f"Question: {example['inputs']}\nAnswer: {example['targets']}<EOS>"
        return text


    def formatting_prompts_complete(example):
        ''' completions '''
        output_texts = []
        for i in range(len(example['inputs'])):
            #text = f"### Question: {example['inputs'][i]}\n ### Answer: {example['targets'][i]}"
            text = f"{example['inputs'][i]} {example['targets'][i]}</s>"
            output_texts.append(text)
        return output_texts

    # for completion fine-tuning -- did not flag with previous example:
    #response_template = " ### Answer:"
    #collator = DataCollatorForCompletionOnlyLM(response_template=tokenizer.encode(response_template, add_special_tokens=False)[2:], tokenizer=tokenizer,)

    # fails
    # response_template = " "
    # collator = DataCollatorForCompletionOnlyLM(response_template=tokenizer.encode(response_template, add_special_tokens=False), tokenizer=tokenizer,)


    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, trust_remote_code=True)

    initial_token_count = len(tokenizer)
    response_template = " "
    added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [response_template]})
    model.resize_token_embeddings(new_num_tokens=initial_token_count+added_token_count)

    response_template = " "
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    args = TrainingArguments(
            logging_steps=1,
            save_strategy='steps',
            evaluation_strategy="no",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_checkpointing=False,
            gradient_accumulation_steps=1,
            learning_rate=4e-4,
            max_steps=38000,
            weight_decay=0.01,
            push_to_hub=False,
            save_steps=200,
            report_to='none',
            optim="adamw_bnb_8bit",  # faster forward/backward
            lr_scheduler_type="cosine",
            output_dir="Phi2-Packing",
            fp16=True,
            bf16=False,
            tf32=True,
            deepspeed=deepspeed,           
        )

    # SFT
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=flan_ds,
        #eval_dataset=flan_ds['test'],
        formatting_func=formatting_prompts_complete,
        data_collator=collator,  # for completion fine-tuning
        max_seq_length=800,  # 2048
        neftune_noise_alpha=5,
        packing=False,
        dataset_num_proc=384,
        args=args,
    )

    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)

    print("Starting training")
    trainer.train()



ray.init(
    runtime_env={
        "env_vars": {
            "RAY_AIR_LOCAL_CACHE_DIR": output_path,
        },
    }
)



# # data
# flan_ds = load_dataset("json", data_dir=data_path, split='train').remove_columns(['task']).train_test_split(test_size=0.2, seed=4)

# ray_datasets = {
#     "train": ray.data.from_huggingface(flan_ds["train"]),
#     "valid": ray.data.from_huggingface(flan_ds["test"]),
# }


# training notes: 
# ZeRO-2 // batch_size 2 // 2048 max_seq_length = 80 gb GRAM used on A100 (80GB) // 15-20s a step at grad accum 1
# ZeRO-2 // batch_size 12 // 512 max_seq_length = 64 gb GRAM on A100 // training on completions // ~ 20/25s a step
# ZeRO-2 // batch_size 6 // 800 max_seq_length = 64-74 gb GRAM on A100 // training on completions // ~ 20/25s a step

batch_size = 6

# trainer
my_trainer = TorchTrainer(
    train_func,
    run_config=RunConfig(storage_path=output_path, name="Phi2-Packing", checkpoint_config=CheckpointConfig(num_to_keep=7, checkpoint_frequency=0)),
    scaling_config=ScalingConfig(num_workers=16, use_gpu=True, resources_per_worker={"GPU": 1},),
    # datasets=ray_datasets,
    # dataset_config=ray.train.DataConfig(datasets_to_split=["train"]),

)

result = my_trainer.fit()



# 


# override
# class CustomTrainer(Trainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def create_optimizer_and_scheduler(self, num_training_steps):
#         self.optimizer = adamw_bnb_8bit(self.model.parameters(), 
#                                         lr=self.args.learning_rate, 
#                                         weight_decay=self.args.weight_decay)
        
#         self.lr_scheduler = get_linear_schedule_with_warmup(
#             self.optimizer, 
#             num_warmup_steps=self.args.warmup_steps, 
#             num_training_steps=num_training_steps
#         )

