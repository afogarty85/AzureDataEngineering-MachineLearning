import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, T5ForConditionalGeneration
from transformers.optimization import Adafactor, get_inverse_sqrt_schedule, AdafactorSchedule
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from torch.optim.lr_scheduler import LambdaLR
from time import time
from itertools import chain
import math
from tqdm.auto import tqdm
import datasets
from pretrain_utils.utils import compute_input_and_target_lengths, DataCollatorForT5MLM
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)

# 1 step = 1 gradient update
# T5 is pretrained on C4 for this many steps: 524,288
# T5 used 128 batch size of token length 512 and target length of 114
# T5 1.1 uses no dropout in pre-training
# T5 1.1 gets 1.942 perplexity on held-out test set at 65536 steps
# Pretraining LR: constant 0.01 for the first 10k steps, then exponentially decays the learning rate until pre-training is over (inverse sqrt sched)
# Finetuning LR: constant 0.001

# load c4
dataset = datasets.load_dataset('c4', 'en', streaming=True,).remove_columns(['timestamp', 'url'])

# init tokenizer
tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base', use_fast=False)

# get lengths
noise_density = 0.15
input_length = 512
mean_noise_span_length = 3.0

expanded_inputs_length, targets_length = compute_input_and_target_lengths(
    inputs_length=input_length,
    noise_density=noise_density,
    mean_noise_span_length=mean_noise_span_length,
)

# tokenize fn
def tokenize_function(examples):
    return tokenizer(examples['text'], return_attention_mask=False)

# first map
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns='text',
)

# Main data processing function that will concatenate all texts from our dataset and generate chunks of expanded_inputs_length.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= expanded_inputs_length:
        total_length = (total_length // expanded_inputs_length) * expanded_inputs_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)]
        for k, t in concatenated_examples.items()
    }
    return result

# second map
tokenized_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
)

# shuffle ds
tokenized_datasets = tokenized_datasets.shuffle(seed=42, buffer_size=10_000)

# init config
config = AutoConfig.from_pretrained('google/t5-v1_1-base', vocab_size=len(tokenizer))
config.dropout_rate = 0.0

# init model
model = T5ForConditionalGeneration(config=config)

# compile
model = torch.compile(model)

# params
per_device_train_batch_size = 24
gradient_accumulation_steps = 5  # 120 batch size
num_train_epochs = 1
max_train_steps = 65536  # T5 1.1 paper

# init data collator; will take care of randomly masking the token
data_collator = DataCollatorForT5MLM(tokenizer=tokenizer,
                                    noise_density=noise_density,
                                    mean_noise_span_length=mean_noise_span_length,
                                    input_length=input_length,
                                    target_length=targets_length,
                                    pad_token_id=model.config.pad_token_id,
                                    decoder_start_token_id=model.config.decoder_start_token_id,
)

# init data loaders
train_loader = torch.utils.data.DataLoader(tokenized_datasets['train'],
                                            batch_size=per_device_train_batch_size,
                                            drop_last=True,
                                            num_workers=8,
                                            collate_fn=data_collator,
                                            pin_memory=True)

eval_loader = torch.utils.data.DataLoader(tokenized_datasets['validation'],
                                            batch_size=per_device_train_batch_size*2,
                                            num_workers=8,
                                            collate_fn=data_collator,
                                            pin_memory=True)

# optimizer
optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=0.01)

# scheduler
lr_scheduler = get_scheduler(name='constant', optimizer=optimizer)

# accelerate
gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=gradient_accumulation_steps, adjust_scheduler=True)
accelerator = Accelerator(mixed_precision='bf16',
                            gradient_accumulation_plugin=gradient_accumulation_plugin,
                            device_placement=True,
                            )

# initialize device
model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, eval_loader, lr_scheduler)

# set scheduler to save
accelerator.register_for_checkpointing(lr_scheduler)

# train fn
progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
completed_steps = 0
best_metric = 0
best_metric_checkpoint = None
logging_steps = 25
last_logged_step = 0
loss_results = []

# train loop
for epoch in range(1, num_train_epochs + 1):
    print(f" Starting epoch {epoch}")
    start_time = time()
    model.train()
    total_loss = 0

    # set epoch on data set
    tokenized_datasets.set_epoch(epoch)

    for step, batch in enumerate(train_loader):

        # forward -- with gradient accumulation
        with accelerator.accumulate(model):
            outputs = model(**batch)

            # loss / store
            loss = outputs.loss
            total_loss += loss.detach().float()

            # backward
            accelerator.backward(loss)

            # update
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # check if update happened
        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

            if completed_steps % logging_steps == 0:
                # report current findings
                curr_loss = round(total_loss.item() / (completed_steps - last_logged_step), 3)
                curr_perplexity = np.exp(curr_loss)
                loss_results.append(curr_loss)
                print(f"Step: {completed_steps},  Loss: {round(curr_loss, 3)}, Perplexity: {round(curr_perplexity, 3)}")
                # reset
                total_loss = 0
                last_logged_step = completed_steps

        # end at first stage
        if completed_steps >= max_train_steps:
            break

      # report timings
    end_time = time()
    print(f"Epoch {epoch} training took {int(end_time-start_time)} seconds")
    accelerator.wait_for_everyone()
    accelerator.save_state(f'./model_checkpoint_c4/step_{completed_steps}')
    # save config
    model.config.to_json_file(f'./model_checkpoint_c4/step_{completed_steps}/config.json')


def evaluate(model, eval_loader, accelerator):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_loader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(per_device_train_batch_size*2)))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = np.exp(curr_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity, eval_loss

# get eval results
eval_perplexity, eval_loss = evaluate(model=model, eval_loader=eval_loader, accelerator=accelerator)
