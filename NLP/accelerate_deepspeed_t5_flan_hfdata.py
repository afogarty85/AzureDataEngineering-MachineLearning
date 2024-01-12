import pandas as pd
import numpy as np
import math
import torch
import os
from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import Adafactor, get_scheduler, SchedulerType
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, GradientAccumulationPlugin
from time import time
from tqdm.auto import tqdm
from argparse import ArgumentParser, BooleanOptionalAction
from deepspeed.ops.adam import FusedAdam
import apex
from rouge_score import rouge_scorer
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)

# flan-t5-small uses these params for ft: 64 batch size, 0.05 drop out, 5e-4 lr
# 80M Flan-T5-Small 64 0.05 5e-4 98k; same for large; also uses adafactor -

# accelerate launch \
# --config_file /mnt/c/Users/afogarty/Desktop/ML/SES/default_config.yaml \
# --use_deepspeed \
# accelerate_gdcofill.py \
# --model_name_or_path google/flan-t5-small \
# --mixed_precision bf16 \
# --gradient_accumulation_steps 1 \
# --per_device_train_batch_size 128 \
# --per_device_eval_batch_size 128 \
# --num_train_epochs 30 \
# --logging_steps 100 \
# --load_ds_from_disk \
# --output_dir /mnt/c/Users/afogarty/Desktop/GDCOFill/accelerate_small \
# --learning_rate 1e-4 \
# --weight_decay 0.01 \
# --checkpointing_steps epoch \
# --lr_scheduler_type constant \
# --max_source_len 256 \
# --max_target_len 128 \
# --eval False \
# --token_calc False \
# --lora False \
# --lora_r 8 \
# --lora_alpha 16 \
# --int8 False \
# --gradient_checkpointing False \
# --zero_stage 2 \
# --num_cpu_threads_per_process 18 \
# --gradient_clipping 1 \
# --fused_adam False \
# --adafactor

# stay in e-4 range; 1e-4 seems better

# check data lens

# # get input lengths
# out = raw_dataset['train'].map(lambda examples: tokenizer(examples['Comments'], truncation=False),
#                                batched=True,
#                                num_proc=8)

# # get target lengths
# out = raw_dataset['train'].map(lambda examples: tokenizer(examples['ResolutionsJson'], truncation=False),
#                                batched=True,
#                                num_proc=8)

# # get lens alt
# out = raw_dataset.map(lambda x: {'len': (len(x['input_ids']))}, batched=False)

# def generate_stats(ds):
#     lengths = sorted(len(lst) for lst in ds['input_ids'])
#     curr_info = dict(min=lengths[0], max=lengths[-1], median=lengths[len(lengths) // 2])
#     curr_info.update({"99_percentile": lengths[round(len(lengths) * 0.99)]})
#     return curr_info

# generate_stats(out)
# train X: '99_percentile': 258
# train y: '99_percentile': 100


def parse_args():
    # set parser requirements
    parser = ArgumentParser(description="Accelerate Ops")
    parser.add_argument("--mixed_precision", type=str, default='bf16', help="Mixed precision setting: bf16, fp16, fp32, etc",)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.",)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the eval dataloader.",)
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.",)
    parser.add_argument("--max_source_len", type=int, default=None, help="# of tokens for source data",)
    parser.add_argument("--max_target_len", type=int, default=None, help="# of tokens for target data",)
    parser.add_argument("--logging_steps", type=int, default=None, help="log every n steps",)
    parser.add_argument("--val_max_target_len", type=int, default=None, help="val max target length for predictions",)
    parser.add_argument("--zero_stage", type=int, default=2, help="Deepspeed stage; 2 or 3",)
    parser.add_argument("--gradient_clipping", type=int, default=1, help="Gradient clipping value",)
    parser.add_argument("--num_cpu_threads_per_process", type=int, default=12, help="Num threads per process",)                       
    parser.add_argument("--checkpointing_steps", type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.", )
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--load_ds_from_disk", type=bool, default=True, action=BooleanOptionalAction, help="Whether to load the pre-tokenized data set from disk; use --no-load_ds_from_disk to set False", )
    parser.add_argument("--adafactor", type=bool, default=True, action=BooleanOptionalAction, help="Whether to use Adafactor",)    
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--load_best_model", action="store_true", help="Whether to load the best model at the end of training",)
    parser.add_argument("--eval", default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to evaluate or not -- can be costly in time!",)
    parser.add_argument("--fused_adam", default=True, type=lambda x: (str(x).lower() == 'true'),help="Whether we want to use fused adam; trade speed for GPU memory (can't offload!)",)    
    parser.add_argument("--gradient_checkpointing", default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to use gradient checkpointing",)       
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="constant", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],)    
    # initialize arg parser
    args = parser.parse_args()
    return args


def eval(args, model, tokenizer, eval_loader, accelerator):
    '''
    Evaluate function with roguel
    '''

    # place in eval
    model.eval()

    gen_kwargs = {
    "max_length": args.val_max_target_len if args.val_max_target_len is not None else args.max_target_len,
    "num_beams": 5,
    "remove_invalid_values": True,
    "num_return_sequences": 1,
    "do_sample": True, # False = Greedy
    "temperature": 1,
    "top_p": 1,
    "early_stopping": True,
    "repetition_penalty": 1.0,
    }

    print(f"Using these text generating args: {gen_kwargs}")

    pred_storage = []
    label_storage = []
    input_storage = []
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    
    for step, batch in enumerate(eval_loader):
        with torch.no_grad():

            # generate predictions
            generated_tokens = accelerator.unwrap_model(model).generate(batch["input_ids"],
                                                                        attention_mask=batch["attention_mask"],
                                                                        **gen_kwargs)
            
            generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)

            # unpack labels
            labels = batch["target_ids"]

            # If we did not pad to max length, we need to pad the labels too
            labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            # set preds and labels to cpu
            generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()

            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            # decode
            decoded_inputs = tokenizer.batch_decode(batch['input_ids'].cpu().numpy(), skip_special_tokens=True)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # send to storage
            pred_storage.append(decoded_preds)
            label_storage.append(decoded_labels)
            input_storage.append(decoded_inputs)

    # join batches
    label_storage = np.concatenate(label_storage, axis=0)
    pred_storage = np.concatenate(pred_storage, axis=0)

    precision_total = 0
    recall_total = 0
    for pred, label in zip(pred_storage, label_storage):
        precision_val = scorer.score(target=label, prediction=pred)['rouge1'][0] # prec
        recall_val = scorer.score(target=label, prediction=pred)['rouge1'][1] # recall

        precision_total += precision_val
        recall_total += recall_val

    print(f"rouge1 precision: { round(precision_total / len(label_storage), 3) } ")
    print(f"rouge1 recall: { round(recall_total / len(label_storage), 3)} ")

    return precision_total, recall_total


def main():

    # get args
    args = parse_args()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    # set padding; longest = dynamic; max_length is static padding
    padding = "max_length"

    if args.load_ds_from_disk:
        # load
        processed_datasets = load_from_disk('./data/gdco_tokenized_limited')

    else:
        # build ds
        raw_dataset = load_dataset("parquet", data_files='data/gdco_train_limited.parquet')
        train_testvalid = raw_dataset['train'].train_test_split(test_size=0.2, seed=0)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=0)

        # join again
        raw_dataset = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'valid': test_valid['train']})

        def preprocess_function(examples):
            inputs = examples['Comments']
            targets = examples['ResolutionsJson']
            model_inputs = tokenizer(inputs, max_length=args.max_source_len, padding=padding, truncation=True)

            # Setup the tokenizer for targets
            labels = tokenizer(text_target=targets, max_length=args.max_target_len, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length":
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # process text to tokens
        processed_datasets = raw_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=8,
                    remove_columns=['Comments', 'ResolutionsJson'],
                    load_from_cache_file=True,
                    desc="Running tokenizer on dataset",
                )
        
        #save
        processed_datasets.save_to_disk('./data/gdco_tokenized_limited', max_shard_size="128MB")

    # warm steps
    args.num_warmup_steps = int(0.06 * (len(processed_datasets['train']) // args.per_device_train_batch_size) * args.num_train_epochs)

    # init model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path,
                                                        device_map='auto',
                                                        low_cpu_mem_usage=True,
                                                        torch_dtype="auto",
                                                        )

    # set dropout; 0.1 for ft
    if args.dropout_rate:
        model.config.dropout_rate = args.dropout_rate

    # collate
    data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
        )

    # loaders
    train_loader = torch.utils.data.DataLoader(processed_datasets['train'],
                                               batch_size=args.per_device_train_batch_size,
                                               drop_last=True,
                                               shuffle=True,
                                               num_workers=8,
                                               collate_fn=data_collator,
                                               pin_memory=True)
    
    eval_loader = torch.utils.data.DataLoader(processed_datasets['valid'],
                                              batch_size=args.per_device_eval_batch_size,
                                              num_workers=8,
                                              collate_fn=data_collator,
                                              pin_memory=True)

    # define project
    my_proj = ProjectConfiguration(project_dir=args.output_dir,
                                   automatic_checkpoint_naming=True,
                                   total_limit=5,)
    
    # init accelerator
    gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=args.gradient_accumulation_steps, adjust_scheduler=True)
    accelerator = Accelerator(mixed_precision=args.mixed_precision,
                              gradient_accumulation_plugin=gradient_accumulation_plugin,
                              project_config=my_proj,
                              device_placement=True,
                               )
    accelerator.print(f"{AcceleratorState()}")

    # set deepspeed config from accelerator
    accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["stage"] = args.zero_stage    
    accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    accelerator.state.deepspeed_plugin.deepspeed_config["gradient_clipping"] = args.gradient_clipping
    accelerator.state.deepspeed_plugin.deepspeed_config["num_cpu_threads_per_process"] = args.num_cpu_threads_per_process
    accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["round_robin_gradients"] = True

    # cant use fused adam unless we dont offload to device
    if args.fused_adam or args.adafactor:
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_optimizer"]['device'] = 'none'
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_param"]["device"] = 'none'
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_optimizer"]["pin_memory"] = False
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_param"]["pin_memory"] = False

    else:
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_optimizer"]['device'] = 'cpu'
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_param"]["device"] = 'cpu'
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_optimizer"]["pin_memory"] = True
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_param"]["pin_memory"] = True

    # scheduler and training steps
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch * args.gradient_accumulation_steps

    # or torch.optim.AdamW | FusedAdam | apex.optimizers.FusedAdam | Adafactor | bnb.optim.Adam8bit
    if args.fused_adam:
        optimizer = FusedAdam(model.parameters(), adam_w_mode=True, lr=args.learning_rate)
    
    elif args.adafactor:
        # optimizer
        optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=args.learning_rate)

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        
    # set scheduler
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type,
                                optimizer=optimizer,
                                num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
                                num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
                            )
 
    # set checkpoint steps for accelerator save
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None


    if args.gradient_checkpointing:
        print('Turning on gradient checkpoints...')
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # initialize device
    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, eval_loader, lr_scheduler)

    # report batch size; mostly interesting for multi-gpu env
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(processed_datasets['train'])}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    print(f"  Learning Rate = {args.learning_rate}")
    print(f"  L2 = {args.weight_decay}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_metric = 0
    best_metric_checkpoint = None
    total_loss = 0

    # train loop
    for epoch in range(1, args.num_train_epochs + 1):
        print(f" Starting epoch {epoch}")
        start_time = time()
        model.train()

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

            # checkpoint
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    os.makedirs(args.output_dir + '/checkpoints', exist_ok=True)
                    accelerator.save_state()

            # report data so far
            if isinstance(args.logging_steps, int):
                if accelerator.sync_gradients:
                    if completed_steps % args.logging_steps == 0 and step != 0:
                        curr_loss = (total_loss / args.gradient_accumulation_steps).item()
                        curr_perplexity = np.exp(curr_loss)
                        steps_this_epoch = completed_steps % len(train_loader)
                        # report
                        print(f"Epoch: { round(( completed_steps / (num_update_steps_per_epoch * args.gradient_accumulation_steps) ), 3) },\
                         Global Step: {completed_steps}, \
                         Steps This Epoch: {steps_this_epoch}, \
                         Loss: {round(curr_loss, 3)}, \
                         Perplexity: {round(curr_perplexity, 3)}")
                        # reset
                        total_loss = 0

                # if running steps over epochs
                if completed_steps >= args.max_train_steps:
                    break

        # report timings
        end_time = time()
        print(f"Epoch {epoch} training took {int(end_time-start_time)} seconds")
            
        # save state
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)     
            unwrapped_model.save_pretrained(
                args.output_dir + rf'/epoch_{epoch}',
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )
            # save tokenizer
            tokenizer.save_pretrained(args.output_dir + rf'/epoch_{epoch}')

            # save config
            unwrapped_model.config.to_json_file(args.output_dir + rf'/epoch_{epoch}/config.json')

            # manual save example
            # torch.save({'model_state_dict': model.state_dict(),}, 'xd.pt')
            # checkpoint = torch.load('xd.pt', map_location='cpu')
            # model.load_state_dict(checkpoint['model_state_dict'])
            # check weight changes
            # model.state_dict()['lm_head.weight'][0, 0:5]

        # if eval -- its expensive
        if args.eval and epoch % 3 == 0:
            print('Now Evaluating...')

            start_time = time()
            precision_total, recall_total = eval(args=args, model=model, tokenizer=tokenizer, eval_loader=eval_loader, accelerator=accelerator)
            end_time = time()

            print(f"Epoch {epoch} evaluation took {end_time - start_time} seconds")
            # choose recall
            if best_metric < recall_total:
                best_metric = recall_total
                best_metric_checkpoint = os.path.join(args.output_dir, str(epoch))
                print(f"New best metric: {round(best_metric, 3)} at epoch {epoch}")
                print(f"best_metric_checkpoint: {best_metric_checkpoint}")


if __name__ == "__main__":
    main()



