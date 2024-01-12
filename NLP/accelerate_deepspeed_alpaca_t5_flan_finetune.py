import pandas as pd
import numpy as np
import math
import torch
import os
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BatchEncoding, AutoTokenizer
from transformers.optimization import Adafactor, AdafactorSchedule, get_scheduler, SchedulerType
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, DummyScheduler, DummyOptim
from time import time
from tqdm.auto import tqdm
from argparse import ArgumentParser
from datasets import Dataset
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, prepare_model_for_int8_training, AdaLoraConfig, PeftModel
from deepspeed.ops.adam import FusedAdam
import bitsandbytes as bnb
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)

# accelerate launch \
# --config_file /mnt/c/Users/afogarty/Desktop/ML/SES/default_config.yaml \
# --use_deepspeed \
# accelerate_t5.py \
# --model_name_or_path google/flan-t5-large \
# --mixed_precision bf16 \
# --gradient_accumulation_steps 1 \
# --per_device_train_batch_size 32 \
# --per_device_eval_batch_size 32 \
# --num_train_epochs 8 \
# --logging_steps 100 \
# --output_dir /mnt/c/Users/afogarty/Desktop/ML/SES/accelerate_chat_large_lora \
# --learning_rate 3e-3 \
# --weight_decay 0.01 \
# --checkpointing_steps epoch \
# --max_source_len 64 \
# --max_target_len 512 \
# --eval True \
# --token_calc True \
# --lora True \
# --lora_r 8 \
# --lora_alpha 16 \
# --int8 False \
# --gradient_checkpointing False \
# --zero_stage 2 \
# --num_cpu_threads_per_process 18 \
# --gradient_clipping 1 \
# --fused_adam True




class Seq2SeqDataset(torch.utils.data.Dataset):
    '''
    This prepares a custom Torch Data Set
    ----------

    df_path : string
        The path for our parquet file to read as a data frame      

    tokenizer : tokenizer
        The tokenizer we are using

    max_source_len : int
        The max length we want to constrain our input tokens to

    max_target_len : int
        The max length we want to constrain our label tokens to        
                        
    text_col : string
        The input (x) column to tokenize    

    label_col : string
        The label (y) column to tokenize

    Returns
    -------
    sample : dict
        A dictionary containing:
        (1) input_ids for the x, 
        (2) attention_mask for the x,
        (3) input_ids for the y,
        (4) attention_mask for the y,

    '''
    def __init__(self, df_path, tokenizer, max_source_len, max_target_len, text_col, label_col):
        # set init params
        self.df = pd.read_parquet(df_path)
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.text_col = text_col
        self.label_col = label_col

    # get len
    def __len__(self):
        return len(self.df)
    
    def token_counter(self, df, tokenizer, text_col, label_col):
        '''
        Generate token statistics to better manage model memory
        '''

        def tokenize_text(example):
            return tokenizer(example[text_col])

        def tokenize_label(example):
            return tokenizer(example[label_col])
        
        def generate_stats(ds):
            lengths = sorted(len(lst) for lst in ds['input_ids'])
            curr_info = dict(min=lengths[0], max=lengths[-1], median=lengths[len(lengths) // 2])
            curr_info.update({"95_percentile": lengths[round(len(lengths) * 0.95)]})
            return curr_info
        
        # turn to data set
        ds = Dataset.from_pandas(df)

        # tokenize data set
        text_ds = ds.map(tokenize_text, batched=True, num_proc=4)
        label_ds = ds.map(tokenize_label, batched=True, num_proc=4)

        # generate statistics
        info = dict(total_samples=len(df),
                    source=generate_stats(ds=text_ds),
                    target=generate_stats(ds=label_ds),
                    )
        return info
    
    def __tokenstats__(self):
        info = self.token_counter(df=self.df,
                            tokenizer=self.tokenizer,
                            text_col=self.text_col,
                            label_col=self.label_col)
        
        # update token lens in class and args
        self.max_source_len = info['source']['95_percentile']
        self.max_target_len = info['target']['95_percentile']
        return print(f"Setting source and target length to 95th %: Text: {self.max_source_len} Label: {self.max_target_len}")

    def tokenize(self, text, is_source):
        """
        T5 truncates on right by default, but we can easily truncate on left
        for the encoder input as there is no special token on the left side
        """
        x = self.tokenizer(
            text,
            max_length=self.max_source_len if is_source else self.max_target_len,
            padding="max_length",  # go to max length specified above
            truncation=True,  #
            return_tensors="pt",
        )

        if is_source:
            assert x.input_ids.ndim == 2
            assert x.input_ids.shape == x.attention_mask.shape
            length = x.input_ids.shape[1]
            start = max(length - self.max_source_len, 0)
            x.input_ids = x.input_ids[:, start:]
            x.attention_mask = x.attention_mask[:, start:]
            assert x.input_ids.shape[1] == self.max_source_len
        return BatchEncoding(x)

    # pull a sample of data
    def __getitem__(self, idx):

        # extract batch from df and tokenize
        x = self.tokenize(self.df[self.text_col][idx], is_source=True)
        y = self.tokenize(self.df[self.label_col][idx], is_source=False)

        # package up
        return {
            "source_ids": x.input_ids.squeeze(),
            "source_mask": x.attention_mask.squeeze(),
            "target_ids": y.input_ids.squeeze(),
            "target_mask": y.attention_mask.squeeze(),
        }


def parse_args():
    # set parser requirements
    parser = ArgumentParser(description="Accelerate Ops")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='bf16',
        help="Mixed precision setting: bf16, fp16, fp32, etc",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the eval dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_source_len",
        type=int,
        default=None,
        help="# of tokens for source data",
    )
    parser.add_argument(
        "--max_target_len",
        type=int,
        default=None,
        help="# of tokens for target data",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="log every n steps",
    )
    parser.add_argument(
        "--val_max_target_len",
        type=int,
        default=None,
        help="val max target length for predictions",
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=2,
        help="Deepspeed stage; 2 or 3",
    )
    parser.add_argument(
        "--gradient_clipping",
        type=int,
        default=1,
        help="Gradient clipping value",
    )
    parser.add_argument(
        "--num_cpu_threads_per_process",
        type=int,
        default=12,
        help="Num threads per process",
    )                       
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--load_best_model",
        action="store_true",
        help="Whether to load the best model at the end of training",
    )
    parser.add_argument(
        "--token_calc",
        default=True,
        type=lambda x: (str(x).lower() == 'true'),
        help="Whether to check token lengths",
    )
    parser.add_argument(
        "--lora",
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
        help="Whether to check token lengths",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="lora R; setting r == alpha seems to not do well",
    )
    parser.add_argument(
        "--int8",
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
        help="Whether to load the model in int8",
    )   
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="lora alpha; setting r == alpha seems to not do well",
    )        
    parser.add_argument(
        "--eval",
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
        help="Whether to evaluate or not -- can be costly in time!",
    )
    parser.add_argument(
        "--fused_adam",
        default=True,
        type=lambda x: (str(x).lower() == 'true'),
        help="Whether we want to use fused adam; trade speed for GPU memory (can't offload!)",
    )    
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
        help="Whether to evaluate or not -- can be costly in time!",
    )       
    parser.add_argument(
            "--lr_scheduler_type",
            type=SchedulerType,
            default="linear",
            help="The scheduler type to use.",
            choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        )    
    # initialize arg parser
    args = parser.parse_args()

    return args


def postprocess_text(preds, labels):
    '''
    For prediction step -- slight cleanup and conversion to list
    '''
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # expects list
    #preds = list(map(lambda x: [x], preds))
    labels = list(map(lambda x: [x], labels))
    return preds, labels


def set_similarity_score(decoded_preds, decoded_labels):
    '''
    Compare how well we match set data
    '''
    # score holder
    intersection_score = 0
    for preds, labels in zip(decoded_preds, decoded_labels):
        # split and remove whitespace
        preds = set([p.strip() for p in preds.split(',')])
        labels = set([l.strip() for l in labels.split(',')])
        # get count
        intersection_count = len(set.intersection(preds, labels))
        # add to score
        intersection_score += intersection_count
    return intersection_score


def exact_match_score(decoded_preds, decoded_labels):
    '''
    Look for exact matches in strings
    '''
    # score holder
    em_score = 0
    for preds, labels in zip(decoded_preds, decoded_labels):
        # clean
        preds = preds.strip()
        labels = labels.strip()
        # check equality
        em = int(preds == labels)
        # add to score
        em_score += em
    return em_score


def eval(args, ds, model, metric, tokenizer, eval_loader, accelerator):
    '''
    Evaluate function with bleu
    '''

    # place in eval
    model.eval()

    gen_kwargs = {
    "max_length": args.val_max_target_len if args.val_max_target_len is not None else ds.max_target_len,
    "num_beams": 4,
    "min_length": 10,
    "length_penalty": False,
    "no_repeat_ngram_size": 0,
    "encoder_no_repeat_ngram_size": 0,
    "repetition_penalty": 2.5,
    }

    print(f"Using these text generating args: {gen_kwargs}")

    total_intersection_score = 0

    for step, batch in enumerate(eval_loader):

        with torch.no_grad():

            # generate predictions
            generated_tokens = accelerator.unwrap_model(model).generate(input_ids=batch["source_ids"],
                                            attention_mask=batch["source_mask"],
                                            **gen_kwargs
                                            )
            
            # unpack labels
            labels = batch["target_ids"]

            # set preds and labels to cpu
            generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()

            # Replace -100 in the labels as we can't decode them; not needed in this case
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            # decode
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            if step == 0:
                decoded_prompts = tokenizer.batch_decode(batch["source_ids"], skip_special_tokens=True)
                # print a few examples
                for prompt, generated_response, response in zip(decoded_prompts[:4], decoded_preds[:4], decoded_labels[:4]):
                    print(f"Prompt: {prompt} | Generated Response: {generated_response} | Label Response: {response}\n")

            # clean and send to metric
            #decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            #metric.add_batch(predictions=decoded_preds, references=decoded_labels, )
            total_intersection_score += set_similarity_score(decoded_preds, decoded_labels)

            
    # compute result
    #result = metric.compute()
    #metric_score = result['rougeL']
    #print({"rogueL": metric_score})

    print({"Total Intersection": total_intersection_score})
    return total_intersection_score


def main():

    # get args
    args = parse_args()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    # build ds 
    # '/mnt/c/Users/afogarty/Desktop/ML/SES/alpaca_clean_to_pq.parquet'
    # '/mnt/c/Users/afogarty/Desktop/newest_demo.parquet'
    myds = Seq2SeqDataset(df_path=r'/mnt/c/Users/afogarty/Desktop/newest_demo.parquet',
                    tokenizer=tokenizer,
                    max_source_len=args.max_source_len,
                    max_target_len=args.max_target_len,
                    text_col='text',
                    label_col='label'
                    )    
    
    # update token lengths
    if args.token_calc:
        myds.__tokenstats__()


    def collate_fn(examples):
        '''
        Pack the samples together into a batch
        '''
        source_ids = []
        source_mask = []
        target_ids = []
        target_mask = []    
        for example in examples:
            source_ids.append((example["source_ids"]))
            source_mask.append((example["source_mask"]))
            target_ids.append((example["target_ids"]))
            target_mask.append((example["target_mask"]))
            
        source_ids = torch.stack(source_ids)
        source_mask = torch.stack(source_mask)
        target_ids = torch.stack(target_ids)
        target_mask = torch.stack(target_mask)

        return {"source_ids": source_ids, "source_mask": source_mask,
                "target_ids": target_ids, "target_mask": target_mask,}



    # set train set
    train_set = myds

    # some indices of interest to eval
    eval_idx_interest = [8557,  8178, 10459, 10423,  9496, 10954,  9186,  7595,  8652,
                         8583,  8381, 10720,  6720,  6694,  7969, 10153,  5813,  5901,
                         7436,  7793,  8731,  7877,  6228,  8214,  8430,  7488,  8190,
                         8411,  7167,  6874,  7764,  9080, 10410,  9523,  7189,  9933,
                         11118,  8601,  9072, 10790,  8126,  6590,  9471, 10342,  6132,
                         9737,  7846,  6607, 10055,  9281,  8585,  7371,  7162,  8428,
                         6460,  6578,  8862, 10072,  7196,  7501,  5614,  8001,  5978,
                         7304,  9438,  7980,  5743,  7120,  5638, 10693,  9614,  6126,
                         9995, 10254,  5733,  7550,  9248, 10203,  5903,  9591,  6098,
                         9308,  9263,  9494,  6042,  7051,  6453,  7018,  7074,  6324,
                         10656,  9566,  6532, 10466,  9384,  9595,  8535, 10126,  8504,
                         8300]
    valid_set = torch.utils.data.Subset(dataset=myds, indices=eval_idx_interest)

    # loaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.per_device_train_batch_size,
                                               drop_last=True,
                                               shuffle=True,
                                               num_workers=6,
                                               collate_fn=collate_fn,
                                               pin_memory=True)
    
    eval_loader = torch.utils.data.DataLoader(valid_set,
                                              batch_size=args.per_device_eval_batch_size,
                                              num_workers=6,
                                              collate_fn=collate_fn,
                                              pin_memory=True)

    # warm steps
    args.num_warmup_steps = int(0.06 * (len(train_set) // args.per_device_train_batch_size) * args.num_train_epochs)

    # lora
    if args.lora:
        print('LoRA Triggered!')

        # set config
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, target_modules=["q", "v"], inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none")

        # load model
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path,
                                                            device_map='auto',
                                                            low_cpu_mem_usage=True,
                                                            load_in_8bit=True if args.int8 else False,
                                                            torch_dtype=torch.float16 if args.int8 is True else "auto",
                                                            )
        
        # if lora + int8
        if args.int8:
            model = prepare_model_for_int8_training(model)
        
        # set peft
        model = get_peft_model(model, peft_config)

        # set id
        peft_model_id = f"{args.output_dir}_{peft_config.peft_type}_{peft_config.task_type}"


    if not args.lora:
        print('Bypassing LoRA...')
        # init model
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path,
                                                            device_map='auto',
                                                            low_cpu_mem_usage=True,
                                                            load_in_8bit=True if args.int8 else False,
                                                            torch_dtype=torch.float16 if args.int8 is True else "auto",
                                                            )
               
        if args.int8:
            print('Bypassing LoRA and going Int8!')
            model = prepare_model_for_int8_training(model)


    # define project
    my_proj = ProjectConfiguration(project_dir=args.output_dir if args.lora is not True else peft_model_id,
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
    if args.fused_adam:
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_optimizer"]['device'] = 'none'
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_param"]["device"] = 'none'
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_optimizer"]["pin_memory"] = False
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_param"]["pin_memory"] = False

    else:
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_optimizer"]['device'] = 'cpu'
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_param"]["device"] = 'cpu'
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_optimizer"]["pin_memory"] = True
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_param"]["pin_memory"] = True


     # split weights in two groups, one with weight decay and the other without
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # scheduler and training steps
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch * args.gradient_accumulation_steps

    # Creates Dummy Optimizer if `optimizer` was spcified in the config file else creates Adam Optimizer
    #optimizer = DummyOptim(optimizer_grouped_parameters, lr=args.learning_rate)

    # Creates Dummy Scheduler if `scheduler` was spcified in the config file else creates `args.lr_scheduler_type` Scheduler
    #lr_scheduler = DummyScheduler(optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps)

    # or torch.optim.AdamW | FusedAdam | apex.optimizers.FusedAdam | Adafactor | bnb.optim.Adam8bit
    if args.fused_adam:
        optimizer = FusedAdam(model.parameters() if args.lora else optimizer_grouped_parameters,
                            adam_w_mode=True,
                            lr=args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters() if args.lora else optimizer_grouped_parameters,
                            lr=args.learning_rate)
        
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type,
                                optimizer=optimizer,
                                num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
                                num_training_steps=args.max_train_steps * args.gradient_accumulation_steps
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
    print(f"  Num examples = {len(train_set)}")
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
    metric = evaluate.load("rouge")

    # train loop
    for epoch in range(1, args.num_train_epochs + 1):
        print(f" Starting epoch {epoch}")
        start_time = time()
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):

            # recode pad to -100 to dodge loss
            batch["target_ids"] = torch.where(batch["target_ids"] == tokenizer.pad_token_id, -100, batch["target_ids"])

            # forward -- with gradient accumulation
            with accelerator.accumulate(model):
                outputs = model(input_ids=batch["source_ids"],
                                attention_mask=batch["source_mask"],
                                labels=batch["target_ids"],
                                decoder_attention_mask=batch["target_mask"],
                                )
        
                # loss / store
                loss = outputs.loss
                total_loss += loss.detach().float()

                # backward
                accelerator.backward(loss)

                # update
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # checkpoint
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    os.makedirs(args.output_dir + '/checkpoints', exist_ok=True)
                    accelerator.save_state()

            # check if update happened
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if completed_steps % args.logging_steps == 0:
                    # report current findings
                    curr_loss = (total_loss / accelerator.gradient_accumulation_steps).item()
                    curr_perplexity = math.exp(curr_loss)
                    # report
                    print(f"Epoch: { round(( completed_steps / (num_update_steps_per_epoch * args.gradient_accumulation_steps) ), 3) }, Step: {completed_steps}, Steps This Epoch: {steps_this_epoch}, Loss: {round(train_loss, 3)}, Perplexity: {round(train_perplexity, 3)}")
                    # reset
                    total_loss = 0

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
            tokenizer.save_pretrained(args.output_dir + rf'/epoch_{epoch}')

            # manual save example
            # torch.save({'model_state_dict': model.state_dict(),}, 'xd.pt')
            # checkpoint = torch.load('xd.pt', map_location='cpu')
            # model.load_state_dict(checkpoint['model_state_dict'])
            # check weight changes
            # model.state_dict()['lm_head.weight'][0, 0:5]

        # if eval
        if args.eval:
            print('Now Evaluating...')

            start_time = time()
            metric_score = eval(args=args, ds=myds, model=model, metric=metric, tokenizer=tokenizer, eval_loader=eval_loader, accelerator=accelerator)
            end_time = time()

            print(f"Epoch {epoch} evaluation took {end_time - start_time} seconds")
            if best_metric < metric_score:
                best_metric = metric_score
                best_metric_checkpoint = os.path.join(args.output_dir, str(epoch))
                print(f"New best metric: {round(best_metric, 3)} at epoch {epoch}")
                print(f"best_metric_checkpoint: {best_metric_checkpoint}")



if __name__ == "__main__":
    main()
