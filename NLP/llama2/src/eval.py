from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
import ray
import functools
import tree
from azureml.core import Run
import pickle
import argparse
from argparse import BooleanOptionalAction
import shutil
import glob
import os
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from azureml.core import Run


# init Azure ML run context data
aml_context = Run.get_context()


def parse_args():

    parser = argparse.ArgumentParser(description="Simple example of eval script.")
    parser.add_argument("--num_devices", type=int, default=4, help="Number of devices to use.")
    parser.add_argument("--ctx-len", type=int, default=512, help="Token length." )
    parser.add_argument("--experiment-name", type=str, default='_', help="Experiment name." )
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size" )
    parser.add_argument("--use_instruct", default=False, type=lambda x: (str(x).lower() == 'true'), help="If added, use instruct parsing", )
    parser.add_argument("--test_path", type=str, default='test.jsonl', help="Test file name." )
    args = parser.parse_args()

    return args


class TorchPredictor:
    def __init__(self, args, model):
        self.model = model.cuda()
        self.model.eval()
        self.args = args
        self.scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)

    def __call__(self, batch):

        # transform to tensor / attach to GPU
        batch["input_ids"] = torch.as_tensor(batch["input_ids"], dtype=torch.int64, device="cuda")
        batch["attention_mask"] = torch.as_tensor(batch["attention_mask"], dtype=torch.int64, device="cuda")

        # like no_grad
        with torch.inference_mode():

            # forward and back to cpu
            out = self.model.generate(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            **{
                                "max_length": args.ctx_len + 300,
                                "do_sample": False,
                                }
                            )

            # decode
            if args.use_instruct:
                decode_out = tokenizer.batch_decode(out, skip_special_tokens=True)
            else:
                decode_out = tokenizer.batch_decode(out, skip_special_tokens=False)

            # predictions
            if args.use_instruct:
                predicted_jsons = [s.split('[/INST]')[-1].strip(' ').replace('Output: ', '').replace('output: ', '').replace('\n', '') for s in decode_out]
            else:
                predicted_jsons = [s.split('<START_A>')[-1].split('<END_A>')[0].strip(' ') for s in decode_out]

            # rouge_prec
            rouge_prec = [self.scorer.score(pred, true)['rouge1'][0] for pred, true in zip(predicted_jsons, batch['y_true'])]

            # rouge_recall
            rouge_recall = [self.scorer.score(pred, true)['rouge1'][1] for pred, true in zip(predicted_jsons, batch['y_true'])]

            return {
                "y_pred": predicted_jsons,
                "y_true": batch['y_true'],
                "raw": batch['input'],
                'rouge_precision': rouge_prec,
                'rouge_recall': rouge_recall,                
                }


# collate to strip answer
def collate_fn(batch, tokenizer, block_size, device, myargs):

    # eval collate
    if myargs.use_instruct:
        batch['input'] = [s.split('[/INST]')[0] + '[/INST]' for s in batch['input']]
    else:
        batch['input'] = [s.split('<END_Q>')[0] + '<END_Q> ' for s in batch['input']]

    out_batch = tokenizer(
        list(batch["input"]),
        padding="max_length",
        max_length=block_size,
        truncation=True,
        return_tensors="pt",
    )
    out_batch["labels"] = out_batch["input_ids"].clone()

    out_batch = tree.map_structure(lambda x: x.to(device), out_batch)

    return out_batch


if __name__ == "__main__":
    
    print(f"Ray version: {ray.__version__}")
    
    args = parse_args()
    
    if args.use_instruct:
        print("Using instruct-style parsing")

    if not ray.is_initialized():
        # init a driver
        ray.init()
        print('ray initialized')

    ctx = ray.data.context.DatasetContext.get_current()
    ctx.use_streaming_executor = True
    ctx.execution_options.preserve_order = False

    # load test data
    test_path = aml_context.input_datasets['train_files'] + '/' + args.test_path
    
    test_ds = ray.data.read_json(test_path, parallelism=200).repartition(96)

    # assign y_True
    if args.use_instruct:
        test_ds = ray.data.from_pandas(
            test_ds.to_pandas().assign(
                y_true=test_ds.to_pandas()['input'].map(
                    lambda x: x.split('\nOutput:')[-1].replace('</s>', '').strip(' '))))
    else:
        test_ds = ray.data.from_pandas(
            test_ds.to_pandas().assign(
                y_true=test_ds.to_pandas()['input'].map(
                    lambda x: x.split('<START_A>')[-1].replace('<END_A>', '').strip(' '))))

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Run.get_context().input_datasets['fine_tuned_model'],
        )
    print('loaded tokenizer')
    
    # for batch inference
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
    print('loaded tokenizer')

    # model
    model = AutoModelForCausalLM.from_pretrained(
        Run.get_context().input_datasets['fine_tuned_model'],
        torch_dtype=torch.bfloat16)
    print('loaded model')

    # tokenize ds
    collate_partial = functools.partial(
        collate_fn,
        tokenizer=tokenizer,
        block_size=args.ctx_len,
        device='cpu',
        myargs=args,
    )

    # tokenize
    test_ds_tokenized = test_ds.map_batches(collate_partial, batch_size=4096)
    print('tokenizing data')

    # add raw text in and labels in
    test_ds_tokenized = ray.data.from_pandas(pd.concat([test_ds.to_pandas(), test_ds_tokenized.to_pandas()], axis=1))

    # repartition again
    test_ds_tokenized = test_ds_tokenized.repartition(96)

    # get preds
    predictions = test_ds_tokenized.map_batches(TorchPredictor(args=args, model=model), num_gpus=1, batch_size=args.batch_size, compute=ray.data.ActorPoolStrategy(size=args.num_devices))
    print('preds generated!')

    # to pandas
    predictions_pd = predictions.to_pandas()
    predictions_pd['experiment_name'] = args.experiment_name

    # write
    with open(aml_context.output_datasets['output_dir'] + '/' + args.experiment_name + '.pickle', 'wb') as handle:
        pickle.dump(predictions_pd, handle, protocol=pickle.HIGHEST_PROTOCOL)


