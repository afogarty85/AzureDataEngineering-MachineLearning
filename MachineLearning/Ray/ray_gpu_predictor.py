from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from datasets import load_dataset
import torch
import ray
from azureml.core import Run
import argparse
import pandas as pd
import numpy as np
from azureml.core import Run
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from functools import partial
import nest_asyncio
nest_asyncio.apply()


# init Azure ML run context data
aml_context = Run.get_context()

device = 'cuda'


def parse_args():

    parser = argparse.ArgumentParser(description="Simple example of eval script.")
    parser.add_argument("--num_devices", type=int, default=4, help="Number of devices to use.")
    parser.add_argument("--source_len", type=int, default=512, help="Token length." )
    parser.add_argument("--target_len", type=int, default=512, help="Token length." )
    parser.add_argument("--input_col", type=str, default='_', help="X." )
    parser.add_argument("--target_col", type=str, default='_', help="y." )
    parser.add_argument("--padding", type=str, default='max_length', help="Padding" )
    parser.add_argument("--experiment_name", type=str, default='_', help="Experiment name." )
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size" )
    parser.add_argument("--train_path", type=str, default='', help="Test file" )
    args = parser.parse_args()

    return args


class TorchPredictor:
    def __init__(self, model, tokenizer, max_length):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.model.eval()
        self.max_length = max_length

    def __call__(self, batch):

        # transform to tensor / attach to GPU
        batch["input_ids"] = torch.as_tensor(batch["input_ids"], dtype=torch.int64, device=device)
        batch["attention_mask"] = torch.as_tensor(batch["attention_mask"], dtype=torch.int64, device=device)

        # like no_grad
        with torch.inference_mode():

            # forward and back to cpu
            out = self.model.generate(input_ids=batch['input_ids'],
                                      attention_mask=batch['attention_mask'],
                                      **{
                "max_length": self.max_length,
                "do_sample": False,
                "repetition_penalty": 1.2
            },
                return_dict_in_generate=True,
                output_scores=True
            )

            decode_out = self.tokenizer.batch_decode(out['sequences'], skip_special_tokens=True)
            decode_input = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

            transition_scores = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True).to(
                dtype=torch.float16)
            transition_proba = torch.exp(transition_scores).cpu().numpy()

            return {
                "y_pred": decode_out,
                "y_true": batch["true_label"],
                "input": decode_input,
                "proba": transition_proba[:, 0]
            }



# collate fn
def collate_fn(batch, args, tokenizer):
    model_inputs = tokenizer(
        list(batch[args.input_col]),
        max_length=args.source_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    model_inputs["true_label"] = batch['label']
    return model_inputs



if __name__ == "__main__":
    
    print(f"Ray version: {ray.__version__}")
    
    args = parse_args()
    ctx = ray.data.context.DatasetContext.get_current()
    ctx.use_streaming_executor = True
    ctx.execution_options.preserve_order = True
    ctx.execution_options.locality_with_output = True

    # get secs
    KVUri = "https://moaddev6131880268.vault.azure.net"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=KVUri, credential=credential)

    # secret vals
    CLIENTID = client.get_secret("dev-synapse-sqlpool-sp-id").value
    CLIENTSECRET = client.get_secret("dev-synapse-sqlpool-sp-secret").value
    TENANT = client.get_secret("tenant").value

    # load saved model
    tokenizer = AutoTokenizer.from_pretrained(aml_context.input_datasets['model_path'], use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(aml_context.input_datasets['model_path'],
                                                    torch_dtype=torch.bfloat16,
                                                    use_cache=False,
                                                    )



    input_path = aml_context.input_datasets['train_files']
    args.train_path = input_path + args.train_path

    ds = ray.data.read_parquet(args.train_path, ray_remote_args={"num_cpus": 0.25})

    # split
    train, test = ds.train_test_split(test_size=0.1, seed=4)

    # process text to tokens
    test = test.map_batches(partial(collate_fn,
                                            args=args,
                                            tokenizer=tokenizer,
                                            )).repartition(96)

    # get preds
    predictions = test.map_batches(TorchPredictor(model=model, tokenizer=tokenizer, max_length=args.target_len),
                                      num_gpus=1, batch_size=args.batch_size, concurrency=args.num_devices)

    # to pandas
    predictions_pd = predictions.to_pandas()
    print('preds generated!')
    predictions_pd['experiment_name'] = args.experiment_name

    # set dtype
    predictions_pd['y_pred'] = predictions_pd['y_pred'].astype('string')
    predictions_pd['experiment_name'] = predictions_pd['experiment_name'].astype('string')
    predictions_pd['proba'] = predictions_pd['proba'].astype(np.float32)
    predictions_pd['input'] = predictions_pd['input'].astype('string')

    print(f"Generated a prediction data set of shape: {predictions_pd.shape}")

    # write
    dating = str(pd.Timestamp('today', tz='UTC'))[:19].replace(' ', '_').replace(':', '_').replace('-', '_')
    predictions_pd.to_parquet(aml_context.output_datasets['output_dir'] + '/' + f'{args.experiment_name}_{dating}.parquet')
    


