from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import torch
import ray
import functools
import tree
from azureml.core import Run
import pickle
import argparse
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from azureml.core import Run
from azure.identity.aio import ClientSecretCredential
from azure.storage.filedatalake.aio import DataLakeServiceClient
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from io import BytesIO
import re
import asyncio
import gc
import nest_asyncio
nest_asyncio.apply()



device = 'cuda'


def parse_args():

    parser = argparse.ArgumentParser(description="Simple example of eval script.")
    parser.add_argument("--num_devices", type=int, default=4, help="Number of devices to use.")
    parser.add_argument("--source_len", type=int, default=512, help="Token length." )
    parser.add_argument("--target_len", type=int, default=512, help="Token length." )
    parser.add_argument("--experiment_name", type=str, default='_', help="Experiment name." )
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size" )
    parser.add_argument("--input_col", type=str, default='', help="Input X column" )
    parser.add_argument("--natural_key", type=str, default='Id', help="Natural key col" )
    args = parser.parse_args()

    return args


class TorchPredictor:
    def __init__(self, model, tokenizer, max_length, natural_key):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.model.eval()
        self.max_length = max_length
        self.natural_key = natural_key

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

            transition_scores = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True).to(
                dtype=torch.float16)
            transition_proba = torch.exp(transition_scores).cpu().numpy()

            return {
                "y_pred": decode_out,
                self.natural_key: batch[self.natural_key],
                "Rev": batch['Rev'],
                "proba": transition_proba[:, 0]
            }




if __name__ == "__main__":
    
    print(f"Ray version: {ray.__version__}")
    
    args = parse_args()
    ctx = ray.data.context.DatasetContext.get_current()
    ctx.use_streaming_executor = True
    ctx.execution_options.preserve_order = True

    # get secs
    KVUri = "https://moaddev6131880268.vault.azure.net"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=KVUri, credential=credential)

    # secret vals
    CLIENTID = client.get_secret("dev-synapse-sqlpool-sp-id").value
    CLIENTSECRET = client.get_secret("dev-synapse-sqlpool-sp-secret").value
    TENANT = client.get_secret("tenant").value

        
    # init Azure ML run context data
    aml_context = Run.get_context()


    ### alt fast method
    async def fetch(file_path):
        """
        Connects to ADLSGen2 and retrieves any file of choice as bytes
        """
        # get access to AAD
        async with ClientSecretCredential(TENANT, CLIENTID, CLIENTSECRET) as credential:
            # get access to ADLS
            async with DataLakeServiceClient(
                account_url=f"https://moaddevlake.dfs.core.windows.net", credential=credential,) as datalake_service_client:
                # get access to FS
                async with datalake_service_client.get_file_system_client(f"moaddevlakefs") as file_system_client:
                    file_client = file_system_client.get_file_client(f"/{file_path}")
                    print(f"Generating data ADLS from file path: {file_path}")
                    # pull and download file
                    download = await file_client.download_file(max_concurrency=75)
                    downloaded_bytes = await download.readall()
                    # report progress
                    print(f"Finished downloading bytes of length: {len(downloaded_bytes)}")
        return downloaded_bytes

    # load data for inference
    print('downloading file...')
    file_paths = [
                #'general/tmp/ticketsnapshotall/ticketsnapshot_inference.parquet',  # one-time all-up data run
                'general/ticketsnapshotall/ticketsnapshot_inference.parquet'  # latest data
                ]

    bytes_df = asyncio.run(asyncio.gather(*map(fetch, file_paths)))
    inference_df = pd.read_parquet(BytesIO(bytes_df[0]))
    print('rx file...', inference_df.shape)


    # clean
    inference_df['ResolutionDetails'] = inference_df['ResolutionDetails'].map(lambda x: re.sub(
        pattern="[\\d\\w]*\\d\\w[\\d\\w]*|[\\d\\w]*\\w\\d[\\d\\w]*", repl='', string=x, flags=re.M))

    inference_df['ResolutionDetails'] = inference_df['ResolutionDetails'].str.replace('\n', ' ')
    inference_df['ResolutionDetails'] = inference_df['ResolutionDetails'].str.replace('\t', ' ')
    inference_df['ResolutionDetails'] = inference_df['ResolutionDetails'].str.replace('\xa0', '')
    inference_df['ResolutionDetails'] = inference_df['ResolutionDetails'].str.strip(' ')

    def scanner_prompt(row):
        '''
        Use Pretrained GDCO Scanner with prompt to look for valid tickets
        '''
        s = (
            f"Determine if this JSON summarization is accurate based on the context below: {row['ResolutionsJson']}. Answer True or False.\n\n"
            f"Context: {row['ResolutionDetails']}\n")
        return s

    def no_issues_prompt(s):
        '''
        Prompt for LLM -- look for no issues found
        '''
        s = (f"""Use the context to determine if any part was replaced.
                Answer True if a part was replaced and False if no part was replaced.\n\n
                Context: {s}
            """)
        return s

    # assign prompt
    inference_df['NoIssuesPrompt'] = inference_df['ResolutionDetails'].map(no_issues_prompt)

    # assign prompt
    inference_df['ScannerPrompt'] = inference_df.apply(scanner_prompt, axis=1)

    if 'NoIssues' in args.experiment_name:
        print('loading pre-trained T5')
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            'google/flan-t5-xl',
            )
        print('loaded tokenizer')
        
        # model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            'google/flan-t5-xl',
            torch_dtype=torch.bfloat16)
        print('loaded model')

    else:
        print('loading fine tuned T5')
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            aml_context.input_datasets['fine_tuned_model'],
            )
        print('loaded tokenizer')
        
        # model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            aml_context.input_datasets['fine_tuned_model'],
            torch_dtype=torch.bfloat16)
        print('loaded model')


    # to ray
    test_ds = ray.data.from_pandas(inference_df).repartition(96)
    

    def tokenize(batch: pd.DataFrame, input_col, block_length, natural_key) -> dict:
        out_batch = tokenizer(
            list(batch[input_col]),
            truncation=True,
            max_length=block_length,
            padding="max_length",
            return_tensors="np",
        )
                                                           
        out_batch[natural_key] = batch[natural_key]
        out_batch['Rev'] = batch['Rev']

        return out_batch


    test_ds_tokenized = test_ds.map_batches(tokenize,
                                            fn_kwargs={"input_col": args.input_col,
                                                        "block_length": args.source_len,
                                                        "natural_key": args.natural_key},
                                            batch_format="pandas")
    print('tokenizing data')

    # get preds
    predictions = test_ds_tokenized.map_batches(TorchPredictor(model=model, tokenizer=tokenizer, max_length=args.target_len, natural_key=args.natural_key), num_gpus=1, batch_size=args.batch_size, compute=ray.data.ActorPoolStrategy(size=args.num_devices))
    
    # to pandas
    predictions_pd = predictions.to_pandas()
    print('preds generated!')
    predictions_pd['experiment_name'] = args.experiment_name

    # set dtype
    predictions_pd[args.natural_key] = predictions_pd[args.natural_key].astype(np.int64)
    predictions_pd['y_pred'] = predictions_pd['y_pred'].astype('string')
    predictions_pd['experiment_name'] = predictions_pd['experiment_name'].astype('string')
    predictions_pd['proba'] = predictions_pd['proba'].astype(np.float32)
    predictions_pd['Rev'] = predictions_pd['Rev'].astype(np.int32)

    print(f"Generated a prediction data set of shape: {predictions_pd.shape}")

    # write
    dating = str(pd.Timestamp('today', tz='UTC'))[:19].replace(' ', '_').replace(':', '_').replace('-', '_')
    predictions_pd.to_parquet(aml_context.output_datasets['output_dir'] + '/' + f'{args.experiment_name}_{dating}.parquet')
    


