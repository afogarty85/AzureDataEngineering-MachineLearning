from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import os
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
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
import re
import json
import gc
from accelerate import Accelerator
import os  
from transformers import pipeline
from datasets import Dataset  
from datasets import load_dataset, interleave_datasets
torch.backends.cuda.matmul.allow_tf32 = True


# init Azure ML run context data
aml_context = Run.get_context()
device = 'cuda'

output_path = aml_context.output_datasets['output_dir']
model_path = aml_context.input_datasets['model_path']
train_files = aml_context.input_datasets['train_files']



def parse_args():

    parser = argparse.ArgumentParser(description="Simple example of eval script.")
    parser.add_argument("--num_devices", type=int, default=1, help="Number of devices to use.")
    parser.add_argument("--ctx_len", type=int, default=512, help="Token length." )
    parser.add_argument("--input_col", type=str, default='prompt', help="X." )
    parser.add_argument("--padding", type=str, default='max_length', help="Padding" )
    parser.add_argument("--experiment_name", type=str, default='_', help="Experiment name." )
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size" )
    args = parser.parse_args()

    return args


def ticket_clean_fn(df):
    df['ResolutionDetails'] = df['ResolutionDetails'].map(lambda x: re.sub(
        pattern="[\\d\\w]*\\d\\w[\\d\\w]*|[\\d\\w]*\\w\\d[\\d\\w]*", repl='', string=x, flags=re.M) if pd.notnull(x) else x)
    df['ResolutionDetails'] = df['ResolutionDetails'].str.replace('\n', ' ')
    df['ResolutionDetails'] = df['ResolutionDetails'].str.replace('\t', ' ')
    df['ResolutionDetails'] = df['ResolutionDetails'].str.replace('\xa0', '')
    df['ResolutionDetails'] = df['ResolutionDetails'].str.strip(' ')
    df['ResolutionDetails'] = df['ResolutionDetails'].str.replace(r'&nbsp', ' ', regex=True) \
        .str.replace(r'\s\s+', ' ', regex=True) \
        .str.lstrip(' ') \
        .str.rstrip(' ')
    
    df['ResolutionDetails'] = df['ResolutionDetails'].fillna('')
    return df



class HuggingFacePredictor:
    def __init__(self):
        
        model_path = ray.get(model_path_ref)  
        self.model = pipeline("text-generation", model=model_path, tokenizer=tokenizer, torch_dtype=torch.float16, device_map='auto', batch_size=args.batch_size)

    # batch inference
    def __call__(self, batch):

        with torch.no_grad():
            predictions = self.model(list(batch["text"]), **gen_sample_kwargs)

            # split / decode
            decode_out = [sequences[0]["generated_text"].replace('JSON Structure:\n', '').replace('Correct Score:\n', '') for sequences in predictions]

            return {
                    "y_pred": decode_out,
                    "ResolutionDetails": batch['ResolutionDetails'],
                    "ResolutionsJson": batch['ResolutionsJson'],
                    "TicketId": batch['TicketId'],
                    "Task": batch['task'],
                }






print(f"Ray version: {ray.__version__}")

args = parse_args()
ctx = ray.data.context.DatasetContext.get_current()
ctx.use_streaming_executor = True
ctx.execution_options.preserve_order = False
ctx.execution_options.locality_with_output = True


# get secs
KVUri = "https://moaddev6131880268.vault.azure.net"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)

# secret vals
CLIENTID = client.get_secret("dev-synapse-sqlpool-sp-id").value
CLIENTSECRET = client.get_secret("dev-synapse-sqlpool-sp-secret").value
TENANT = client.get_secret("tenant").value


# get latest data
def generate_latest_from_kusto(query):
    '''
    Get latest data from kusto to build inference table against
    '''

    # kusto cluster
    cluster = 'https://datagalaxy.westus3.kusto.windows.net'
    # kusto db
    kusto_db = 'COIDG'
    # service principal: get AAD auth
    kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(
        cluster, CLIENTID, CLIENTSECRET, TENANT
    )
    client = KustoClient(kcsb)

    # send query
    response = client.execute(kusto_db, query)

    # get data
    table = response.primary_results[0]
    columns = [col.column_name for col in table.columns]
    df = pd.DataFrame(table.raw_rows, columns=columns)
    return df


# get last n days of tickets
# ====================================================

# get tickets
build_query = (f'''
set maxmemoryconsumptionperiterator=68719476736;
set max_memory_consumption_per_query_per_node=68719476736;
set notruncation;
cluster('https://datagalaxy.westus3.kusto.windows.net').database('COIDG').TicketRevisions
| project TicketId, Rev, CreatedDate, WorkEndDate, DeviceOperationalState, State,
            ResolutionDetails, ResolutionsJson
| where State == 'Resolved'
| where DeviceOperationalState == 'Production'
| where WorkEndDate > ago(14d)
| where ResolutionDetails != ''
| summarize arg_max(Rev,*) by TicketId
''')


# get data
print('Getting data from Kusto')
df = generate_latest_from_kusto(build_query)
print(f"Generated a DF of shape: {df.shape} \n")



# model / tokenizer
# ====================================================
tokenizer = AutoTokenizer.from_pretrained(model_path, padding='max_length', truncation=True, max_length=args.ctx_len)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
model_path_ref = ray.put(model_path)  



# preprocess
# ====================================================
df = ticket_clean_fn(df)

# limit long ResolutionsJson
df['ResolutionsJson'] = df['ResolutionsJson'].str.slice(stop=256)  

# Revised prompt template  
def map_instruct_prompt(row):  
    ticket_notes = row['ResolutionDetails']  
    ticket_json = row['ResolutionsJson']  
  
    chat_list = [  
        {  
            "role": "system",  
            "content": (  
                "Your task is to analyze the ticket notes provided and create a JSON structure with only the "  
                "specified 'Action' and 'ChimeraType' fields. Use the valid options provided to determine the "  
                "correct values for these fields. If the information is unclear or incomplete, use 'Unclear'. "  
                "If the information is not provided, use 'Not Provided'.\n\n"  
                "Valid 'Action' options: Execute Software, Test, Clean, Shutdown, Reboot, Reseat, Move, Inspect, "  
                "Update Data, Change, Investigate, Startup, Replace, Configure\n\n"  
                "Valid 'ChimeraType' options: PLR4 Cable, RackManagementSwitch, CMOS Battery, PDU or UPS, E1.S Drive, "  
                "UPS, PDU, Hard Disk Drive, Network card, Array Cache, NVRAM, Power Cables, SAN Controller Module, "  
                "DC-SCM, FPGA, GPU Tray, M.2 Drive, Array Battery, Chassis Manager, Corsica Card, Server Blade, BMC, "  
                "OCulink Cable, FusionIO Card, Heatsink, InfiniBand Cable, OAM Rear, Fan, Single Mode Fiber Cable, "  
                "Rack Manager, rPDU, Cerberus Security Card, Power Supply, Processor, Copper Network Cable, System Board, "  
                "NVDIMM Battery, GPU Module, Network Device, AEC Y Cable, Other Part, Backplane/Riser Card, Network "  
                "Device, Switchboard, DIMM, SFP Optic, Array Controller, TPM, Multi Mode Fiber Cable, Linking Board, "  
                "PDB, PSM4 Cable, HIB, Internal Cable, Server Chassis, Network Module, DCPMM, Cerberus Security Card, "  
                "AOC Cable, Row Manager, Flash Drive, DAC Cable, NVDIMM, Solid State Drive, OAM Front\n\n"  
                "Below are examples of how to format the JSON structure based on given ticket notes:\n\n"  
                "Example 1:\n"  
                "Ticket Notes: \"The server blade in rack 3A requires a reboot due to a system freeze.\"\n"  
                "JSON Structure: [{\"Action\":\"Reboot\",\"ChimeraType\":\"Server Blade\"}]\n\n"  
                "Example 2:\n"  
                "Ticket Notes: \"Network card in rack 11B is malfunctioning and needs replacement.\"\n"  
                "JSON Structure: [{\"Action\":\"Replace\",\"ChimeraType\":\"Network card\"}]"  
            )  
        },  
        {  
            "role": "user",  
            "content": f"Ticket Notes:\n{ticket_notes}"  
        }
    ]  
  
    return chat_list  

def map_accuracy_prompt(row):  
    ticket_notes = row['ResolutionDetails']  
    ticket_json = row['ResolutionsJson']  
  
    # System instruction content  
    system_content = (  
        "You are tasked with evaluating the accuracy of JSON data when compared to provided ticket notes. "  
        "Your evaluation will be based on two specific fields in the JSON: 'Action' and 'ChimeraType'. "  
        "You will use a binary scoring system to assess the accuracy. Do not provide any partial score.\n\n"  
        "- **Score of 1**: Both the 'Action' and 'ChimeraType' in the JSON match exactly with the ticket notes.\n"  
        "- **Score of 0**: If any field does not match the ticket notes.\n\n"  
        "Valid JSON Options\n\n"  
        "Valid 'Action' options include: Execute Software, Test, Clean, Shutdown, Reboot, Reseat, Move, Inspect, "  
        "Update Data, Change, Investigate, Startup, Replace, Configure.\n"  
        "Valid 'ChimeraType' options include: PLR4 Cable, RackManagementSwitch, CMOS Battery, PDU or UPS, "  
        "E1.S Drive, UPS, PDU, Hard Disk Drive, Network card, Array Cache, NVRAM, Power Cables, SAN Controller Module, "  
        "DC-SCM, FPGA, GPU Tray, M.2 Drive, Array Battery, Chassis Manager, Corsica Card, Server Blade, BMC, "  
        "OCulink Cable, FusionIO Card, Heatsink, InfiniBand Cable, OAM Rear, Fan, Single Mode Fiber Cable, "  
        "Rack Manager, rPDU, Cerberus Security Card, Power Supply, Processor, Copper Network Cable, System Board, "  
        "NVDIMM Battery, GPU Module, Network Device, AEC Y Cable, Other Part, Backplane/Riser Card, Network "  
        "Device, Switchboard, DIMM, SFP Optic, Array Controller, TPM, Multi Mode Fiber Cable, Linking Board, "  
        "PDB, PSM4 Cable, HIB, Internal Cable, Server Chassis, Network Module, DCPMM, Cerberus Security Card, "  
        "AOC Cable, Row Manager, Flash Drive, DAC Cable, NVDIMM, Solid State Drive, OAM Front.\n\n"  
        "Training Examples\n\n"  
        "Do not include these examples in your final answer. They are for guidance only.\n\n"  
        "Example 1:\n"  
        "Ticket Notes: Failure(s) Found: Switchboard; Repair Action(s): Replaced\n"  
        "JSON: [{\"Action\":\"Replace\",\"ChimeraType\":\"Switchboard\"}]\n"  
        "Correct Score: 1\n\n"  
        "Example 2:\n"  
        "Ticket Notes: Swapped the DIMMs\n"  
        "JSON: [{\"Action\":\"Replace\",\"ChimeraType\":\"Processor\"}]\n"  
        "Correct Score: 0"  
    )  
  
    # Create the chat list with the formatted messages  
    chat_list = [  
        {  
            "role": "system",  
            "content": system_content  
        },  
        {  
            "role": "user",  
            "content": f"Ticket Notes:\n{ticket_notes}\n\nJSON Data:\n{ticket_json}"  
        }
    ]  
  
    return chat_list  

# map
df['prompt'] = df.apply(map_instruct_prompt, axis=1)
df['task'] = 'JSON Prediction'

accuracy_df = df.copy()
accuracy_df['prompt'] = accuracy_df.apply(map_accuracy_prompt, axis=1)
accuracy_df['task'] = 'JSON Accuracy'

# join
df = pd.concat([df, accuracy_df], axis=0).reset_index(drop=True)
print(f"Generated a concat DF of shape: {df.shape} \n")

# convert to chat
def preprocess_function(example):
    messages = example["prompt"]

    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return example

# temp inject
# df = pd.read_parquet(train_files + '/ticket_snapshot_inference_base.parquet').sample(frac=1).reset_index(drop=True)
# #df = df[:5000]
# print(f"Temp insert shape: {df.shape}")

# to HF ds
ds = Dataset.from_pandas(df)  

# map template
ds = ds.map(
    preprocess_function,
    num_proc=24,
    desc="Applying chat template",
)

# to ray data
df = ray.data.from_huggingface(ds).repartition(16)



# inference
# ====================================================

# Generation-specific parameters  
gen_sample_kwargs = {  
    "max_new_tokens": 125,
    "num_beams": 4,  
    "num_return_sequences": 1,  
    "do_sample": False,
    "return_full_text": False,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
}


# get preds
print(f"Using this batch size {args.batch_size}")
predictions = df.map_batches(HuggingFacePredictor, num_gpus=1, batch_size=args.batch_size, compute=ray.data.ActorPoolStrategy(size=args.num_devices))


# write and format
# ====================================================
predictions_pd = predictions.to_pandas()
print('preds generated!')

predictions_pd['experiment_name'] = args.experiment_name

# set dtype
print(predictions_pd.columns)
for col in predictions_pd.columns:
    if col in ['y_pred', 'experiment_name', 'task']:
        predictions_pd[col] = predictions_pd[col].astype('string')
    
    elif col in ['TicketId']:
        predictions_pd[col] = predictions_pd[col].astype(np.int64)


print(f"Generated a prediction data set of shape: {predictions_pd.shape}")

# disk
dating = str(pd.Timestamp('today', tz='UTC'))[:19].replace(' ', '_').replace(':', '_').replace('-', '_')
predictions_pd.to_parquet(output_path + '/' + f'{args.experiment_name}_{dating}.parquet')



