import pickle
import json  
import re
import ast
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
import numpy as np
from datasets import load_dataset


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
        },  
        {  
            "role": "assistant",  
            "content": f"JSON Structure:\n{ticket_json}"  
        }  
    ]  
  
    return chat_list  



# apply
instruct_tune_df['prompt'] = instruct_tune_df.apply(map_instruct_prompt, axis=1)


# Revised prompt template  
def map_accuracy_prompt(row):  
    ticket_notes = row['ResolutionDetails']  
    ticket_json = row['ResolutionsJson']  
    ticket_score = row['Score']  
  
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
        },  
        {  
            "role": "assistant",  
            "content": f"Correct Score:\n{ticket_score}"  
        }  
    ]  
  
    return chat_list  

# apply
accuracy_df['prompt'] = accuracy_df.apply(map_accuracy_prompt, axis=1)


# combine
instruct_tune_df = pd.concat([instruct_tune_df, accuracy_df], axis=0).reset_index(drop=True)


# shuffle
instruct_tune_df = instruct_tune_df.sample(frac=1).reset_index(drop=True)
instruct_tune_df.to_parquet('./data/causal_lm_sft_zephyr_allup.parquet')  ## 1030 ctx len

tokenizer = AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta',
                                          use_fast=True)


ds = load_dataset("parquet", data_files='./data/causal_lm_sft_zephyr.parquet', split='train', num_proc=96).train_test_split(test_size=0.1, seed=4)

instruct_tune_df['prompt'].sample(n=1).item()

def preprocess_function(example):
    messages = example["prompt"]

    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return example

ds['train'] = ds['train'].map(
    preprocess_function,
    num_proc=8,
    desc="Applying chat template",
)

ds['test'] = ds['test'].map(
    preprocess_function,
    num_proc=8,
    desc="Applying chat template",
)

test_message = ds['test']['prompt'][355]
tokenized_chat = tokenizer.apply_chat_template(test_message, tokenize=True, add_generation_prompt=True)

# check stats
dataset = Dataset.from_pandas(instruct_tune_df)

def tokenize_and_get_length_batched(batch):  
    lengths = []  
    for prompts in batch['prompt']:  
        # Concatenate all 'content' from the list of dictionaries into a single string  
        concatenated_content = ' '.join(entry['content'] for entry in prompts)  
        # Tokenize the concatenated content  
        tokenized_output = tokenizer(concatenated_content, truncation=False)  
        # Append the length of the tokenized content to the lengths list  
        lengths.append(len(tokenized_output['input_ids']))  
    return {'length': lengths}  

  
# Assuming 'ds' is your loaded dataset and 'train' is the split you're working with  
lengths_dataset = dataset.map(tokenize_and_get_length_batched, batched=True, num_proc=8)  
  
# Now, 'lengths_dataset' contains a new column 'length' with the token lengths for each entry  
lengths = sorted(lengths_dataset['length'])  

percentile_index = round(len(lengths) * 0.999) - 1
length_at_percentile = lengths[percentile_index]  
print(f"99.9th percentile token length: {length_at_percentile}")    # 841


