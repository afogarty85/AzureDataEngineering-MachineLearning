import json
import pandas as pd
import re
from datasets import Dataset
from transformers import (
    AutoTokenizer,
)
from datasets import load_dataset                   


# load ds
df = pd.read_parquet(r'data/gpt4_scored_full_sample.parquet')

# remove S/Ns
df['ResolutionDetails'] = df['ResolutionDetails'].map(lambda x: re.sub(pattern="[\\d\\w]*\\d\\w[\\d\\w]*|[\\d\\w]*\\w\\d[\\d\\w]*", repl='', string=x, flags=re.M))

# get top scores
df = df.query('score == 5').reset_index(drop=True)

# filter
df = df[['ResolutionDetails', 'ResolutionsJson']].drop_duplicates().reset_index(drop=True)

# # look at sn
# df[df['ResolutionDetails'].str.contains('SN', case=False)].sample(n=2)['ResolutionDetails'].values

# hf ds
hf_ds = Dataset.from_pandas(df[['ResolutionDetails', 'ResolutionsJson']]).train_test_split(0.2, seed=4)



# NO INSTRUCT
# gen data jsonl
for key, ds in hf_ds.items():
    with open(f"./data/{key}_noinstruct.jsonl", "w") as f:
        for item in ds:
            newitem = {}
            newitem["input"] = (
                    f"<START_Q>{item['ResolutionDetails']}<END_Q>"
                    f"<START_A>{item['ResolutionsJson']}<END_A>"
                )
            f.write(json.dumps(newitem) + "\n")



# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    legacy=True,
    add_eos_token=False,
    add_bos_token=False,
    token='')
tokenizer.pad_token = tokenizer.eos_token


# check size
hf_ds = load_dataset("json", data_files='data/test_noinstruct.jsonl', split="train")


def generate_stats(ds):
    lengths = sorted(len(lst) for lst in ds['input_ids'])
    curr_info = dict(min=lengths[0], max=lengths[-1], median=lengths[len(lengths) // 2])
    curr_info.update({"99_percentile": lengths[round(len(lengths) * 0.99)]})
    return curr_info

# get input lengths
out = hf_ds.map(lambda examples: tokenizer(examples['input'], truncation=False),
                               batched=True,
                               num_proc=8)


generate_stats(out)  # 269





# gen system message
system_message = """Given the input, construct a valid JSON with keys and values. The JSON must accurately describe the input. Ignore references to serial numbers or its common abbreviations while interpreting the input.
A valid JSON has two possible keys: Action and ChimeraType. 
An valid Action can contain any of the following: .
A valid ChimeraType can contain any of the following: 
"""


# gen data jsonl
for key, ds in hf_ds.items():
    with open(f"./data/{key}_oldinstruct.jsonl", "w") as f:
        for item in ds:
            newitem = {}
            newitem["input"] = (
                "<s>[INST] <<SYS>>\n"
                f"{system_message}"
                "<</SYS>>\n"
                f"{item['ResolutionDetails']} [/INST] {item['ResolutionsJson']} </s>"
            )
            f.write(json.dumps(newitem) + "\n")




# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    legacy=True,
    add_eos_token=False,
    add_bos_token=False,
    token='')
tokenizer.pad_token = tokenizer.eos_token



# check size
hf_ds = load_dataset("json", data_files='data/test_oldinstruct.jsonl', split="train")


def generate_stats(ds):
    lengths = sorted(len(lst) for lst in ds['input_ids'])
    curr_info = dict(min=lengths[0], max=lengths[-1], median=lengths[len(lengths) // 2])
    curr_info.update({"99_percentile": lengths[round(len(lengths) * 0.99)]})
    return curr_info

# get input lengths
out = hf_ds.map(lambda examples: tokenizer(examples['input'], truncation=False),
                               batched=True,
                               num_proc=8)


generate_stats(out)  # 260 // 269



# new instruct


# gen system message
sys_message = f"""You are a JSON generator. Always output your answer in JSON. For each Text, construct a JSON output that accurately describes what happened. Ignore references to serial numbers or its common abbreviations while interpreting the Text.
A valid JSON has two possible keys: Action and ChimeraType. 
An valid Action can contain any of the following: .
A valid ChimeraType can contain any of the following: 
Instead of including Output in your answer, your answer should include a valid JSON.
A couple of examples are below.

Example 1)
Text: 
Output: 

Example 2)
Text: 
Output: 

Example 3)
Text: 
Output: 
"""


# gen data jsonl
for key, ds in hf_ds.items():
    with open(f"./data/{key}_newinstruct.jsonl", "w") as f:
        for item in ds:
            newitem = {}
            newitem["input"] = (
                "<s>[INST] <<SYS>>\n"
                f"{sys_message}"
                "<</SYS>>\n"
                f"Text: {item['ResolutionDetails']} [/INST]"
                "\n"
                f"Output: {item['ResolutionsJson']} </s>"
            )
            f.write(json.dumps(newitem) + "\n")


