import pandas as pd
import numpy as np
import asyncio
import nest_asyncio
import aiohttp
import json
from tenacity import wait_exponential, retry, stop_after_attempt
nest_asyncio.apply()


# api key
api_key = 'api_key'

# model url
url = 'https://cxod-devlab-openai-scus.openai.azure.com/openai/deployments/davinci_mine/completions?api-version=2022-12-01'

# REST headers
headers = {
    "Content-Type": "application/json",
    "api-key": api_key,
}

async def async_send_request(prompt, headers, session, semaphore):
    """
    Leverage semaphore to regulate the amount of requests being made
    """
    async with semaphore:
        if semaphore.locked():
            await asyncio.sleep(5)
        async with session.post(url=url, headers=headers, data=json.dumps(prompt)) as resp:
            return await resp.json()

@retry(wait=wait_exponential(multiplier=2, min=15, max=45), stop=stop_after_attempt(10))
async def async_main_request(semaphore, prompt_list, headers):
    """
    main caller
    """
    s = asyncio.Semaphore(value=semaphore)
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*[async_send_request(prompt=prompt, headers=headers, session=session, semaphore=s) for prompt in prompt_list])


# set requirements
prompt_prefix = 'Summarize:\n'
prompt_ending = '\n\n###\n\n'

# load data
df = pd.read_csv(r'data/raw_gpu.csv')

# add in summarization
df['summarize'] = prompt_prefix + df['Comments'] + prompt_ending

# build prompt set
prompt_list = [{
    "prompt": val,
    "temperature": 0.3,
    "max_tokens": 128,
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "best_of": 1,
} for val in df['summarize'].values]

# need to batch most likely
prompt_list = np.array_split(prompt_list, 50, axis=0)
batch_set = []

# for each batch,
for i, batch in enumerate(prompt_list):
    print(f'now on batch {i}')
    loop = asyncio.get_event_loop()
    out = loop.run_until_complete(async_main_request(semaphore=64, prompt_list=batch, headers=headers))
    batch_set.append(out)


# collapse
pd.json_normalize(pd.concat([pd.json_normalize(_)['choices'].explode() for _ in batch_results]).reset_index(drop=True))['text'].to_frame()

# non async request
import requests
post_data = {
    "prompt": prompt,
    "temperature": 0.3,
    "max_tokens": 128,
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "best_of": 1,
}

resp = requests.post(url="https://cxod-devlab-openai-scus.openai.azure.com/openai/deployments/davinci_mine/completions?api-version=2022-12-01",
              headers=headers,
              data=json.dumps(post_data)
              )

resp.json()['choices'][0]['text']
