import pandas as pd
import aiohttp
import asyncio
import requests
import numpy as np
import nest_asyncio
from tenacity import wait_exponential, retry, stop_after_attempt
nest_asyncio.apply()



class ado_generator_by_batch():
    '''
    Generate Full ADO Work Items via Batch Processing

    Azure DevOps has limits on how many records we can pull at one time;
    we solve this problem by slicing our search windows

    Sample Use

    # load processor
    processor = generate_ado_batch(headers, url, payload, config)

    # get all work ids
    raw = processor.execute_ado_operations()

    '''
    def __init__(self, config, logger, args):
        self.config = config
        self.headers = config.headers
        self.url = config.url
        self.logger = logger

    async def async_send_request(self, url, req, headers, type, session, semaphore):
        """
        Leverage semaphore to regulate the amount of requests being made
        We only want revisions as a field of interest
        """
        if type == 'get':
            async with semaphore:
                if semaphore.locked():
                    await asyncio.sleep(1)
                async with session.get(url=req, headers=self.headers) as resp:
                    return await resp.json()

        if type == 'post':
            async with semaphore:
                if semaphore.locked():
                    await asyncio.sleep(1)
                async with session.post(url=self.url + "_apis/wit/wiql?api-version=6.0", headers=self.headers, json={"query": req}) as resp:
                    return await resp.json()

    @retry(wait=wait_exponential(multiplier=2, min=15, max=45), stop=stop_after_attempt(10))
    async def async_main_request(self, url, semaphore, req_list, headers, type):
        """
        main caller
        """
        s = asyncio.Semaphore(value=semaphore)
        async with aiohttp.ClientSession() as session:
            return await asyncio.gather(*[self.async_send_request(url=self.url, req=req, headers=self.headers, type=type, session=session, semaphore=s) for req in req_list])

    def execute_ado_operations(self):
        '''
        (1): get work ids
        (2): get work items
        (3): get work items with excessive revisions
        (4): join (2) and (3)
        (5): clean up
        '''

        # time travel to 2017 feb
        url_post_list = [f"Select [System.Id] From WorkItems where [System.TeamProject] = @project AND [System.CreatedDate] <= @today-{i} AND [System.CreatedDate] >= @today-{21+i}" for i in range(0, 1900, 22)]

        # get work ids
        loop = asyncio.get_event_loop()
        out = loop.run_until_complete(self.async_main_request(url=self.url, semaphore=1000, req_list=url_post_list, headers=self.headers, type='post'))
        workitem_set = pd.concat([pd.json_normalize(_['workItems']) for _ in out]).reset_index(drop=True)['id'].values

        # get work item history
        url_list = [self.url + f'_apis/wit/workitems/{id}/revisions?$expand=all&$skip=0&api-version=6.0' for id in workitem_set]
        # but batch it
        url_list = np.array_split(url_list, 50, axis=0)
        batch_set = []

        # for each batch,
        for i, batch in enumerate(url_list):
            self.logger.info(f'now on batch {i}')
            loop = asyncio.get_event_loop()
            out = loop.run_until_complete(self.async_main_request(url=self.url, semaphore=64, req_list=batch, headers=self.headers, type='get'))
            batch_set.append(out)

        # combine batch_set
        batch_set = list(sum(batch_set, []))

        self.logger.info('Generating work ids and those that need revs...')
        # get # of revs from each work id
        total_revs = pd.concat([pd.json_normalize(_)['count'] for _ in batch_set])
        retry_list = [200 if i == 200 else 0 for i in total_revs.values]

        self.logger.info('Unpacking work items...')
        # unpack the work ids
        workItems = pd.concat([pd.json_normalize(_['value']) for _ in batch_set])

        # send new requests for those that broke 200 revs
        indices_of_interest = [i for i, j in enumerate(retry_list) if j != 0]
        ids_of_interest = [j for i, j in enumerate(workitem_set) if i in indices_of_interest]

        # rebuild new urls
        self.logger.info(f'Sending these extra revs {len(ids_of_interest)}...')
        url_retry_list = [self.url + f'_apis/wit/workitems/{id}/revisions?$expand=all&$skip=200&api-version=6.0' for id in ids_of_interest]
        loop = asyncio.get_event_loop()
        out = loop.run_until_complete(self.async_main_request(url=self.url, semaphore=64, req_list=url_retry_list, headers=self.headers, type='get'))

        # get extended items
        workItems_extended = pd.concat([pd.json_normalize(_['value']) for _ in out])

        # concat final results
        self.logger.info('Concatenating final results...')
        workItems = pd.concat([workItems_extended, workItems], axis=0)

        # drop duplicates
        self.logger.info('Cleaning up the dataframe...')
        workItems = workItems.drop_duplicates(['id', 'rev']).reset_index(drop=True)

        # deal with 'fields' in cols
        workItems.columns = workItems.columns.str.replace('fields.', '', regex=False)

        # truncate cols
        if self.config.cols is not None:
            workItems = workItems[self.config.cols]

        # drop cols
        try:
            workItems = workItems.drop(self.config.drop, axis=1)
        except Exception as e:
            self.logger.error(f" {e}")

        # find unnecessary cols
        drop_list = workItems.filter(regex='.descriptor|.id|.imageUrl|.href').columns.tolist()

        # drop them
        workItems = workItems.drop(drop_list, axis=1)

        # find html cols that are excessively long
        s_cols = [x for x, d in zip(workItems.columns, workItems.dtypes.apply(lambda x: x.name)) if d == 'object']

        # look for html and edit it out
        for col in s_cols:
            if workItems[col].astype(str).str.contains("<span").any():
                workItems[col] = workItems[col].str.replace('<[^<]+?>', ' ', regex=True)

        return workItems
