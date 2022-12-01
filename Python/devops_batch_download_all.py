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
        self.args = args

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
                    return await resp.read()

        if type == 'post':
            async with semaphore:
                if semaphore.locked():
                    await asyncio.sleep(1)
                async with session.post(url=self.url + "_apis/wit/wiql?timePrecision=true&api-version=7.0", headers=self.headers, json={"query": req}) as resp:
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
        def date_imputer(days):
            ''' gen dates for work id searching '''
            the_date = (pd.Timestamp("today", tz="US/Eastern") - datetime.timedelta(days=days))
            return the_date.strftime('%Y-%m-%d')

        if self.args.lastrev:
            self.logger.info('Getting work items that have been modified in the last 3 days...')
            # last 3 days
            url_post_list = [f"Select [System.Id] From WorkItems where [System.TeamProject] = @project AND [System.ChangedDate] <= @today-{i} AND [System.ChangedDate] >= @today-{3}" for i in range(0, 11)]
        else:
            self.logger.info('Grabbing entire history of work items...')
            # since jan 1 2019; fairly obtuse way to get lots of workids owing to devops arbitray limitations
            url_post_list1 = [f"Select [System.Id] From WorkItems where [System.TeamProject] = @project AND [System.CreatedDate] >= '{date_imputer(i)}T00:00:00.0000000' AND [System.CreatedDate] <= '{date_imputer(0+i)}T08:00:00.0000000'" for i in range(0, 1429, 1)]
            url_post_list2 = [f"Select [System.Id] From WorkItems where [System.TeamProject] = @project AND [System.CreatedDate] >= '{date_imputer(i)}T08:00:00.0000000' AND [System.CreatedDate] <= '{date_imputer(0+i)}T14:59:59.9999999'" for i in range(0, 1429, 1)]
            url_post_list3 = [f"Select [System.Id] From WorkItems where [System.TeamProject] = @project AND [System.CreatedDate] >= '{date_imputer(i)}T14:59:59.9999999' AND [System.CreatedDate] <= '{date_imputer(0+i)}T23:59:59.9999999'" for i in range(0, 1429, 1)]
            url_post_list = url_post_list1 + url_post_list2 + url_post_list3

        # get work ids
        loop = asyncio.get_event_loop()
        out = loop.run_until_complete(self.async_main_request(url=self.url, semaphore=64, req_list=url_post_list, headers=self.headers, type='post'))
        workitem_set = pd.concat([pd.json_normalize(_['workItems']) for _ in out]).reset_index(drop=True)['id'].values
        self.logger.info(f'Found this many work items: {workitem_set.shape[0]}')

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

        # filter
        batch_set = [i for i in batch_set if len(i) != 0 and i[:8].decode('utf-8') == '{"count"']

        self.logger.info('Generating work ids and those that need revs...')
        # get # of revs from each work id
        total_revs = pd.concat([pd.json_normalize(json.loads(_)) for _ in batch_set])['count']
        retry_list = [200 if i == 200 else 0 for i in total_revs.values]

        self.logger.info('Unpacking work items...')
        # unpack the work ids
        workItems = pd.concat([pd.json_normalize(json.loads(_)['value']) for _ in batch_set])

        # send new requests for those that broke 200 revs
        indices_of_interest = [i for i, j in enumerate(retry_list) if j != 0]
        ids_of_interest = [j for i, j in enumerate(workitem_set) if i in indices_of_interest]

        # rebuild new urls
        if len(ids_of_interest) > 0:
            self.logger.info(f'Sending these extra revs {len(ids_of_interest)}...')
            url_retry_list = [self.url + f'_apis/wit/workitems/{id}/revisions?$expand=all&$skip=200&api-version=6.0' for id in ids_of_interest]
            loop = asyncio.get_event_loop()
            out = loop.run_until_complete(self.async_main_request(url=self.url, semaphore=64, req_list=url_retry_list, headers=self.headers, type='get'))

            # get extended items
            workItems_extended = pd.concat([pd.json_normalize(json.loads(_)['value']) for _ in out])

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

        # drop cols identified in config
        try:
            workItems = workItems.drop(self.config.drop, axis=1)
        except Exception as e:
            self.logger.error(f" {e}")

        # find unnecessary cols
        drop_list = workItems.filter(regex='.descriptor|.id|.imageUrl|.href|.url').columns.tolist()

        # drop them
        workItems = workItems.drop(drop_list, axis=1)

        # find html cols that are excessively long
        s_cols = [x for x, d in zip(workItems.columns, workItems.dtypes.apply(lambda x: x.name)) if d == 'object']

        # look for html and edit it out
        for col in s_cols:
            if workItems[col].astype(str).str.contains("<span").any():
                workItems[col] = workItems[col].str.replace('<[^<]+?>', ' ', regex=True)

        # reset index again
        workItems = workItems.reset_index(drop=True)

        self.logger.info(f'Saving a dataframe of this shape: {workItems.shape}')

        return workItems
