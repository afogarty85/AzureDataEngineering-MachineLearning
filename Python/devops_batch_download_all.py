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
                    return await resp.json()

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
            url_post_list = [f"Select [System.Id] From WorkItems where [System.TeamProject] = @project AND [System.ChangedDate] <= @today-{i} AND [System.ChangedDate] >= @today-{i}" for i in range(0, 3)]
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
        workItem_set = pd.concat([pd.json_normalize(_['workItems']) for _ in out])['id'].values
        self.logger.info(f'Found this many work items: {workItem_set.shape[0]}')

        # get work item history urls to send
        url_list = [self.url + f'_apis/wit/workitems/{id}/revisions?$expand=all&$skip=0&api-version=6.0' for id in workItem_set]
        
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

        self.logger.info('Unpacking work items...')
        # unpack the work ids
        workItems = pd.concat([pd.json_normalize(_['value']) for _ in batch_set])

        self.logger.info('Finding work items that need more revisions...')
        # get # of revs from each work id
        revs = pd.concat([pd.json_normalize(_) for _ in batch_set])['count'].values

        # max rev count scalar
        max_rev_count = 200

        # if true, then we have more searching to do
        if max_rev_count in revs:
            self.logger.info('Looking for revisions...')
            revision_storage = []
            # set rev count
            rev_count = 200
            while max_rev_count in revs:
                self.logger.info(f'now on rev: {rev_count}')

                # filter ids to those we need
                retry_list = [200 if i == 200 else 0 for i in revs]
                indices_of_interest = [i for i, j in enumerate(retry_list) if j != 0]
                ids_of_interest = [j for i, j in enumerate(workItem_set) if i in indices_of_interest]

                self.logger.info(f'Gettings revs for this many IDs: {len(ids_of_interest)}...')

                # build new get list
                url_list = [self.url + f'_apis/wit/workitems/{id}/revisions?$expand=all&$skip={rev_count}&api-version=6.0' for id in ids_of_interest]

                # get the first set
                loop = asyncio.get_event_loop()
                out = loop.run_until_complete(self.async_main_request(url=self.url, semaphore=64, req_list=url_list, headers=self.headers, type='get'))

                # get the work items
                workItems_extended = pd.concat([pd.json_normalize(_['value']) for _ in batch_set])

                # append
                revision_storage.append(workItems_extended)

                # get new set of revs and ids
                rev_ids = pd.concat([pd.json_normalize(_['value']) for _ in out]).groupby(['id']).size().reset_index(name='count')

                # current set of workitems
                workItem_set = rev_ids['id'].values

                # current revs
                revs = rev_ids['count'].values

                # update skip list
                rev_count += 200

            # concat rev results at the end
            self.logger.info('Concatenating rev results...')
            workItems_extended = pd.concat(_ for _ in revision_storage)

            # concat into main workitems
            self.logger.info('Concatenating revs + total work items...')
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
