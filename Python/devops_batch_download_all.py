import pandas as pd
import aiohttp
import asyncio
import requests
import numpy as np
import nest_asyncio
from tenacity import wait_exponential, retry, stop_after_attempt
nest_asyncio.apply()


class generate_ado_batch():
    '''
    Generate Full ADO Work Items via Batch Processing

    Azure DevOps has limits on how many records we can pull at one time;
    we solve this problem by slicing our search windows

    Sample Use

    # load processor
    batch_processor = generate_ado_batch(headers, url, payload, config)
    # get all work ids
    work_id_groups = batch_processor.find_workids()
    # start loop
    out = batch_processor.generate_ado_data_by_batch(work_groups=work_id_groups)
    '''
    def __init__(self, config, logger, args):
        self.config = config
        self.headers = config.headers
        self.url = config.url
        self.payload = config.wiql

    @retry(wait=wait_exponential(multiplier=2, min=15, max=45),
           stop=stop_after_attempt(10))
    def retry_workitems(self, wiql_payload):
        try:
            response = requests.post(url=self.url + '_apis/wit/wiql?api-version=6.0',
                                     json=wiql_payload,
                                     headers=self.headers)

        except Exception as excep:
            print(f"Exception: {excep}")
        return response

    def find_workids(self):
        # institute 3 week window searches
        delta_time = 21
        base_time = 0
        # create empty storage df
        storage_df = pd.DataFrame()
        # search this total range of days
        total_search = (pd.Timestamp("today") - pd.Timestamp('2018-01-01')).days

        while delta_time <= total_search:
            # report findings
            print(f'now searching delta time: {delta_time}')

            # create time windows
            base_date = str(pd.Timestamp('today', tz='US/Pacific') - pd.Timedelta(base_time, unit='D'))[:10]
            delta_date = str(pd.Timestamp('today', tz='US/Pacific') - pd.Timedelta(delta_time, unit='D'))[:10]

            # search in slices by getting all IDs
            query = f"""
            SELECT
                [System.Id]
            FROM workitems
            WHERE [System.TeamProject] = @project
            AND [System.CreatedDate] < '{base_date}'
            AND [System.CreatedDate] >= '{delta_date}'
            """

            # turn query into a wiql payload
            payload_wiql = {}
            payload_wiql['query'] = query

            # update search time
            delta_time += 21
            base_time += 21

            # send API request with failure retry
            response = self.retry_workitems(payload_wiql)

            # if no response, go again
            if not response.ok:
                continue
            try:
                # otherwise extra work items and save them to storage
                out1 = pd.json_normalize(response.json()['workItems'])[['id']]
                storage_df = pd.concat([storage_df, out1], axis=0)
            except Exception as e:
                print(f'Exception found: {e}')

        # return work ID results
        work_ids = storage_df['id'].values
        # split into chunks for further searching
        work_groups = np.array_split(work_ids, 50, axis=0)
        return work_groups

    # new async
    async def send_request(self, workItem, semaphore, session):
        '''
        Leverage semaphore to regulate the amount of requests being made
        '''
        container = []
        async with semaphore:
            url = self.url + f'_apis/wit/workItems/{workItem}/revisions?api-version=6.0'
            async with session.get(url, headers=self.headers) as resp:
                wItem = await resp.read()
                wItem = json.loads(wItem)
                wItem = pd.json_normalize(wItem['value'])
                wItem.columns = wItem.columns.str.replace('fields.', '', regex=False)
                container.append(wItem)
        return container

    # main async call
    async def async_main(self, workItems):
        '''
        main caller
        '''
        s = asyncio.Semaphore(value=75)
        async with aiohttp.ClientSession() as session:
            return await asyncio.gather(*[self.send_request(workItem, semaphore=s, session=session) for workItem in workItems])

    @retry(wait=wait_exponential(multiplier=2, min=15, max=45),
           stop=stop_after_attempt(10))
    def retry_data(self, subset):
        try:
            loop = asyncio.get_event_loop()
            container_ = loop.run_until_complete(self.async_main(workItems=subset))
            out = pd.concat(file[0] for file in container_)
        except Exception as excep:
            print(f"Exception: {excep}")
        return out

    def generate_ado_data_by_batch(self, work_groups):
        total_results = []
        for i, subset in enumerate(work_groups):
            print(f"now on subset: {i}")
            out = self.retry_data(subset)
            total_results.append(out)

        # process and organize data
        raw = pd.concat(file for file in total_results)
        # truncate to just get the cols we want
        raw = raw[self.config.cols]
        return raw
