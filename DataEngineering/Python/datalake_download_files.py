from azure.storage.filedatalake.aio import DataLakeServiceClient
from azure.identity.aio import ClientSecretCredential
import asyncio
import nest_asyncio
from loguru import logger
import pandas as pd
from io import BytesIO
import aiohttp
nest_asyncio.apply()


# params
adls_account_url = 'https://accountname.dfs.core.windows.net'
tenant = 'tenant_vals'
client_id = 'sp_client_id'
client_secret = 'sp_client_secret'
file_system = 'adlslakeFS'
# The file of interest
file_path = 'RAW/StorageFolder1/my_file.parquet'


# download many files, as an example; all parquet files in a folder
# and turn them into a pandas dataframe

async def concat_parquet_adls_files(self, file_path, columns):
    '''
    file_path -- the root directory containing many parquet

    '''
    # create credentials
    # get access to AAD
    async with ClientSecretCredential(tenant, client_id, client_secret) as credential:
        # get access to ADLS
        async with DataLakeServiceClient(account_url=f"https://lakelocation.dfs.core.windows.net", credential=credential) as datalake_service_client:
            # get access to FS
            async with datalake_service_client.get_file_system_client(f"lakefs") as file_system_client:
                paths = file_system_client.get_paths(f"/{file_path}")
                byte_storage = []
                path_storage = []
                async for path in paths:
                    path_storage.append(path.name)
                path_storage = [x for x in path_storage if '.parquet' in x]
                for path in path_storage:
                    file_client = file_system_client.get_file_client(f'{path}')
                    # download file, concurrency is parallel connections
                    logger.info(f'Downloading {path}')
                    download = await file_client.download_file(max_concurrency=75)
                    downloaded_bytes = await download.readall()
                    byte_storage.append(downloaded_bytes)
            # concat data
            logger.info('Concatenating parquet files')
            full_df = pd.concat(pd.read_parquet(BytesIO(parquet_file), columns=columns)
                for parquet_file in byte_storage)

            # remove storage
            del byte_storage
            return full_df

def execute_concat_parquet_adls_files(self, file_path, columns):
    '''
    Executes retrieve_concat_parquet_adls_files
    '''
    # use async
    loop = asyncio.get_event_loop()
    df = loop.run_until_complete(self.concat_parquet_adls_files(file_path, columns))
    return df


# download large numbers of files asynchronously
async def find_files_by_path(path):
    ''' async call for looking for file paths, given RAW base '''
    pathing = []
    async with ClientSecretCredential(tenant, client_id, client_secret) as credential:
        # get access to ADLS
        async with DataLakeServiceClient(account_url=f"https://lakelocation.dfs.core.windows.net", credential=credential) as datalake_service_client:
            # get access to FS
            async with datalake_service_client.get_file_system_client(f"lakefs") as file_system_client:
                paths = file_system_client.get_paths(path=f"/{path}")
                async for path in paths:
                    pathing.append(path.name)
    return pathing


async def retrieve_adls_file_as_bytes(file_path, semaphore):
    '''
    Connects to ADLSGen2 and retrieves any file of choice as bytes
    '''
    # get access to AAD
    bytes_container = []
    async with ClientSecretCredential(tenant, client_id, client_secret) as credential:
        # get access to ADLS
        async with DataLakeServiceClient(account_url=f"https://lakelocation.dfs.core.windows.net", credential=credential) as datalake_service_client:
            # get access to FS
            async with datalake_service_client.get_file_system_client(f"lakefs") as file_system_client:
                async with semaphore:
                    file_client = file_system_client.get_file_client(f"/{file_path}")
                    logger.info(f"Generating data ADLS from file path: {file_path}")
                    # pull and download file
                    download = await file_client.download_file(max_concurrency=75)
                    downloaded_bytes = await download.readall()
                    # report progress
                    logger.info(f"Finished downloading bytes of length: {len(downloaded_bytes)}")
                    bytes_container.append(downloaded_bytes)
                    if semaphore.locked():
                        await asyncio.sleep(15)
    return bytes_container


async def async_doc_main(paths, semaphore):
    '''
    main caller
    '''
    s = asyncio.Semaphore(value=semaphore)
    return await asyncio.gather(*[retrieve_adls_file_as_bytes(file_path=path, semaphore=s) for path in paths])

# get file paths in this folder in our lake
loop = asyncio.get_event_loop()
file_paths = loop.run_until_complete(find_files_by_path(path='RAW/Folder/SubFolder'))

# for each file found, retrieve the file as bytes
loop = asyncio.get_event_loop()
bytes_container = loop.run_until_complete(async_doc_main(paths=file_paths, semaphore=25))
