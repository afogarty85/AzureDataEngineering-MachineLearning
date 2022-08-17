from azure.storage.filedatalake.aio import DataLakeServiceClient
from azure.identity.aio import ClientSecretCredential
import datetime
import asyncio
import nest_asyncio


# params
adls_account_url = 'https://accountname.dfs.core.windows.net'
tenant = 'tenant_vals'
client_id = 'sp_client_id'
client_secret = 'sp_client_secret'
file_system = 'adlslakeFS'
# The file of interest
file_path = 'RAW/StorageFolder1/my_file.parquet'

'''
Connects to ADLSGen2 and retrieves any file of choice as bytes
'''
# get access to AAD
async with ClientSecretCredential(tenant, client_id, client_secret) as credential:
    # get access to ADLS
    async with DataLakeServiceClient(account_url=f"{adls_account_url}", credential=credential) as datalake_service_client:
        # get access to FS
        async with datalake_service_client.get_file_system_client(f"{file_system}") as file_system_client:
            file_client = file_system_client.get_file_client(f"/{file_path}")
            self.logger.info(f"Generating data ADLS from file path: {file_path}")
            # pull and download file asynchronously
            download = await file_client.download_file(max_concurrency=75)
            downloaded_bytes = await download.readall()
            # report progress
            self.logger.info(f"Finished downloading bytes of length: {len(downloaded_bytes)}")

# download many files, as an example; all parquet files in a folder
# and turn them into a pandas dataframe

async def concat_parquet_adls_files(self, file_path, columns):
    '''
    file_path -- the root directory containing many parquet

    '''
    # create credentials
    # get access to AAD
    async with ClientSecretCredential(self.args.TENANT, self.args.CLIENTID, self.args.CLIENTSECRET) as credential:
        # get access to ADLS
        async with DataLakeServiceClient(account_url=f"https://moad{self.args.step}lake.dfs.core.windows.net", credential=credential) as datalake_service_client:
            # get access to FS
            async with datalake_service_client.get_file_system_client(f"moad{self.args.step}lakefs") as file_system_client:
                paths = file_system_client.get_paths(f"/{file_path}")
                byte_storage = []
                path_storage = []
                async for path in paths:
                    path_storage.append(path.name)
                path_storage = [x for x in path_storage if '.parquet' in x]
                for path in path_storage:
                    file_client = file_system_client.get_file_client(f'{path}')
                    # download file, concurrency is parallel connections
                    self.logger.info(f'Downloading {path}')
                    download = await file_client.download_file(max_concurrency=75)
                    downloaded_bytes = await download.readall()
                    byte_storage.append(downloaded_bytes)
            # concat data
            self.logger.info('Concatenating parquet files')
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
