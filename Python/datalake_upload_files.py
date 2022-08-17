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
# the base folder to search; will recursively search all sub-folders for all files
upload_path = 'RAW/StorageFolder1'
file_name = 'test_file.json'


# example json
json_out = {
    "childItems": [
        {
            "name": "PartItem/2022-03-31_PartItem.parquet",
            "subpath": "/PartItem",
            "filename": "2022-03-31_PartItem.parquet",
            "subfolder": "PartItem",
            "type": "File"
        },
        {
            "name": "2022-03-31_PlatformPlanApproval.parquet",
            "subpath": "/PlatformPlanApproval",
            "filename": "2022-03-31_PlatformPlanApproval.parquet",
            "subfolder": "PlatformPlanApproval",
            "type": "File"
        },
        {
            "name": "2022-03-31_ResourceDesign.parquet",
            "subpath": "/ResourceDesign",
            "filename": "2022-03-31_ResourceDesign.parquet",
            "subfolder": "ResourceDesign",
            "type": "File"
        }
    ]
}

'''
upload files to ADLS storage storage
'''
async with ClientSecretCredential(tenant, client_id, client_secret) as credential:
    # get access to ADLS
    async with DataLakeServiceClient(account_url=f"{adls_account_url}", credential=credential) as datalake_service_client:
        # get access to FS
        async with datalake_service_client.get_file_system_client(f"{file_system}") as file_system_client:
            file_client = file_system_client.get_file_client(f"/{upload_path}/{file_name}")
            await file_client.create_file()
            self.logger.info('Uploading...')
            await file_client.append_data(data=json_out, offset=0)
            self.logger.info('Upload complete')
            await file_client.flush_data(len(json_out))


# write dataframe as parquet to ADLSGEN2 datalake
async def parquet_upload(self, df, file_path):
    """
    Write Parquet File to ADLS

    Args:
        file_system: String denoting the filesystem for the ADLSGEN2
        file_path: The file path, given the file_system to reach the file
        concurrency: Number of parallel connections to download the file
    """
    parquet_kwargs = {
        'coerce_timestamps': 'us',
        'allow_truncated_timestamps': True,
    }

    # get access to AAD
    async with ClientSecretCredential(self.args.TENANT, self.args.CLIENTID, self.args.CLIENTSECRET) as credential:
        # get access to ADLS
        async with DataLakeServiceClient(account_url=f"https://moad{self.args.step}lake.dfs.core.windows.net", credential=credential) as datalake_service_client:
            # get access to FS
            async with datalake_service_client.get_file_system_client(f"moad{self.args.step}lakefs") as file_system_client:
                # create file client with a path to the exact file
                file_client = file_system_client.get_file_client(f'{file_path}')
                # turn to bytes
                writtenbytes = io.BytesIO()
                df.to_parquet(writtenbytes, **parquet_kwargs)
                # upload file
                await file_client.create_file()
                self.logger.info('Uploading...')
                await file_client.append_data(data=writtenbytes.getvalue(), offset=0)
                self.logger.info('Upload complete')
                await file_client.flush_data(writtenbytes.tell())


def execute_parquet_upload(self, df, file_path):
    '''
    Executes ADLSGen2 parquet file upload operatins
    '''
    loop = asyncio.get_event_loop()
    loop.run_until_complete(self.parquet_upload(df,
                                                file_path))
