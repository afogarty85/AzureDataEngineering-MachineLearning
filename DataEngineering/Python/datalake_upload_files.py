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



from azure.storage.filedatalake.aio import DataLakeServiceClient as DataLakeServiceClientS
from azure.identity.aio import ClientSecretCredential  as ClientSecretCredentialS
# write dataframe as parquet to ADLSGEN2 datalake
def parquet_upload(df, file_path):
    """
    Upload parquet files to ADLS

    Args:
        df: Dataframe
        file_path: The file path, given the file_system to reach the file
    """
    parquet_kwargs = {
        "coerce_timestamps": "us",
        "allow_truncated_timestamps": True,
    }

    # get access to AAD -- not using AIO here
    credential = ClientSecretCredentialS(args.TENANT, args.CLIENTID, args.CLIENTSECRET)
    # get into lake
    datalake_service_client = DataLakeServiceClientS(account_url=f"https://mylake.dfs.core.windows.net", credential=credential)
    # get into fs container
    file_system_client = datalake_service_client.get_file_system_client(f"mylakefs")
    # create file client with a path to the exact file
    file_client = file_system_client.get_file_client(f"{file_path}")
    # turn to bytes
    writtenbytes = io.BytesIO()
    df.to_parquet(writtenbytes, **parquet_kwargs)
    # upload file
    file_client.create_file()
    self.logger.info("Uploading...")
    file_client.upload_data(writtenbytes.getvalue(), length=len(writtenbytes.getvalue()), timeout=600, overwrite=True)
    self.logger.info("Upload complete")
