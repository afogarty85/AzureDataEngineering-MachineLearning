from azure.storage.filedatalake.aio import DataLakeServiceClient
from azure.identity.aio import ClientSecretCredential
import datetime
import asyncio
import nest_asyncio
# find files located in datalake, asynchronously, by path

# params
adls_account_url = 'https://accountname.dfs.core.windows.net'
tenant = 'tenant_vals'
client_id = 'sp_client_id'
client_secret = 'sp_client_secret'
file_system = 'adlslakeFS'
# the base folder to search; will recursively search all sub-folders for all files
path = 'RAW/StorageFolder1'

# execute
'''
async call for looking for file paths, given base loc
will also return the date the file was last modified;
useful for hunting new files
'''
pathing = []
pathing_times = {}
async with ClientSecretCredential(tenant, client_id, client_secret) as credential:
    # get access to ADLS
    async with DataLakeServiceClient(account_url=f"{adls_account_url}", credential=credential) as datalake_service_client:
        # get access to FS
        async with datalake_service_client.get_file_system_client(f"{file_system}") as file_system_client:
            paths = file_system_client.get_paths(path=f"/{path}")
            async for path in paths:
                # append path
                pathing.append(path.name)
                # get last modified
                pathing_times[path.name] = path.last_modified

# filter blobs for last n hours
hours = 24
# create delta
delta_time = datetime.datetime.now() - datetime.timedelta(hours=int(hours))
# filter paths by time
pathing_times = {k:v for k, v in pathing_times.items() if v.replace(tzinfo=None) > lastDayDateTime.replace(tzinfo=None)}
# remove folders if new, too
pathing_times = {k:v for k, v in pathing_times.items() if ('.parquet' in k) or ('.json' in k) or ('.csv.gz' in k)}
