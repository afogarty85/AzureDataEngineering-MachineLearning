
from azure.storage.blob.aio import BlobServiceClient, BlobClient, download_blob_from_url, ContainerClient
from azure.identity.aio import ClientSecretCredential
import os
import io
from zipfile import ZipFile
import pandas as pd

# service principal data
TENANT = ''
CLIENTID = ''
CLIENTSECRET = ''

# blob details
blob_endpoint = 'https://xd.blob.core.windows.net/'
container = 'xd'


# azure blob async: find blobs
last_modified = {}
pathing = []
async with ClientSecretCredential(TENANT, CLIENTID, CLIENTSECRET) as credential:
    async with BlobServiceClient(account_url=f"{blob_endpoint}", credential=credential) as blob_service_client:
        async with blob_service_client.get_container_client(f'{container}') as blob_container_client:
            blobs = blob_container_client.list_blobs()
            async for blob in blobs:
                # for each blob, get its name
                pathing.append(blob.name)
                # get last modified
                last_modified[blob.name] = blob.last_modified


# download file given URI
async with ClientSecretCredential(TENANT, CLIENTID, CLIENTSECRET) as credential:
    async with BlobServiceClient(account_url=f"{blob_endpoint}", credential=credential) as blob_service_client:
        async with blob_service_client.get_container_client(f'{container}') as blob_container_client:
            await download_blob_from_url('https://xd.blob.core.windows.net/xd/P229794620001012_2021-07-06T14.11.16_START.zip', credential=credential, output=blob.name)


# download found blob to disk
async with ClientSecretCredential(TENANT, CLIENTID, CLIENTSECRET) as credential:
    async with BlobServiceClient(account_url=f"{blob_endpoint}", credential=credential) as blob_service_client:
        async with blob_service_client.get_container_client(f'{container}') as blob_container_client:
            blobs = blob_container_client.list_blobs()
            async for blob in blobs:
                # for each blob, create a file with its same name
                download_file_path = os.path.join(rf'C:\Users\Andrew\Desktop\temp1\{blob.name}')
                # open the file
                with open(file=download_file_path, mode="wb") as download_file:
                    # get the data
                    blob_data = await blob_container_client.download_blob(blob)
                    # write into it
                    await blob_data.readinto(download_file)


# download found zip-file blob to bytes; extract file of interest in memory
async with ClientSecretCredential(TENANT, CLIENTID, CLIENTSECRET) as credential:
    async with BlobServiceClient(account_url=f"{blob_endpoint}", credential=credential) as blob_service_client:
        async with blob_service_client.get_container_client(f'{container}') as blob_container_client:
            blobs = blob_container_client.list_blobs()
            async for blob in blobs:
                # identify blob
                blob_data = await blob_container_client.download_blob(blob)
                # open bytes
                writtenbytes = io.BytesIO()
                # read into bytes
                await blob_data.readinto(writtenbytes)
                # zipfile
                f = ZipFile(writtenbytes)
                #print(f.namelist()); 'srv_info.csv'
                with f.open([s for s in f.namelist() if 'srv_info.csv' in s][0], 'r') as g:
                    df = pd.read_csv(g)
