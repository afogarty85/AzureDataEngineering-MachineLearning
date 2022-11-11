
from azure.storage.blob.aio import BlobServiceClient, BlobClient, download_blob_from_url, ContainerClient
from azure.identity.aio import ClientSecretCredential
import os
import io
from zipfile import ZipFile
import pandas as pd
import json
from loguru import logger
import asyncio
import nest_asyncio
nest_asyncio.apply()  # important!

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
                #print(f.namelist());
                with f.open([s for s in f.namelist() if 'srv_info.csv' in s][0], 'r') as g:
                    df = pd.read_csv(g)


# download file given URI v2
async with ClientSecretCredential(TENANT, CLIENTID, CLIENTSECRET) as credential:
    async with BlobServiceClient(account_url=f"{blob_endpoint}", credential=credential) as blob_service_client:
        async with blob_service_client.get_container_client(f'{container}') as blob_container_client:
            # open bytes
            writtenbytes = io.BytesIO()
            # write file to it
            await download_blob_from_url(blob_URI, credential=credential, output=writtenbytes)
            # zipfile
            f = ZipFile(writtenbytes)
            # for the file of interest
            with f.open([s for s in f.namelist() if 'filename.log' in s][0], 'r') as g:
                _log = g.read().decode("utf-8")
                trace_out = do_stuff(_log)

            with f.open([s for s in f.namelist() if 'filename2.log' in s][0], 'r') as g:
                _log2 = g.read().decode("utf-8")
                trace_out2 = do_stuff2(_log2)




# async process zip files
class Zip_File_Processor():
    '''
    Reformed processor to handle within-zip file extraction and processing
    '''
    def __init__(self, args):
        self.args = args

    async def async_zip_file_extractor(self, blobFileName, blobEndPoint, semaphore):
        # storage container
        storage_ = pd.DataFrame()

        try:
            # access blob
            async with ClientSecretCredential(self.args.TENANT, self.args.CLIENTID, self.args.CLIENTSECRET) as credential:
                async with BlobServiceClient(account_url="https://mdaasincomingprod.blob.core.windows.net/", credential=credential) as blob_service_client:
                    async with blob_service_client.get_container_client(blobEndPoint) as blob_container_client:
                        async with semaphore:
                            logger.info(f'Starting: {blobFileName}, {blobEndPoint}')
                            # open bytes
                            writtenbytes = io.BytesIO()
                            # write file to it
                            await download_blob_from_url(f'https://url.blob.core.windows.net/{blobEndPoint}/{blobFileName}', credential=credential, output=writtenbytes)
                            # zipfile
                            f = ZipFile(writtenbytes)

                            with f.open([s for s in f.namelist() if 'CsiDiag/CSI_DER.json' in s][0], 'r') as g:
                                # decode binary
                                _log = json.load(g)
                                # process
                                outfile = await self.process_csi_der_json(_log)
                                # add fileName
                                outfile['blobFileName'] = blobFileName
                                # store
                                storage_ = pd.concat([storage_, outfile], axis=0)

                            if semaphore.locked():
                                await asyncio.sleep(1)

                            logger.info(f'Completed: {blobFileName}')

        except Exception as e:
            logger.error(f'Exception: {e}')

        return storage_


    async def async_file_as_bytes_generator(self, blobFileName, blobEndPoint, semaphore):
        '''
        main caller
        '''
        semaphore = asyncio.Semaphore(value=semaphore)
        return await asyncio.gather(*[self.async_zip_file_extractor(fn, ep, semaphore) for fn, ep in zip(blobFileName, blobEndPoint)])


    async def process_csi_der_json(self, log_):
        '''
        Do Stuff
        '''
        # recieve data
        pass

        return log_


# sample usage:
zip_proc = Zip_File_Processor(args=args)
loop = asyncio.get_event_loop()
log_ = loop.run_until_complete(zip_proc.async_file_as_bytes_generator(blobFileName=blobFileNameList, blobEndPoint=blobEndPointList, semaphore=25))
