import asyncio
from azure.storage.blob.aio import (   BlobServiceClient,
    BlobClient,
    download_blob_from_url,
    ContainerClient,
)

from zipfile import ZipFile



async def zip_reader(blobFileName, blobEndPoint, semaphore):
    # access blob
    async with ClientSecretCredential(TENANT, CLIENTID, CLIENTSECRET) as credential:
        async with BlobServiceClient(account_url="https://blobendpoint.blob.core.windows.net/", credential=credential, max_single_get_size=64 * 1024 * 1024, max_chunk_get_size=32 * 1024 * 1024) as blob_service_client:
            async with blob_service_client.get_blob_client(container=blobEndPoint, blob=blobFileName) as blob_client:
                async with semaphore:
                    logger.info(f"Starting: {blobFileName}, {blobEndPoint}")

                    # open bytes
                    writtenbytes = io.BytesIO()

                    # write file to it
                    stream = await blob_client.download_blob(max_concurrency=25)
                    stream = await stream.readinto(writtenbytes)

                    # zipfile
                    f = ZipFile(writtenbytes)

                    # file list
                    file_list = [s for s in f.namelist()]

                    # send to df
                    t_df = pd.DataFrame({'fileList': file_list})

                    # add fileName
                    t_df['blobFileName'] = blobFileName
                    t_df['blobEndPoint'] = blobEndPoint

                    if semaphore.locked():
                        await asyncio.sleep(1)

                    logger.info(f"Completed: {blobFileName}")

                    return t_df


async def async_file_as_bytes_generator(blobFileName, blobEndPoint, task_limit):
    """
    main caller
    """
    semaphore = asyncio.Semaphore(value=task_limit)
    
    all_tasks = {zip_reader(fn, ep, semaphore) for fn, ep in zip(blobFileName, blobEndPoint)}
    current_tasks = set()
    while all_tasks or current_tasks:
        while all_tasks and len(current_tasks) < task_limit:
            current_tasks.add(all_tasks.pop())
            
        done, incomplete = await asyncio.wait(current_tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                yield task.result()
            except Exception as e:
                print(e)
        current_tasks = incomplete

# get results
async for dataframe in async_file_as_bytes_generator(blobFileNameSet, blobEndPointSet, 4):
    print(dataframe.shape)