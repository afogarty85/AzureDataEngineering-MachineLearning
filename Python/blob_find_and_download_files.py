
from azure.storage.blob.aio import BlobServiceClient, BlobClient, download_blob_from_url, ContainerClient
last_modified = {}
pathing = []


blob_endpoint = 'https://xd.blob.core.windows.net/'
container = 'xd'
# azure blob async: find blobs
async with ClientSecretCredential(TENANT, CLIENTID, CLIENTSECRET) as credential:
    async with BlobServiceClient(account_url=f"{blob_endpoint}", credential=credential) as blob_service_client:
        async with blob_service_client.get_container_client(f'{container}') as blob_container_client:
            blobs = blob_container_client.list_blobs()
            async for blob in blobs:
                pathing.append(blob.name)
                # get last modified
                last_modified[blob.name] = blob.last_modified

            await blob_container_client.download_blob(blob)

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
                download_file_path = os.path.join(rf'C:\Users\Andrew\Desktop\temp1\{blob.name}')
                with open(file=download_file_path, mode="wb") as download_file:
                    blob_data = await blob_container_client.download_blob(blob)
                    await blob_data.readinto(download_file)
