import azure.functions as func
import logging
import os
import pandas as pd
import numpy as np
import io
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import ClientSecretCredential


# to build; func azure functionapp publish functionappnamehere --build remote --publish-local-settings -i --overwrite-settings -y
# for local build: func start

# reach via cURL: 
# curl -v --header "Content-Type: application/json" --data "{\"file_path\": \"/file/path/filename.xlsx\"}" https://functionappname.azurewebsites.net/api/AppRouteHere?code=apikeyhere

# init app
app = func.FunctionApp()

# # security objects
CLIENTID = os.environ["ID"]
CLIENTSECRET = os.environ["SECRET"]
TENANT = os.environ["TENANT"]

# set other objects
parquet_kwargs = {
     "coerce_timestamps": "us",
     "allow_truncated_timestamps": True,
}


def retrieve_file(file_path):
     """
     Download any file from the lake

     Parameters
     ----------
     file_path : string
          Path to the file, starting at the file system; e.g.,
          /path1/path2/xd.parquet

     Returns
     ----------
     downloaded_bytes : bytes
          A bytes object
     """
     # get access
     credential = ClientSecretCredential(TENANT, CLIENTID, CLIENTSECRET)
     datalake_service_client = DataLakeServiceClient(
          account_url=f"https://accountname.dfs.core.windows.net",
          credential=credential)
     file_system_client = datalake_service_client.get_file_system_client("filesystemname")
     file_client = file_system_client.get_file_client(f"{file_path}")
     # download file
     download = file_client.download_file(max_concurrency=75)
     downloaded_bytes = download.readall()
     # return bytes
     return downloaded_bytes

def write_parquet(df, write_path):
     """
     Write a parquet file to the lake

     Parameters
     ----------
     df : DataFrame

     write_path : string
          Path to the file, starting at the file system; e.g.,
          /TRANSFORMED/filepath1/new_transformed_file.parquet

     Returns
     ----------
     None
     """    
     # get access
     credential = ClientSecretCredential(TENANT, CLIENTID, CLIENTSECRET)
     datalake_service_client = DataLakeServiceClient(
          account_url=f"https://accountname.dfs.core.windows.net",
          credential=credential)
     file_system_client = datalake_service_client.get_file_system_client("filesystemname")
     file_client = file_system_client.get_file_client(f"{write_path}")
     # turn to bytes
     writtenbytes = io.BytesIO()
     df.to_parquet(writtenbytes, **parquet_kwargs)
     # upload file
     file_client.create_file()
     logging.info("Uploading...")
     file_client.upload_data(
          writtenbytes.getvalue(),
          length=len(writtenbytes.getvalue()),
          connection_timeout=3000, overwrite=True)
     logging.info("Upload complete")
     return



@app.function_name(name="HttpTrigger1")
@app.route(route="ses")
def FileExploiter(req: func.HttpRequest) -> func.HttpResponse:
     logging.info('Python HTTP trigger function processed a request.')

     file_path = req.params.get('file_path')
     if not file_path:
          try:
               req_body = req.get_json()
          except ValueError:
               pass
          else:
               file_path = req_body.get('file_path')

     # report state
     logging.info(f'extracting file with path: {file_path}')

     # get bytes
     downloaded_bytes = retrieve_file(file_path)

     # report state
     logging.info(f'Downloaded bytes with len: {len(downloaded_bytes)}')

     # get xlsx as bytes
     df = pd.read_excel(io.BytesIO(downloaded_bytes), skiprows=16, sheet_name='sheet1')

     # report state
     logging.info(f'Opened DF: {df.shape}')

     # upload results
     write_parquet(df=df, write_path='/TRANSFORMED/path1/new_file.parquet')
     logging.info(f'File deployed!')

     if file_path:
          return func.HttpResponse(f"This HTTP triggered function executed successfully.")
     else:
          return func.HttpResponse("This HTTP triggered function executed successfully.", status_code=200)
