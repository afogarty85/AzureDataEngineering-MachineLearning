from azure.ai.ml.entities import AzureDataLakeGen2Datastore, Environment, ManagedIdentityConfiguration
from azure.ai.ml import MLClient, command, Input, Output
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azureml.fsspec import AzureMachineLearningFileSystem
import azureml.dataprep

# set client
subscription_id = "xxxxxxxx-f1cb-48a0-ab87-xxxxxxxxxxx"
resource_group = "rg-azureml"
workspace = "xxx_xxx"

# get a handle to the workspace
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

# init data store
store = AzureDataLakeGen2Datastore(
    name="chiemoaddev",
    description="chiemoaddev",
    account_name="moaddevlake",
    filesystem="chiemoaddevfs"
)

ml_client.create_or_update(store)

# build uri
uri = f'azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace}/datastores/{datastore_name}/paths'

# init connection
fs = AzureMachineLearningFileSystem(uri)

# download files from datastore
fs.download(rpath='TrainingExport/GDCOFill/flan_t5_small_gpu/epoch_20', lpath='epochs/epoch_20', recursive=True,  **{'overwrite': 'MERGE_WITH_OVERWRITE'})