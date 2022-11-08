import os
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
from loguru import logger


# create storage folder; we MUST name it as outputs;
os.makedirs('outputs/subfolder', exist_ok=True)

# upload results using service principal access to AzureML
svc_pr = ServicePrincipalAuthentication(tenant_id=TENANT,
                                service_principal_id=SP_CLIENT_ID,
                                service_principal_password=SP_CLIENT_SECRET)

# log into our workspace
ws = Workspace(subscription_id=SUBSCRIPTION_ID,
       resource_group=RESOURCE_GROUP,
       auth=svc_pr,
       workspace_name=WORKSPACE_NAME)

logger.info("Starting to upload to moaddevlakefs")
# upload data to lake
Dataset.File.upload_directory(src_dir='outputs/subfolder',
                              target=DataPath(ws.datastores['lakefilesystem'],
                                              'RAW/SourceReplication/subfolder'),
                              show_progress=True)
