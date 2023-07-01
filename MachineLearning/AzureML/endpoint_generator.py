# import required libraries
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
    Workspace,
    OnlineRequestSettings
)
from azure.identity import DefaultAzureCredential
import json
import pandas as pd


# enter details of your Azure Machine Learning workspace
subscription_id = ""
resource_group = ""
workspace = ""


# get a handle to the workspace
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

# Define an endpoint name
endpoint_name = ""
# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name = endpoint_name, 
    description="",
    auth_mode="aml_token"
)

# set traffic to 100%
endpoint.traffic = {"blue": 100}

# update
ml_client.online_endpoints.begin_create_or_update(endpoint).result()


# location of trained model
model = Model(path="../train_data/xgboost_trained.json")

# set env
env = Environment(
    conda_file="../train_data/conda.yml",
    image="mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest",
)

# init deployment
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    environment=env,
    code_configuration=CodeConfiguration(
        code="../train_data/", scoring_script="api.py"
    ),
    instance_type="Standard_DS3_v2",
    instance_count=1,
    request_settings=OnlineRequestSettings(request_timeout_ms=90000)
)

# update
ml_client.online_deployments.begin_create_or_update(blue_deployment).result()

# sample dict
sample = {
	"repo_name": "",
	"commit_id": "",
	"pipeline_name": ""
}

# to json
with open("../train_data/sample_json.json", 'w') as fp:
    json.dump(sample, fp)

# test the blue deployment with some sample data
out = ml_client.online_endpoints.invoke(
    endpoint_name=endpoint_name,
    deployment_name="blue",
    request_file="../train_data/sample_json.json",
)


# load result
df = pd.read_json(json.loads(out))
df = pd.json_normalize(df['data'])
df.head()