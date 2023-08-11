# import required libraries
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AzureDataLakeGen2Datastore, Environment, ManagedIdentityConfiguration
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import MLClient, command, Input, Output, dsl, load_component

# set up environment
env = Environment(
    image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
    conda_file="./inference/conda_env.yaml",
    name="my-env"
)

# set inputs
inputs={
    "FLAN_T5_FT": Input(
        type="uri_folder",
        path="azureml://datastores/chiemoaddev/paths/TrainingExport/GDCOFill/flan_t5_small_augment/epoch_20",
        mode='download',  # vs download
    ),
}

# set outputs
outputs={
    "out": Output(
        type="uri_folder",
        path="azureml://datastores/moaddevlake/paths/RAW/History/ML/GDCOFill",
    ),
}


try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()


# Get a handle to workspace
ml_client = MLClient.from_config(credential=credential)


# download settings
env_var = {
"RSLEX_DOWNLOADER_THREADS": 48,  # download  ; num_cores * 4
"AZUREML_DATASET_HTTP_RETRY_COUNT": 10,  # download
}

# define the command
inference_command = command(
    experiment_name='inference gdcofill',
    code="./inference",
    command="python inference.py",
    environment=env,
    instance_count=1,
    environment_variables=env_var,
    compute="A100S",
    shm_size='32g',
    inputs=inputs,
    outputs=outputs,
    identity=ManagedIdentityConfiguration(), # grant storage blob contrib IAM to cluster SP
)


# the dsl decorator tells the sdk that we are defining an Azure ML pipeline
@dsl.pipeline(description="GDCO INF", default_compute='A100S',)
def gdco_inference_pipeline():
    inference_job = inference_command()
    return {}

# gen pipeline
pipeline_to_submit = gdco_inference_pipeline()

# submit
pipeline_job = ml_client.jobs.create_or_update(pipeline_to_submit,)
pipeline_job