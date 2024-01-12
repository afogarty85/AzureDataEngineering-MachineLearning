from azure.ai.ml.entities import AzureDataLakeGen2Datastore, Environment, ManagedIdentityConfiguration
from azure.ai.ml import MLClient, command, Input, Output
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
import json
from azure.ai.ml.entities import (
    VsCodeJobService,
    TensorBoardJobService,
    JupyterLabJobService,
    SshJobService,
)
import mlflow


# Enter details of your AML workspace
subscription_id = "dc3f939d-f1cb-48a0-ab87-ae7dca8e3e8e"
resource_group = "rg-azureml"
workspace = "IcM_GPU"

# get a handle to the workspace
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)


# access datalake container
store = AzureDataLakeGen2Datastore(
    name="moaddevlake",
    description="moaddevlake",
    account_name="moaddevlake",
    filesystem="moaddevlakefs"
)

ml_client.create_or_update(store)


# access datalake container
store = AzureDataLakeGen2Datastore(
    name="chiemoaddev",
    description="chiemoaddev",
    account_name="moaddevlake",
    filesystem="chiemoaddevfs"
)

ml_client.create_or_update(store)


# set mlflow
mlflow.set_experiment('accelerate_llama2')


# set up environment
env = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
    conda_file="./src/conda_env.yml",
    name="my-env"
)


# set params
display_name = f'accelerate_llama2_inference_7b_64bz_noinstruct'

# set inputs
inputs={
    "fine_tuned_model": Input(
        type="uri_folder",
        path="azureml://datastores/chiemoaddev/paths/TrainingExport/accelerate_llama2_7b_2epoch_linear_noinstruct/TorchTrainer_2023-10-06_12-17-39/TorchTrainer_57cc5_00000_0_2023-10-06_12-17-39/checkpoint_000001",
        mode='download',  # ro_mount  vs download
    ),
    "train_files": Input(
        type="uri_folder",
        path="azureml://datastores/chiemoaddev/paths/SupportFiles/ResolutionsJson",
        mode='download',  # ro_mount  vs download
    ),
}


# set outputs
outputs={
    "output_dir": Output(
        type="uri_folder",
        path=f"azureml://datastores/chiemoaddev/paths/TrainingExport/{display_name}",
    ),
}

num_devices = 8

accelerate_cmd = (f"""python eval.py  \
        --num_devices {num_devices} \
        --ctx-len 1186 \
        --batch-size 12 \
        --use_instruct True \
        --test_path test_newinstruct_oos.jsonl \
        --experiment-name {display_name}
        """)

# download settings
env_var = {
"RSLEX_DOWNLOADER_THREADS": 48,  # download  ; num_cores * 4
"AZUREML_DATASET_HTTP_RETRY_COUNT": 10,  # download
"DATASET_MOUNT_READ_THREADS": 48,  # ro_mount  ; num_cores * 4
}

# define the command
command_job = command(
    experiment_name='llama2',
    display_name=display_name,  # for mlflow
    code="./src",
    command=accelerate_cmd,
    environment=env,
    instance_count=num_devices,
    distribution={
        "type": "Ray",
        "include_dashboard": True,
        "head_node_additional_args": r"""--head --object-store-memory 210000000000""",
        },
    environment_variables=env_var,
    compute="A100S",
    shm_size='220g',
    inputs=inputs,
    outputs=outputs,
    identity=ManagedIdentityConfiguration(), # grant storage blob read IAM to cluster SP
)


returned_job = ml_client.jobs.create_or_update(command_job)
returned_job