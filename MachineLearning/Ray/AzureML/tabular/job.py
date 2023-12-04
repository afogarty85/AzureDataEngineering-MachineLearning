from azure.ai.ml.entities import AzureDataLakeGen2Datastore, Environment, ManagedIdentityConfiguration
from azure.ai.ml import MLClient, command, Input, Output, RayDistribution, MpiDistribution
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
import json
from azure.ai.ml.entities import (
    VsCodeJobService,
    TensorBoardJobService,
    JupyterLabJobService,
    SshJobService,
)

# Enter details of your AML workspace
subscription_id = "x"
resource_group = "x"
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
# set up environment
env = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
    conda_file="./src/conda_env.yml",
    name="my-env"
)

# set inputs
inputs={
    "train_files": Input(
        type="uri_folder",
        path="azureml://datastores/moaddevlake/paths/Delta/Gold/ML/RandomizedTest",
        mode='download',  # vs download
    ), 
}

# set outputs
outputs={
    "output_dir": Output(
        type="uri_folder",
        mode="rw_mount",
        path="azureml://datastores/chiemoaddev/paths/TrainingExport/RandomizedTest",
    ),
}

# download settings
env_var = {
"RSLEX_DOWNLOADER_THREADS": 40*4,  # download  ; num_cores * 4
"AZUREML_DATASET_HTTP_RETRY_COUNT": 10,  # download
}

# define the command
command_job = command(
    experiment_name='ray_tune_syndrome',
    code="./src",
    command="""python tune_syndrome.py \
            --mx "fp16" \
            --batch_size_per_device 32768 \
            --eval_batch_size_per_device 32768 \
            --num_devices 1 \
            --num_workers 16 \
            --num_epochs 1 \
            --lr 1e-4 \
            --seed 1 \
            --weight_decay 0.01
            """,
    environment=env,
    instance_count=4,  # n clusters
    environment_variables=env_var,
    distribution={
        "type": "Ray",
        "include_dashboard": True,
        "head_node_additional_args": r"""--object-store-memory 300000000000""",
        "worker_node_additional_args": r"""--object-store-memory 300000000000""",
        },   
    compute="lowpriNVLINK",  # lowpriNVLINK // lowpriV100
    shm_size='660g',
    inputs=inputs,
    outputs=outputs,
    identity=ManagedIdentityConfiguration(), # grant storage blob read IAM to cluster SP
)


returned_job = ml_client.jobs.create_or_update(command_job)
returned_job



# other

# define the command
command_job = command(
    experiment_name='ray_train_syndrome',
    code="./src",
    command="""python train_syndrome.py \
            --mx "fp16" \
            --batch_size_per_device 40960 \
            --eval_batch_size_per_device 40960 \
            --num_devices 1 \
            --num_workers 32 \
            --num_epochs 3 \
            --lr 0.0008 \
            --seed 1 \
            --weight_decay 0.01
            """,
    environment=env,
    instance_count=4,  # n clusters
    environment_variables=env_var,
    distribution={
        "type": "Ray",
        "include_dashboard": True,
        "head_node_additional_args": r"""--object-store-memory 300000000000""",
        "worker_node_additional_args": r"""--object-store-memory 300000000000""",
        },   
    compute="lowpriNVLINK",  # lowpriNVLINK // lowpriV100
    shm_size='660g',
    inputs=inputs,
    outputs=outputs,
    identity=ManagedIdentityConfiguration(), # grant storage blob read IAM to cluster SP
)


returned_job = ml_client.jobs.create_or_update(command_job)
returned_job