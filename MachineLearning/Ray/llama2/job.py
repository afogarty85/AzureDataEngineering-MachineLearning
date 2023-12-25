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


# set params
num_devices = 8
size = 13
display_name = f'accelerate_llama2_{size}b_2epoch_linear_instruct_gpt4sample_4_cosine_smtest'

# set inputs
inputs={
    "train_files": Input(
        type="uri_folder",
        path="azureml://datastores/chiemoaddev/paths/SupportFiles/ResolutionsJson",
        mode='download',  # ro_mount  vs download
    ),
    # "llama2_model": Input(
    #     type="uri_folder",
    #     path="azureml://datastores/chiemoaddev/paths/SupportFiles/Llama270B",
    #     mode='ro_mount',  # ro_mount  vs download
    # ),    
}


# set outputs
outputs={
    "output_dir": Output(
        type="uri_folder",
        path=f"azureml://datastores/chiemoaddev/paths/TrainingExport/{display_name}",
    ),
}


# meta-llama/Llama-2-{size}b-chat-hf
accelerate_cmd = (f"""python train.py \
        --mixed_precision "bf16" \
        --batch-size-per-device 32 \
        --eval-batch-size-per-device 24 \
        --num-devices {num_devices} \
        --model_name "meta-llama/Llama-2-{size}b-chat-hf" \
        --ds-config "zero3_llama_2_7b.json" \
        --num-checkpoints-to-keep 1 \
        --use_instruct True \
        --lr_scheduler_type "cosine" \
        --use_flash True \
        --num-epochs 2 \
        --ctx-len 1050 \
        --train_path "/train_newinstruct3.jsonl" \
        --test_path "/test_newinstruct3.jsonl" \
        --lr 2e-5
"""
)



# download settings
env_var = {
"RSLEX_DOWNLOADER_THREADS": 24*num_devices,  # download  ; num_cores * 4
"AZUREML_DATASET_HTTP_RETRY_COUNT": 10,  # download
"DATASET_MOUNT_READ_THREADS": 24*num_devices,  # ro_mount  ; num_cores * 4
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
        "head_node_additional_args": r"""--head --object-store-memory 220000000000""",
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