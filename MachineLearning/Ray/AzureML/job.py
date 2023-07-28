# set up environment
env = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
    conda_file="./src/conda_env.yml",
    name="my-env"
)

# set inputs
inputs={
    "RTE": Input(
        type="uri_folder",
        path="azureml://datastores/moaddevlake/paths/Delta/Gold/ML/RandomizedTest",
        mode='download',  # vs download
    ),
}

# set outputs
outputs={
    "output1": Output(
        type="uri_folder",
        path="azureml://datastores/chiemoaddev/paths/TrainingExport/RandomizedTest",
    ),
}

# download settings
env_var = {
"RSLEX_DOWNLOADER_THREADS": 24*4,  # download  ; num_cores * 4
"AZUREML_DATASET_HTTP_RETRY_COUNT": 10,  # download
}

# define the command
command_job = command(
    experiment_name='ray_command',
    code="./src",
    command="python train.py",
    environment=env,
    instance_count=1,  # n clusters
    environment_variables=env_var,
    distribution={
        "type": "Ray",
        "include_dashboard": True,
        },   
    compute="V100Cluster",
    shm_size='32g',
    inputs=inputs,
    outputs=outputs,
    identity=ManagedIdentityConfiguration(), # grant storage blob read IAM to cluster SP
)


returned_job = ml_client.jobs.create_or_update(command_job)
returned_job