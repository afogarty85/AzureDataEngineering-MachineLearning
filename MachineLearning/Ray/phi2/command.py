# set up environment
env = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
    conda_file="./src/conda_env.yml",
    name="my-env"
)
# 

# set params
num_devices = 16
display_name = f'localtest'

# set inputs
inputs={
    "data_files": Input(
        type="uri_folder",
        path="azureml://datastores/chiemoaddev/paths/SupportFiles/FLANSMall/flan2022_submix",
        mode='ro_mount',  # ro_mount  vs download
    ),
} 

# set outputs
outputs={
    "output_dir": Output(
        type="uri_folder",
        path=f"azureml://datastores/chiemoaddev/paths/TrainingExport/Phi2",
        mode='rw_mount',  # ro_mount  vs download
    ),
}


accelerate_cmd = (f"""python my_phi2.py """)



# download settings
env_var = {""
"RSLEX_DOWNLOADER_THREADS": 1000,  # download  ; num_cores * 4
"AZUREML_DATASET_HTTP_RETRY_COUNT": 10,  # download
"DATASET_MOUNT_READ_THREADS": 1000,  # ro_mount  ; num_cores * 4
"DATASET_MOUNT_READ_BLOCK_SIZE": 1024*10,  # MB
"DATASET_MOUNT_BLOCK_FILE_CACHE_MAX_QUEUE_SIZE": 1024*30,
"DATASET_MOUNT_BLOCK_FILE_CACHE_WRITE_THREADS": 768,
"DATASET_MOUNT_BLOCK_FILE_CACHE_ENABLED": True,
"DATASET_MOUNT_READ_BUFFER_BLOCK_COUNT": 1000,
}



# define the command
command_job = command(
    experiment_name='localtest',
    display_name=display_name,  # for mlflow
    code="./src",
    command=accelerate_cmd,
    environment=env,
    instance_count=num_devices,
    distribution={
        "type": "Ray",
        "include_dashboard": True,
        "head_node_additional_args": r"""--head --object-store-memory 200000000000""",
        },   
    environment_variables=env_var,
    compute="A100S",
    shm_size='200g',
    inputs=inputs,
    outputs=outputs,
    identity=ManagedIdentityConfiguration(), # grant storage blob read IAM to cluster SP
)


returned_job = ml_client.jobs.create_or_update(command_job)
returned_job