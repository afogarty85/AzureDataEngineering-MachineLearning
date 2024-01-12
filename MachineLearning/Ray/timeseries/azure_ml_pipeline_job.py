# set up environment
env = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
    conda_file="./src/conda_env.yml",
    name="my-forecast-env"
)

# download settings
env_var = {
"RSLEX_DOWNLOADER_THREADS": 24*4,  # download  ; num_cores * 4
"AZUREML_DATASET_HTTP_RETRY_COUNT": 10,  # download
"DATASET_MOUNT_READ_THREADS": 24*4,  # ro_mount  ; num_cores * 4
}


# set outputs
stats_tune_outputs={
    "output_dir": Output(
        type="uri_folder",
        path=f"azureml://datastores/chiemoaddev/paths/TrainingExport/LicenseForecast/LicenseForecast_Stats_Tuner",
    ),
}


# define the command
stats_tune_job = command(
    experiment_name='license_forecast_pipeline',
    display_name="LicenseForecast_Stats_Tuner",  # for mlflow
    code="./src",
    command="python stats_forecast_tune.py",
    environment=env,
    instance_count=50,
    distribution={
        "type": "Ray",
        "include_dashboard": True,
        },   
    environment_variables=env_var,
    compute="cpu-cluster",
    shm_size='20g',
    inputs=None,
    outputs=stats_tune_outputs,
    identity=ManagedIdentityConfiguration(), # grant storage blob read IAM to cluster SP
)




# set inputs
stats_pred_inputs={
    "tune_files": Input(
        type="uri_folder",
        path="azureml://datastores/chiemoaddev/paths/TrainingExport/LicenseForecast/LicenseForecast_Stats_Tuner",
        mode='download',  # ro_mount  vs download
    ),    
}

# set outputs
stats_pred_outputs={
    "output_dir": Output(
        type="uri_folder",
        path=f"azureml://datastores/chiemoaddev/paths/TrainingExport/LicenseForecast/LicenseForecast_Stats_Predict",
        mode="rw_mount",
    ),
}



# define the command
stats_pred_job = command(
    experiment_name='license_forecast_pipeline',
    display_name="LicenseForecast_Stats_Predict",  # for mlflow
    code="./src",
    command="python stats_forecast_predict.py",
    environment=env,
    instance_count=50,
    distribution={
        "type": "Ray",
        "include_dashboard": True,
        },   
    environment_variables=env_var,
    compute="cpu-cluster",
    shm_size='20g',
    inputs=stats_pred_inputs,
    outputs=stats_pred_outputs,
    identity=ManagedIdentityConfiguration(), # grant storage blob read IAM to cluster SP
)


## NN

# set outputs
nn_tune_outputs={
    "output_dir": Output(
        type="uri_folder",
        path=f"azureml://datastores/chiemoaddev/paths/TrainingExport/LicenseForecast/LicenseForecast_NN_Tuner",
    ),
}

# define the command
nn_tune_job = command(
    experiment_name='license_forecast_tuning',
    display_name="LicenseForecast_NN_Tuner",  # for mlflow
    code="./src",
    command="python neural_forecast_tune.py",
    environment=env,
    instance_count=3,
    distribution={
        "type": "Ray",
        "include_dashboard": True,
        },   
    environment_variables=env_var,
    compute="lowpriNVLINK",
    shm_size='600g',
    inputs=None,
    outputs=nn_tune_outputs,
    identity=ManagedIdentityConfiguration(), # grant storage blob read IAM to cluster SP
)


# set inputs
nn_pred_inputs={
    "tune_files": Input(
        type="uri_folder",
        path="azureml://datastores/chiemoaddev/paths/TrainingExport/LicenseForecast/LicenseForecast_NN_Tuner",
        mode='download',  # ro_mount  vs download
    ),    
} # 

# set outputs
nn_pred_outputs={
    "output_dir": Output(
        type="uri_folder",
        path=f"azureml://datastores/chiemoaddev/paths/TrainingExport/LicenseForecast/LicenseForecast_NN_Pred",
        mode="rw_mount",
    ),
}


# define the command
nn_pred_job = command(
    experiment_name='license_forecast_predictor',
    display_name="LicenseForecast_NN_Pred",  # for mlflow
    code="./src",
    command="python neural_forecast_predict.py",
    environment=env,
    instance_count=3,
    distribution={
        "type": "Ray",
        "include_dashboard": True,
        },   
    environment_variables=env_var,
    compute="lowpriNVLINK",
    shm_size='600g',
    inputs=nn_pred_inputs,
    outputs=nn_pred_outputs,
    identity=ManagedIdentityConfiguration(), # grant storage blob read IAM to cluster SP
)


@dsl.pipeline(description="LicenseForecastAll")
def forecast_pipeline():
    stats_tune = stats_tune_job()
    nn_tune = nn_tune_job()
    
    # force ordering
    stats_pred = stats_pred_job(tune_files=stats_tune.outputs.output_dir)
    nn_pred = nn_pred_job(tune_files=nn_tune.outputs.output_dir)

# gen pipeline
pipeline_to_submit = forecast_pipeline()
pipeline_to_submit.settings.force_rerun = True

pipeline_job = ml_client.jobs.create_or_update(pipeline_to_submit)
pipeline_job

