# set up environment
env = Environment(
    image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
    conda_file="./src/conda_env.yml",
    name="my-env"
)

# set inputs
inputs={
    "RTE": Input(
        type="uri_folder",
        path="azureml://datastores/chiemoaddev/paths/SupportFiles/GDCOFill",
        mode='download',  # vs download
    ),
}

# set outputs
outputs={
    "output1": Output(
        type="uri_folder",
        path="azureml://datastores/chiemoaddev/paths/TrainingExport/GDCOFill",
    ),
}

accelerate_cmd = (f"""accelerate launch \
--config_file default_config.yaml \
--use_deepspeed \
accelerate_gdcofill.py \
--model_name_or_path google/flan-t5-small \
--mixed_precision bf16 \
--gradient_accumulation_steps 1 \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 32 \
--num_train_epochs 20 \
--logging_steps 10 \
--no-load_ds_from_disk \
--output_dir flan_t5_small_gpu \
--learning_rate 3e-4 \
--weight_decay 0.01 \
--checkpointing_steps epoch \
--lr_scheduler_type constant \
--max_source_len 280 \
--max_target_len 101 \
--eval False \
--lora False \
--lora_r 8 \
--lora_alpha 16 \
--int8 False \
--gradient_checkpointing False \
--zero_stage 2 \
--num_cpu_threads_per_process 18 \
--gradient_clipping 1 \
--fused_adam False \
--adafactor""")


# download settings
env_var = {
"RSLEX_DOWNLOADER_THREADS": 48,  # download  ; num_cores * 4
"AZUREML_DATASET_HTTP_RETRY_COUNT": 10,  # download
}

# define the command
command_job = command(
    experiment_name='accelerate_gdcofill',
    code="./src",
    command=f"{accelerate_cmd}",
    environment=env,
    instance_count=1,
    environment_variables=env_var,
    compute="A100S",
    shm_size='32g',
    inputs=inputs,
    outputs=outputs,
    identity=ManagedIdentityConfiguration(), # grant storage blob read IAM to cluster SP
)


returned_job = ml_client.jobs.create_or_update(command_job)
returned_job