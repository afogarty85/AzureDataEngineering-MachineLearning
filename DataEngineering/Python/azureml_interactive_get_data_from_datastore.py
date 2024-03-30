from azureml.core import Workspace, Datastore, Dataset  
# azureml-dataprep==4.12.1
ws = Workspace.from_config()
datastore = Datastore.get(ws, "chiemoaddev")
dataset = Dataset.File.from_files(path=(datastore, 'TrainingExport/TicketSnapshotInferenceModel/Llama2-13b/SFTComplete/SFT/TorchTrainer_b8280_00000_0_2024-02-08_01-52-30/checkpoint_000001/checkpoint'))
mounted_path = dataset.mount()
mounted_path.start()  

# Get the mount point
dataset_mount_folder = mounted_path.mount_point