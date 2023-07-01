from adlfs import AzureBlobFileSystem
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
import pyarrow.compute as pc
import ray
import psutil
import pandas as pd
from ray.data import ActorPoolStrategy
from azureml.core import Run


if not ray.is_initialized():
    ray.init()

# repartition for parallel
num_cpus = psutil.cpu_count(logical=False)
num_partitions = min(num_cpus - 2, 32)

# to use managed identity, requires azure-keyvault-secrets==4.6.0 azure-identity==1.11.0; useful for jupyter
secret_client = SecretClient(vault_url="https://moaddev6131880268.vault.azure.net/", credential=DefaultAzureCredential())

# set storage options
storage_options={
                 'account_name': 'moaddevlake',
                 'tenant_id': "72f988bf-86f1-41af-91ab-2d7cd011db47",
                 'client_id': secret_client.get_secret("dev-synapse-sqlpool-sp-id").value,
                 'client_secret': secret_client.get_secret("dev-synapse-sqlpool-sp-secret").value,
                 }

# apply storage connector info
abfs = AzureBlobFileSystem(**storage_options)

# get the data and filter
ds = ray.data.read_parquet(
    ["az://chiemoaddevfs/out/Gold/ML/RandomizedTest/"],  # container / path
    filesystem=abfs,
    use_threads=True,
    filter=pc.field('instance_id').isin(['20230613']),   
    parallelism=num_partitions
).repartition(num_blocks=num_partitions, shuffle=False)

# alternatively getting the data from AML Input
# aml_context = Run.get_context()
# ds = ray.data.read_parquet(aml_context.input_datasets['RandomizedTest'],
#                             use_threads=True,
#                             columns=feature_selection + ['SYNDROME', 'INPUT__YML__testname', 'instance_id'],
#                             parallelism=num_partitions,
#                             filter=pc.field('instance_id').isin(['20230613']),
#                             ).repartition(num_partitions, shuffle=False)

# get count
ds.count()

# get columns
ds.columns()

# udf map
def transform_df(df: pd.DataFrame) -> pd.DataFrame:

    # recode lower counts
    threshold = 75000

    # identify rows with less than threshold occurrences
    m = df.groupby('SYNDROME')['SYNDROME'].transform('size').lt(threshold)

    # update
    df.loc[m, 'SYNDROME'] = 'other'

    # factorize label
    df["SYNDROME"] = pd.factorize(df['SYNDROME'])[0]
    return df

# transform
ds = ds.map_batches(
                    transform_df,
                    compute=ActorPoolStrategy(min_size=2, max_size=num_cpus), # how many CPU cores to claim
                    batch_format="pandas",
                    )

# trigger operation
ds.count()

# get class label mappings
classes = {label: i for i, label in enumerate(ds.unique("SYNDROME"))}

# udf
def transform_df(df: pd.DataFrame, classes) -> pd.DataFrame:
    '''
    map strings to int labels
    '''
    df['SYNDROME'] = df['SYNDROME'].map(classes)
    return df

# map
print(f'Transforming data...')
ds = ds.map_batches(transform_df, fn_kwargs={"classes": classes}, batch_format="pandas", compute=ActorPoolStrategy(min_size=2, max_size=num_cpus),)
