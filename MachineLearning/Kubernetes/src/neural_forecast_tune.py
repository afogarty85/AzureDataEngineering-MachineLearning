import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
import ray
from neuralforecast.losses.numpy import mse, mae
import pandas as pd
from ray.tune.schedulers import ASHAScheduler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from azureml.core import Run
import pickle
import os
from ray.tune import CLIReporter
from neuralforecast.losses.pytorch import MSE, DistributionLoss
from ray.train import CheckpointConfig
from ray.tune.search import ConcurrencyLimiter
import pandas as pd
import numpy as np
from typing import Optional
from azure.keyvault.secrets import SecretClient
from azure.storage.filedatalake import DataLakeServiceClient  
from azure.identity import DefaultAzureCredential, ClientSecretCredential
import io



ray.init(ignore_reinit_error=True)


def write_pickle_to_datalake(account_name, file_system_name, directory_name, file_name, data):
    """
    Writes a Python object as a pickle file to Azure Data Lake Storage Gen2.
    
    account_name: The name of the Azure Data Lake Storage Gen2 account.
    file_system_name: The name of the file system (container) in Azure Data Lake.
    directory_name: The name of the directory within the file system.
    file_name: The name of the pickle file to create or update.
    data: The Python object to serialize and write to the pickle file.
    """
    
    # Authenticate with Azure using DefaultAzureCredential
    service_client = DataLakeServiceClient(account_url=f"https://{account_name}.dfs.core.windows.net",
                                           credential=DefaultAzureCredential())

    # Get a reference to the file system (container) and directory
    file_system_client = service_client.get_file_system_client(file_system=file_system_name)
    directory_client = file_system_client.get_directory_client(directory_name)

    # Get a reference to the file
    file_client = directory_client.get_file_client(file_name)

    # Serialize the data with pickle
    serialized_data = pickle.dumps(data)

    # Upload the serialized data
    file_client.create_file()  # Comment this line if the file already exists and you're appending
    file_client.append_data(data=serialized_data, offset=0, length=len(serialized_data))
    file_client.flush_data(len(serialized_data))

    print(f"Pickle file '{file_name}' written to the Data Lake in directory '{directory_name}'.")



def fetch(file_path, container, TENANT, CLIENTID, CLIENTSECRET):
    """
    Connects to ADLSGen2 and retrieves any file of choice as bytes
    """
    # Create a credential object using client secret
    credential = ClientSecretCredential(TENANT, CLIENTID, CLIENTSECRET)

    # Create a DataLakeServiceClient object
    datalake_service_client = DataLakeServiceClient(
        account_url=f"https://moaddevlake.dfs.core.windows.net", credential=credential)
    
    # Get a file system client object
    file_system_client = datalake_service_client.get_file_system_client(container)
    
    # Get a file client object
    file_client = file_system_client.get_file_client(file_path)
    print(f"Generating data ADLS from file path: {file_path}")
    
    # Download file
    download = file_client.download_file()
    downloaded_bytes = download.readall()
    
    # Report progress
    print(f"Finished downloading bytes of length: {len(downloaded_bytes)}")
    
    return downloaded_bytes

def retrieve_file_from_lake(file_paths, container, TENANT, CLIENTID, CLIENTSECRET):
    print('downloading file...')

    # get df as bytes
    bytes_df = [fetch(file_path, container, TENANT, CLIENTID, CLIENTSECRET) for file_path in file_paths]

    # get df from the first file (assuming file_paths is a list of file paths)
    df = pd.read_parquet(io.BytesIO(bytes_df[0]))
    print('rx file...', df.shape)
    
    return df



def generate_timeseries(start, end, freq, unique_id):
    '''
    start -- '2021-01-01'
    end   -- '2022-11-30'
    freq  -- '1D', 'MS', 'M'
    ids   -- array of unique ids to repeat/replicate across time
    '''
    # daily time range
    trend_range = pd.date_range(start=start, end=end, freq=freq)

    # extend time
    extended_time = np.array([trend_range] *  len(unique_id) ).flatten()

    # extend features
    extended_id = np.repeat(unique_id, len(trend_range), axis=-1)

    # gen data
    tseries = pd.DataFrame({'Date': extended_time, 'unique_id': extended_id})

    # to date
    tseries['Date'] = pd.to_datetime(tseries['Date'])

    # set monthYear dt
    tseries['monthYear'] = pd.to_datetime(tseries['Date']).dt.to_period('M')
    return tseries




def generate_forecasting_df(df: pd.DataFrame, size: int, grain: str, project_peak=False, name: Optional[str] = None) -> pd.DataFrame:
    '''
    path - path to license_forecast_daily
    name - the project name, choose: [Kingsgate, Braga, Ruby Mountain]
    grain - how to transform the data, choose: [day, month]
    size - the cutoff for the timeseries, e.g., I need at least 6 months of data
    project_peak - if True, use FeatureName Proj Peak


    unique_id -- id of series
    ds -- datestamp
    y -- target
    '''

    # set filter
    project_filter = name
    grain_filter = grain

    # sort
    df = df.sort_values(by=['feature_name', 'Date']).reset_index(drop=True)

    # reduce
    df = df[['project_recoded', 'Date', 'feature_name', 'FeatureName_Peak', 'FeatureName_Proj_Peak', 'user_id']]

    # cast
    df['Date'] = pd.to_datetime(df['Date'])

    # rename
    df = df.rename(columns={"feature_name": "unique_id"})

    # rename conditional
    df = df.rename(columns={"FeatureName_Proj_Peak": "y"}) if project_peak else df.rename(columns={"FeatureName_Peak": "y"})

    # filter
    if project_peak:
        # just the proj data
        subset = df.query(f'project_recoded == "{project_filter}" ').copy()

        # limit cols
        subset = subset[['unique_id', 'y', 'Date', 'user_id']]

        # get rid of non-named features            
        feature_list = [x for x in subset['unique_id'].unique() if not any(c.isdigit() for c in x)]
        subset = subset[subset['unique_id'].isin(feature_list)].reset_index(drop=True)

    else:
        subset = df[['unique_id', 'y', 'Date', 'user_id']].copy()
    
        # get rid of non-named features            
        # feature_list = [x for x in subset['unique_id'].unique() if not any(c.isdigit() for c in x)]
        # print(f"Found this filter list: {feature_list}")
        # subset = subset[subset['unique_id'].isin(feature_list)].reset_index(drop=True)

    # gen time series so there are no gaps
    tseries = generate_timeseries(start=df['Date'].min(), end=df['Date'].max(), freq='D', unique_id=subset['unique_id'].unique())

    # find feature name history
    if project_peak:
        min_max_start = df[df['y'].notna()].query(f"project_recoded == '{name}'").groupby(['unique_id']).agg({'Date': ['min', 'max']}).reset_index()
        min_max_start.columns = ['unique_id', 'startDateDay', 'endDateDay']

    else:
        min_max_start = df[df['y'].notna()].groupby(['unique_id']).agg({'Date': ['min', 'max']}).reset_index()
        min_max_start.columns = ['unique_id', 'startDateDay', 'endDateDay']      

    # join min_max dates
    tseries_df = tseries.merge(min_max_start, how='left', on=['unique_id',])

    # set date
    tseries_df['Date'] = pd.to_datetime(tseries_df['Date'])

    # keep just data that should/could have been observed
    tseries_df['to_remove'] = np.where(
        (tseries_df['Date'] < tseries_df['startDateDay']),
        1,
        0)

    tseries_df = tseries_df[tseries_df['to_remove'] == 0] \
        .reset_index(drop=True) \
        .drop(['startDateDay', 'endDateDay', 'to_remove'], axis=1)

    # join in actual data
    tseries_df = tseries_df.merge(subset, how='left', on=['Date', 'unique_id'])

    # fill na
    tseries_df = tseries_df.fillna(0)

    # agg by grain
    if grain_filter == 'month':
        tseries_df['monthYear'] = pd.to_datetime(tseries_df['Date']) + pd.offsets.MonthEnd(n=0)
        tseries_df = tseries_df.groupby(['unique_id', 'monthYear'], as_index=False).agg({"y": "max"})
        tseries_df = tseries_df.rename(columns={"monthYear": "ds"})

        # filter tseries with at least n units of data
        tseries_df['tsSize'] = tseries_df.groupby(['unique_id'])['ds'].transform('count')

        # just retain ts with at least n units of data
        tseries_df = tseries_df[tseries_df['tsSize'] >= size] \
                                .reset_index(drop=True) \
                                .drop(['tsSize'], axis=1)

        # sort
        tseries_df = tseries_df.sort_values(by=['ds']).reset_index(drop=True)

    if grain_filter == 'week':

        # First, ensure that the 'Date' column is of datetime type  
        tseries_df['Date'] = pd.to_datetime(tseries_df['Date'])  

        
        users = tseries_df.groupby(['unique_id', 'Date']).agg({"user_id": "nunique"}).reset_index()
        users.set_index('Date', inplace=True)  
        users_df = users.groupby('unique_id')['user_id'].resample('W').max().reset_index()

        # Set 'Date' as the index of the dataframe  
        tseries_df.set_index('Date', inplace=True)  
        
        # Resample the data on a weekly frequency and get the maximum value of 'y'  
        tseries_df = tseries_df.groupby('unique_id')['y'].resample('W').max().reset_index()  

        # add users
        tseries_df = tseries_df.merge(users_df, how='left', on=['unique_id', 'Date'])

        # filter tseries with at least n units of data
        tseries_df['tsSize'] = tseries_df.groupby(['unique_id'])['Date'].transform('count')

        # just retain ts with at least n units of data
        tseries_df = tseries_df[tseries_df['tsSize'] >= size] \
                                .reset_index(drop=True) \
                                .drop(['tsSize'], axis=1)
        
        # rename
        tseries_df = tseries_df.rename(columns={"Date": "ds"})

        # sort
        tseries_df = tseries_df.sort_values(by=['ds']).reset_index(drop=True)

    if grain_filter == 'day':
        tseries_df = tseries_df.groupby(['unique_id', 'Date'], as_index=False).agg({"y": "max"})
        tseries_df = tseries_df.rename(columns={"Date": "ds"})

        # filter tseries with at least 120 days of data -- get count by group
        tseries_df['tsSize'] = tseries_df.groupby(['unique_id'])['ds'].transform('count')

        # just retain ts with at least n units of data
        tseries_df = tseries_df[tseries_df['tsSize'] >= size] \
                                .reset_index(drop=True) \
                                .drop(['tsSize'], axis=1)
        
        # sort
        tseries_df = tseries_df.sort_values(by=['ds']).reset_index(drop=True)      

    # check
    assert tseries_df.groupby(['unique_id', 'ds'])['ds'].transform('count').sum() == tseries_df.shape[0], 'malformed time series!'
    return tseries_df



def find_zero_groups(df, group_column, zero_column, zero_frac):  
    # Group by group_column and calculate number of zeros in zero_column  
    zero_counts = df.groupby(group_column)[zero_column].apply(lambda x: (x==0).sum())  
  
    # Calculate total counts for each group  
    total_counts = df.groupby(group_column)[zero_column].count()  
  
    # Calculate percentage of zeros in each group  
    zero_percentage = zero_counts / total_counts * 100  
  
    # Find groups where percentage of zeros is x% or more  
    high_zero_groups = zero_percentage[zero_percentage >= zero_frac]  
  
    return high_zero_groups.reset_index()['unique_id'].unique()


# get secs
KVUri = "https://moaddev6131880268.vault.azure.net"

# init connection
credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)

# secret vals
CLIENTID = client.get_secret("dev-synapse-sqlpool-sp-id").value
CLIENTSECRET = client.get_secret("dev-synapse-sqlpool-sp-secret").value
TENANT = client.get_secret("tenant").value

# get main df
df = retrieve_file_from_lake(
    file_paths=['out/Export/LicenseForecast/license_forecast_daily.parquet'],
    container='chiemoaddevfs',
    CLIENTID=CLIENTID,
    CLIENTSECRET=CLIENTSECRET,
    TENANT=TENANT)


tune_dict = {
              "Braga": {"size": 365,
              "name": "Braga",
              "grain": "day",
              "project_peak": True},

              "Kingsgate": {"size": 365,
              "name": "Kingsgate",
              "grain": "day",
              "project_peak": True},

              "Ruby Mountain": {"size": 365,
              "name": "Ruby Mountain",
              "grain": "day",
              "project_peak": True},
              
              "All": {"size": 365,
              "name": "All",
              "grain": "day",
              "project_peak": False},
              }

test_dict = {
              "Braga": {"size": 90,
              "name": "Braga",
              "grain": "day",
              "project_peak": True},

              "Kingsgate": {"size": 90,
              "name": "Kingsgate",
              "grain": "day",
              "project_peak": True},

              "Ruby Mountain": {"size": 90,
              "name": "Ruby Mountain",
              "grain": "day",
              "project_peak": True},

              "All": {"size": 90,
              "name": "All",
              "grain": "day",
              "project_peak": False},
              }


dataset_list = ["Braga", "Kingsgate", "Ruby Mountain", "All"]


for data_set in dataset_list:
    print(f"Now on dataset: {data_set}")
    
    # gen data
    tseries_df = generate_forecasting_df(df=df, **tune_dict[data_set])

    # find zeros
    high_zeros = find_zero_groups(tseries_df, group_column='unique_id', zero_column='y', zero_frac=70)

    # drop
    tseries_df = tseries_df[~tseries_df['unique_id'].isin(high_zeros)].reset_index(drop=True)

    # horizon
    horizon = 90

    # data freq
    freq = 'D'

    # set n_windows
    n_windows = 5 if data_set == 'All' else 2

    # drop test
    tseries_df = tseries_df.groupby('unique_id').apply(lambda x: x.iloc[:-horizon]).reset_index(drop=True)  


    config = {
        "learning_rate": tune.loguniform(1e-6, 5e-2),
        "max_steps": tune.choice([150, 200, 300, 400, 500, 600, 700, 900, 1000, 1200, 1300, 1500]),
        "input_size": tune.choice([7, 14, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150,]),
        "batch_size": tune.choice([128, 512, 1024, 2056, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384, 17408, ]),
        "windows_batch_size": tune.choice([32, 48, 64, 128, 256, 512, 768]),
        "stride": tune.choice([4, 8, 12, 16]),
        "hidden_size": tune.choice([64, 128, 256, 512]),
        "linear_hidden_size": tune.choice([64, 128, 256, 512]),
        "n_heads": tune.choice([4, 8
        , 16, ]),
        "patch_len": tune.choice([8, 12, 16, 24, 32, 48, 64]),
        "random_seed": tune.randint(1, 25),
        "scaler_type": 'robust',
        "gradient_clip_val": 1,
        "accelerator": "gpu",
        "devices": "-1",
        "h": horizon,
        "loss": tune.choice([DistributionLoss(distribution='Poisson', level=[80], return_params=False)])
    }


    reporter = CLIReporter(
            parameter_columns=list(config.keys()),
            metric_columns=["mse", "mae", "training_iteration"],
        )

    scheduler = ASHAScheduler(max_t=1000,
                            grace_period=50,
                            reduction_factor=2,
                            metric="mse",
                            mode="min",)

    def tune_obj(config, df, horizon, freq):
        try:
            model = PatchTST(**(config )  )
            nf = NeuralForecast(models=[model], freq=freq)
            forecast_df = nf.cross_validation(df=df,
                                    step_size=horizon,
                                    n_windows=2,
                                    use_init_models=False,
                                    )
            mse_error = mse(y=forecast_df['y'], y_hat=forecast_df['PatchTST'])
            mae_error = mae(y=forecast_df['y'], y_hat=forecast_df['PatchTST'])
            metrics = {"mse": mse_error, "mae": mae_error}
            ray.train.report(metrics)

        except Exception as e:
            ray.train.report({"mse": np.nan, "mae": np.nan})


    search_alg = HyperOptSearch(metric='mse', mode='min', points_to_evaluate=None)
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=48)

    # run the hyperparameter tuning  
    analysis = tune.run(  
        tune.with_parameters(tune_obj, df=tseries_df, horizon=horizon, freq=freq),  
        num_samples=500,
        config=config,
        #scheduler=scheduler,
        local_dir="/mnt/tuning_files",
        checkpoint_config=CheckpointConfig(
                    num_to_keep=1, checkpoint_frequency=0),
        storage_path="/mnt/tuning_files",
        resume=False,
        search_alg=search_alg,
        progress_reporter=reporter,
        resources_per_trial=tune.PlacementGroupFactory([
                {"GPU": 1},
            ], strategy="SPREAD"),
    )

    # get the best hyperparameters  
    best_config = analysis.get_best_config(metric="mse", mode="min")
    print("Best Config:", best_config)

    # filename
    file_name = data_set.lower().replace(' ', '_')
    print(f"Generating a filename: {file_name}")

    # write
    write_pickle_to_datalake(account_name='moaddevlake',
    file_system_name='chiemoaddevfs',
    directory_name='TrainingExport/LicenseForecast/LicenseForecast_NN_Tuner',
    file_name=f'{file_name}.pickle',
    data=best_config)


ray.shutdown()

#


