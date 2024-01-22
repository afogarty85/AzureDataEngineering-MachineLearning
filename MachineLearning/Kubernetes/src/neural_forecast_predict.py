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
from tqdm.auto import tqdm
import pickle
import os
from ray.tune import CLIReporter
from neuralforecast.losses.pytorch import MSE, DistributionLoss
from ray.util.actor_pool import ActorPool
import time
from ray.train import CheckpointConfig
import gc  
import time  
from itertools import cycle  
import pandas as pd
import numpy as np
from typing import Optional
from azure.keyvault.secrets import SecretClient
from azure.storage.filedatalake import DataLakeServiceClient  
from azure.identity import DefaultAzureCredential, ClientSecretCredential
import io
from pathlib import Path
from datetime import datetime  




def write_file_to_datalake(account_name, file_system_name, directory_name, file_name, data):
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

    writtenbytes = io.BytesIO()
    data.to_parquet(writtenbytes)

    # Upload the serialized data
    file_client.create_file()  # Comment this line if the file already exists and you're appending
    file_client.upload_data(writtenbytes.getvalue(), length=len(writtenbytes.getvalue()), connection_timeout=3000, overwrite=True)

    print(f"File '{file_name}' written to the Data Lake in directory '{directory_name}'.")


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


pickle_bytes = [fetch(file_path, 'chiemoaddevfs', TENANT, CLIENTID, CLIENTSECRET) 
              for file_path in ['TrainingExport/LicenseForecast/LicenseForecast_NN_Tuner/all.pickle',
                                'TrainingExport/LicenseForecast/LicenseForecast_NN_Tuner/braga.pickle',
                                'TrainingExport/LicenseForecast/LicenseForecast_NN_Tuner/kingsgate.pickle',
                                'TrainingExport/LicenseForecast/LicenseForecast_NN_Tuner/ruby_mountain.pickle',]]


dataset_list = ["All", "Braga", "Kingsgate", "Ruby Mountain"]

local_mnt_path = "/mnt/tuning_files"


for i, data_set in enumerate(dataset_list):
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

    # drop test
    tseries_df = tseries_df.groupby('unique_id').apply(lambda x: x.iloc[:-horizon]).reset_index(drop=True)  

    # open tune results
    print("Retrieving tuning file...")
    best_config = pickle.loads(pickle_bytes[i])

    del best_config['loss']

    other_kwargs = {
        "loss": DistributionLoss(distribution='Poisson', level=[80], return_params=False),
        "logger": False,
    }

    file_name = data_set.replace(' ', '_') + '_NN.parquet'
    model_name = data_set.replace(' ', '_') + '_NN'

    # fit best config
    print("Fitting best config...")
    model = PatchTST(**(best_config | other_kwargs)  )
    nf_best = NeuralForecast(models=[model], freq=freq)
    nf_best.fit(df=tseries_df, verbose=False)

    # save model
    model_pathing = local_mnt_path + f'/fine_tuned_model_{model_name}'  
    nf_best.save(path=model_pathing,
            model_index=None, 
            overwrite=True,
            save_dataset=True)

    # test data
    test_df = generate_forecasting_df(df=df, **test_dict[data_set])


    @ray.remote(num_gpus=0.5, num_cpus=0.5, max_restarts=3)
    class BatchPredictor:
        def __init__(self, model_path, data, max_retries=3, retry_delay=5):  
            self.model_path = model_path
            self.results = {}
            self.data = data  
            self.horizon = 90  
            self.max_retries = max_retries  
            self.retry_delay = retry_delay  
            self.load_model()  

        def load_model(self):  
            for i in range(self.max_retries):  
                try:  
                    self.model = NeuralForecast.load(path=self.model_path)  
                    break  
                except Exception as e:  
                    if i < self.max_retries - 1:  # i is zero indexed  
                        time.sleep(self.retry_delay)  
                        continue  
                    else:  
                        raise e  

        def predict(self, feature_name):
            try:
                print(f"Now on feature: {feature_name}")
                X_test = self.data.query(f"unique_id == '{feature_name}' ").reset_index(drop=True)
                y_test = X_test.tail(self.horizon)
                X_test = X_test.drop(X_test.tail(self.horizon).index)

                # forecast vs test
                forecasts = self.model.predict(df=X_test)
                forecasts = forecasts.reset_index().merge(y_test, how='inner', on=['ds', 'unique_id'])
                
                # row-wise error
                forecasts['MAE'] = forecasts.apply(lambda row:  mae(row['y'], row['PatchTST']), axis=1 ).astype(int)

                # in the the future
                X_test = self.data.query(f"unique_id == '{feature_name}'").reset_index(drop=True)
                forecasts_future = self.model.predict(df=X_test).reset_index()
                forecasts = pd.concat([forecasts, forecasts_future], axis=0)
                self.results[feature_name] = forecasts
            except Exception as e:
                print("Failure predicting: {feature_name}", e)
                
        def get_results(self):      
            # concatenate all results    
            if self.results:  
                all_results = pd.concat(self.results.values(), axis=0)    
                return all_results    
            else:  
                return pd.DataFrame()
    
        def shutdown(self):  
            ray.actor.exit_actor()  


    # storage model / df  
    test_df_ref = ray.put(test_df)  
    
    feature_names = test_df['unique_id'].unique()
    num_gpus = 1  
    
    # create one actor per GPU 
    actors = [BatchPredictor.remote(model_pathing, test_df_ref) for _ in range(num_gpus)]    
    pool = ActorPool(actors)    

    # create tasks for all features  
    tasks = []  
    for actor, feature_name in zip(cycle(actors), feature_names):  
        tasks.append(actor.predict.remote(feature_name))  


    # block until all tasks finish and get the results  
    results = ray.get(tasks)

    # get 
    retrieval_tasks = [actor.get_results.remote() for actor in actors]  

    # gather
    final_results = ray.get(retrieval_tasks)  
    storage_df = pd.concat(final_results, axis=0).reset_index(drop=True)

    # kill actors    
    for actor in actors:  
        actor.shutdown.remote()  
        ray.kill(actor)  
        gc.collect()  


    # rename
    storage_df = storage_df.rename(columns={"PatchTST-median": "prediction", "MAE": "absolute_error",
                                            "PatchTST-lo-80": "prediction_low_80", "PatchTST-hi-80": "prediction_high_80",
                                            "y": "actual", "ds": "date", "unique_id": "feature_name"})


    # add monthYear
    storage_df['monthYear'] = pd.to_datetime(storage_df['date']) + pd.offsets.MonthEnd(n=0)

    # add project
    storage_df['project'] = data_set

    # add model
    storage_df['model'] = 'PatchTST'

    # cast
    storage_df['prediction'] = storage_df['prediction'].astype(int)
    storage_df['prediction_low_80'] = storage_df['prediction_low_80'].astype(int)
    storage_df['prediction_high_80'] = storage_df['prediction_high_80'].astype(int)
    storage_df['feature_name'] = storage_df['feature_name'].astype('string')
    storage_df['project'] = storage_df['project'].astype('string')
    storage_df['date'] = storage_df['date'].astype('string')
    storage_df['monthYear'] = storage_df['monthYear'].astype('string')

    # drop
    storage_df = storage_df.drop(['PatchTST'], axis=1).reset_index(drop=True)

    # get date
    date_string = datetime.now().strftime("%Y-%m-%d") + '_'


    # timestamp the preds
    storage_df['inference_as_of'] = datetime.now().strftime("%Y-%m-%d")

    # save
    write_file_to_datalake(account_name='moaddevlake',
                             file_system_name='chiemoaddevfs',
                             directory_name='TrainingExport/LicenseForecast/LicenseForecast_NN_Predict',
                             file_name=f'{date_string + file_name}',
                             data=storage_df)




# close
ray.shutdown()

#


