import numpy as np
import pandas as pd
import ray
from neuralforecast.losses.numpy import mse, mae
import pandas as pd
import pickle
from ray.util.actor_pool import ActorPool
from itertools import cycle  
import pandas as pd
import numpy as np
from typing import Optional
from azure.keyvault.secrets import SecretClient
from azure.storage.filedatalake import DataLakeServiceClient  
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from statsforecast.models import *
from statsforecast import StatsForecast
import io



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


# retrieve tuned parameters
pickle_bytes = [fetch(file_path, 'chiemoaddevfs', TENANT, CLIENTID, CLIENTSECRET) 
              for file_path in ['TrainingExport/LicenseForecast/LicenseForecast_Stats_Tuner/all.pickle',
                                'TrainingExport/LicenseForecast/LicenseForecast_Stats_Tuner/braga.pickle',
                                'TrainingExport/LicenseForecast/LicenseForecast_Stats_Tuner/kingsgate.pickle',
                                'TrainingExport/LicenseForecast/LicenseForecast_Stats_Tuner/ruby_mountain.pickle',]]


dataset_list = ["All", "Braga", "Kingsgate", "Ruby Mountain"]


for i, data_set in enumerate(dataset_list):
    print(f"Now on dataset: {data_set}")

    # horizon
    horizon = 90

    # data freq
    freq = 'D'

    # test data
    tseries_df = generate_forecasting_df(df=df, **test_dict[data_set])
    print(f"Generated a DF of shape: {tseries_df.shape}")

    # open tune results
    best_config = pickle.loads(pickle_bytes[i])

    file_name = data_set.replace(' ', '_') + '_Stats.parquet'

    model_list = {"AutoARIMA": AutoARIMA,
                "AutoETS": AutoETS,
                "AutoTheta": AutoTheta}

    @ray.remote(num_cpus=1)
    class BatchPredictor:
        def __init__(self, data, params, model_list):  
            self.results = {}
            self.data = data  
            self.horizon = 90  
            self.params = params
            self.model_list = model_list

        def predict(self, feature_name):
            print(f"Now on feature: {feature_name}")

            # get model
            best_model_config = self.model_list.get(self.params[feature_name]['model'])()

            # get model str
            model_str = str(best_model_config)

            # get data -- show how we did on the test set
            X_test = self.data.query(f"unique_id ==  '{feature_name}' ").reset_index(drop=True)

            # add constant
            X_test['y'] += 10

            # y_test now has + 10
            y_test = X_test.tail(horizon)
            X_test = X_test.drop(X_test.tail(horizon).index)

            # fit best config
            sf = StatsForecast(
                df=X_test, 
                models=[best_model_config],
                freq=freq, 
                n_jobs=-1,
            )

            # forecast
            forecasts = sf.forecast(h=horizon, level=[80]).reset_index()
            forecasts['unique_id'] = forecasts['unique_id'].astype('string')
            forecasts = forecasts.merge(y_test, how='inner', on=['ds', 'unique_id'])

            # undo constant
            forecasts['y'] -= 10
            forecasts[model_str] -= 10
            forecasts[model_str + '-lo-80'] -= 10
            forecasts[model_str + '-hi-80'] -= 10
            
            # row-wise error
            forecasts['MAE'] = forecasts.apply(lambda row:  mae(row['y'], row[model_str]), axis=1 ).astype(int)

            # generate inference predictions now
            X_test = self.data.query(f"unique_id == '{feature_name}'").reset_index(drop=True)

            # add constant
            X_test['y'] += 10

            # fit again
            sf = StatsForecast(
                df=X_test, 
                models=[best_model_config],
                freq=freq, 
                n_jobs=-1,
            )

            # forecast
            forecasts_future = sf.forecast(h=horizon, level=[80]).reset_index()  
            print(forecasts_future)

            # undo constant
            forecasts_future[model_str] -= 10
            forecasts_future[model_str + '-lo-80'] -= 10
            forecasts_future[model_str + '-hi-80'] -= 10

            # join
            forecasts = pd.concat([forecasts, forecasts_future], axis=0)

            # rename
            forecasts = forecasts.rename(columns={"unique_id": "feature_name",
                                                "ds": "date", 
                                                model_str: "prediction",
                                                model_str + '-lo-80': "prediction_low_80",
                                                model_str + '-hi-80': "prediction_high_80",
                                                "MAE": "absolute_error",
                                                "y": "actual",
            })
            print(f"Number of results after prediction for feature {feature_name}: {forecasts.shape}")  
                

            # add monthYear
            forecasts['monthYear'] = pd.to_datetime(forecasts['date']) + pd.offsets.MonthEnd(n=0)

            # add project
            forecasts['project'] = data_set

            # add model
            forecasts['model'] = model_str

            self.results[feature_name] = forecasts   
        
        def get_results(self):      
            # concatenate all results    
            if self.results:  
                all_results = pd.concat(self.results.values(), axis=0)    
                return all_results    
            else:  
                return pd.DataFrame()

        def shutdown(self):  
            ray.actor.exit_actor()  

    # set of features
    feature_names = list(best_config.keys())
    print(f"Using this feature list: {feature_names}")

    # storage model / df  
    test_df_ref = ray.put(tseries_df)  
    num_cpus = 396
    
    # create one actor per CPU    
    actors = [BatchPredictor.remote(data=test_df_ref, params=best_config, model_list=model_list) for _ in range(num_cpus)]    
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
    storage_df = pd.concat(final_results, axis=0) 

    # kill actors    
    for actor in actors:  
        actor.shutdown.remote()  
        ray.kill(actor)  


    # cast
    storage_df['prediction'] = storage_df['prediction'].astype(int)
    storage_df['prediction_low_80'] = storage_df['prediction_low_80'].astype(int)
    storage_df['prediction_high_80'] = storage_df['prediction_high_80'].astype(int)
    storage_df['feature_name'] = storage_df['feature_name'].astype('string')
    storage_df['project'] = storage_df['project'].astype('string')
    storage_df['date'] = storage_df['date'].astype('string')
    storage_df['monthYear'] = storage_df['monthYear'].astype('string')

    # clamp negatives
    storage_df['prediction'] = storage_df['prediction'].clip(lower=0)
    storage_df['prediction_low_80'] = storage_df['prediction_low_80'].clip(lower=0)
    storage_df['prediction_high_80'] = storage_df['prediction_high_80'].clip(lower=0)

    # save
    write_file_to_datalake(account_name='moaddevlake',
                             file_system_name='chiemoaddevfs',
                             directory_name='TrainingExport/LicenseForecast/LicenseForecast_Stats_Predict',
                             file_name=f'{file_name}',
                             data=storage_df)


# close
ray.shutdown()
