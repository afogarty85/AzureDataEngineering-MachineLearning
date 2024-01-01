import numpy as np
import pandas as pd
import ray
from neuralforecast.losses.numpy import mse, mae
import pandas as pd
from azureml.core import Run
import pickle
from ray.tune import CLIReporter
from ray.util.actor_pool import ActorPool
import gc  
import time  
from itertools import cycle  
import pandas as pd
import numpy as np
from typing import Optional
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
import nest_asyncio
import asyncio
from io import BytesIO
from azure.identity.aio import ClientSecretCredential
from azure.storage.filedatalake.aio import DataLakeServiceClient
from statsforecast.models import *
from statsforecast import StatsForecast
nest_asyncio.apply()


# alt fast method
async def fetch(file_path, container, TENANT, CLIENTID, CLIENTSECRET):
    """
    Connects to ADLSGen2 and retrieves any file of choice as bytes
    """
    # get access to AAD
    async with ClientSecretCredential(TENANT, CLIENTID, CLIENTSECRET) as credential:
        # get access to ADLS
        async with DataLakeServiceClient(
                account_url=f"https://moaddevlake.dfs.core.windows.net", credential=credential,) as datalake_service_client:
            # get access to FS
            async with datalake_service_client.get_file_system_client(container) as file_system_client:
                file_client = file_system_client.get_file_client(f"/{file_path}")
                print(f"Generating data ADLS from file path: {file_path}")
                # pull and download file
                download = await file_client.download_file(max_concurrency=75)
                downloaded_bytes = await download.readall()
                # report progress
                print(f"Finished downloading bytes of length: {len(downloaded_bytes)}")
    return downloaded_bytes


def retrieve_file_from_lake(file_paths, container, TENANT, CLIENTID, CLIENTSECRET):
    print('downloading file...')
    # get df as bytes
    bytes_df = asyncio.run(asyncio.gather(*map(fetch, file_paths, container, TENANT, CLIENTID, CLIENTSECRET)))

    # get df
    df = pd.read_parquet(BytesIO(bytes_df[0]))
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
credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)

# secret vals
CLIENTID = client.get_secret("dev-synapse-sqlpool-sp-id").value
CLIENTSECRET = client.get_secret("dev-synapse-sqlpool-sp-secret").value
TENANT = client.get_secret("tenant").value

# get main df
df = retrieve_file_from_lake(
    file_paths=['out/Export/LicenseForecast/license_forecast_daily.parquet'], container=['chiemoaddevfs'],
    CLIENTID=[CLIENTID],
    CLIENTSECRET=[CLIENTSECRET],
    TENANT=[TENANT])




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


pickle_list = [
 '/All.pickle',
 '/Braga.pickle',
 '/Kingsgate.pickle',
 '/Ruby Mountain.pickle'
 ]

dataset_list = ["All", "Braga", "Kingsgate", "Ruby Mountain"]


aml_context = Run.get_context()
output_path = aml_context.output_datasets['output_dir']
tune_path = aml_context.input_datasets['tune_files']

for data_set, params in zip(dataset_list, pickle_list):
    print(f"Now on dataset: {data_set}")

    # horizon
    horizon = 90

    # data freq
    freq = 'D'

    # test data
    tseries_df = generate_forecasting_df(df=df, **test_dict[data_set])

    # open tune results
    print("Retrieving tuning file...")
    with open(tune_path + params, 'rb') as handle:
        best_config = pickle.load(handle)

    file_name = data_set + '_Stats.parquet'

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
            best_model_config = self.model_list.get(self.params[feature_name]['model'])(
                season_length=self.params[feature_name]['params']['season_length'])

            # get model str
            model_str = str(best_model_config)

            # get data -- show how we did on the test set
            X_test = self.data.query(f"unique_id ==  '{feature_name}' ").reset_index(drop=True)

            # add constant
            X_test['y'] += 10

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
            
            # row-wise error
            forecasts['MAE'] = forecasts.apply(lambda row:  mae(row['y'], row[model_str]-10), axis=1 ).astype(int)

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
                

            # add monthYear
            forecasts['monthYear'] = pd.to_datetime(forecasts['date']) + pd.offsets.MonthEnd(n=0)

            # add project
            forecasts['project'] = data_set

            # add model
            forecasts['model'] = model_str

            self.results[feature_name] = forecasts   
        
        def get_results(self):    
            # concatenate all results  
            all_results = pd.concat(self.results.values(), axis=0)  
            return all_results  

        def shutdown(self):  
            ray.actor.exit_actor()  


    # storage model / df  
    test_df_ref = ray.put(tseries_df)  
    num_cpus = 400  
    
    # create one actor per CPU    
    actors = [BatchPredictor.remote(data=test_df_ref, params=best_config, model_list=model_list) for _ in range(num_cpus)]    
    pool = ActorPool(actors)    

    # set of features
    feature_names = list(best_config.keys())

    # create tasks for all features  
    tasks = []  
    for actor, feature_name in zip(cycle(actors), feature_names):  
        tasks.append(actor.predict.remote(feature_name))  
    
    # block until all tasks finish and get the results  
    ray.get(tasks)  
    
    # get results from actors  
    results = [actor.get_results.remote() for actor in actors]  
    
    # gather all results  
    storage_df = pd.concat(ray.get(results), axis=0)  

    # kill actors    
    for actor in actors:  
        actor.shutdown.remote()  
        ray.kill(actor)  
        gc.collect()  


    # cast
    storage_df['prediction'] = storage_df['prediction'].astype(int)
    storage_df['prediction_low_80'] = storage_df['prediction_low_80'].astype(int)
    storage_df['prediction_high_80'] = storage_df['prediction_high_80'].astype(int)
    storage_df['feature_name'] = storage_df['feature_name'].astype('string')
    storage_df['project'] = storage_df['project'].astype('string')
    storage_df['date'] = storage_df['date'].astype('string')
    storage_df['monthYear'] = storage_df['monthYear'].astype('string')

    # drop constant
    storage_df['prediction'] = storage_df['prediction'] - 10
    storage_df['prediction_low_80'] = storage_df['prediction_low_80']  - 10
    storage_df['prediction_high_80'] = storage_df['prediction_high_80'] - 10

    # clamp negatives
    storage_df['prediction'] = storage_df['prediction'].clip(lower=0)
    storage_df['prediction_low_80'] = storage_df['prediction_low_80'].clip(lower=0)
    storage_df['prediction_high_80'] = storage_df['prediction_high_80'].clip(lower=0)

    # save
    storage_df.to_parquet(output_path + '/' + file_name)


# close
ray.shutdown()
