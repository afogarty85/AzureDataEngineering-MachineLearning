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
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tqdm.auto import tqdm
from azureml.core import Run
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

# init Azure ML run context data
aml_context = Run.get_context()
data_path = aml_context.input_datasets['train_files']
output_path = aml_context.output_datasets['output_dir']
tune_path = aml_context.input_datasets['tune_files']


train_dataset_list = [
 '/feature_name_by_day_by_peak.parquet',
 '/feature_name_day_by_Braga_peak.parquet',
 '/feature_name_day_by_Kingsgate_peak.parquet',
 '/feature_name_day_by_Ruby Mountain_peak.parquet'
 ]

pickle_list = [
 '/feature_name_by_day_by_peak.parquet.pickle',
 '/feature_name_day_by_Braga_peak.parquet.pickle',
 '/feature_name_day_by_Kingsgate_peak.parquet.pickle',
 '/feature_name_day_by_Ruby Mountain_peak.parquet.pickle'
 ]

test_list = [
 '/feature_name_by_day_nolimit_all.parquet',
 '/feature_name_by_day_nolimit_Braga.parquet',
 '/feature_name_by_day_nolimit_Kingsgate.parquet',
 '/feature_name_by_day_nolimit_RubyMountain.parquet'
 ]

out_file_names = [
    '/all.parquet',
    '/braga.parquet',
    '/kingsgate.parquet',
    '/rubymountain.parquet',
]

for train_set, params, test_set, out_name in zip(train_dataset_list, pickle_list, test_list, out_file_names):

    # load data
    df = pd.read_parquet(data_path + train_set)

    # horizon
    horizon = 90

    # data freq
    freq = 'D'

    # drop test
    df = df.groupby('unique_id').apply(lambda x: x.iloc[:-horizon]).reset_index(drop=True)  

    # open tune results
    print("Retrieving tuning file...")
    with open(tune_path + params, 'rb') as handle:
        best_config = pickle.load(handle)

    del best_config['loss']

    other_kwargs = {
        "loss": DistributionLoss(distribution='Poisson', level=[80], return_params=False),
        "logger": False,
    }

    file_name = out_name.split('/')[-1].replace('.parquet', '')


    # fit best config
    print("Fitting best config...")
    model = PatchTST(**(best_config | other_kwargs)  )
    nf_best = NeuralForecast(models=[model], freq=freq)
    nf_best.fit(df=df, verbose=False)

    # save model
    nf_best.save(path=output_path + f'/fine_tuned_model_{file_name}',
            model_index=None, 
            overwrite=True,
            save_dataset=True)


    # test data
    test_df = pd.read_parquet(data_path + test_set)


    @ray.remote(num_gpus=0.5, num_cpus=0.5)
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
        
        def get_results(self):    
            # concatenate all results  
            all_results = pd.concat(self.results.values(), axis=0)  
            return all_results  
    
        def shutdown(self):  
            ray.actor.exit_actor()  


    # storage model / df  
    test_df_ref = ray.put(test_df)  
    
    feature_names = test_df['unique_id'].unique()    
    num_gpus = 24  
    
    # create one actor per GPU    
    actors = [BatchPredictor.remote(output_path + f'/fine_tuned_model_{file_name}', test_df_ref) for _ in range(num_gpus)]    
    pool = ActorPool(actors)    

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


    # rename
    storage_df = storage_df.rename(columns={"PatchTST-median": "prediction", "MAE": "absolute_error",
                                            "PatchTST-lo-80": "prediction_low_80", "PatchTST-hi-80": "prediction_high_80",
                                            "y": "actual", "ds": "Date", "unique_id": "feature_name"})


    # add monthYear
    storage_df['monthYear'] = pd.to_datetime(storage_df['Date']) + pd.offsets.MonthEnd(n=0)

    # add project
    storage_df['project'] = file_name

    # cast
    storage_df['prediction'] = storage_df['prediction'].astype(int)
    storage_df['prediction_low_80'] = storage_df['prediction_low_80'].astype(int)
    storage_df['prediction_high_80'] = storage_df['prediction_high_80'].astype(int)
    storage_df['feature_name'] = storage_df['feature_name'].astype('string')
    storage_df['project'] = storage_df['project'].astype('string')
    storage_df['Date'] = storage_df['Date'].astype('string')
    storage_df['monthYear'] = storage_df['monthYear'].astype('string')

    # drop
    storage_df = storage_df.drop(['PatchTST'], axis=1).reset_index(drop=True)




    # save
    storage_df.to_parquet(output_path + out_name)


# close
ray.shutdown()

#


