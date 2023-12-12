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

# init Azure ML run context data
aml_context = Run.get_context()
data_path = aml_context.input_datasets['train_files']
output_path = aml_context.output_datasets['output_dir']

# load data
df = pd.read_parquet(data_path + '/feature_name_by_week_by_peak.parquet')

# horizon
horizon = 12

# data freq
freq = 'W'

# drop test
df = df.groupby('unique_id').apply(lambda x: x.iloc[:-horizon]).reset_index(drop=True)  

# get features
feature_list = df['unique_id'].unique()

# storage
storage_configs = {}

for feature in tqdm(feature_list):
    print(f"Now on feature: {feature}")
    df = df.query(f"unique_id == '{feature}' ")

    config = {
        "learning_rate": tune.loguniform(1e-5, 5e-2),
        "max_steps": tune.choice([200, 300, 400, 500, 600, 700, 800, 1000]),
        "input_size": tune.choice([2, 3, 4, 5, 6,]),
        "batch_size": tune.choice([7, 14, 21, 28, 35, 50, 64, 96, 128, 256, 512, 768, 1024, 2056, 4096]),
        "windows_batch_size": tune.choice([4, 8, 16, 24, 32, 48, 64, 128, 256]),
        "stride": tune.choice([4, 8, 12, 16]),
        "hidden_size": tune.choice([64, 128, 256, 512, 1024]),
        "linear_hidden_size": tune.choice([64, 128, 256, 512, 1024]),
        "n_heads": tune.choice([4, 8, 16, ]),
        "patch_len": tune.choice([8, 12, 16, 24, 32]),
        "val_check_steps": 10,
        "random_seed": tune.randint(1, 10),
        "scaler_type": 'robust',
        "gradient_clip_val": 1,
        "accelerator": "auto",
        "h": horizon
    }

    early_stopping_args = {
        "monitor": "train_loss",
        "patience": 50,
        "min_delta": 0.05,
        "mode": "min",
    }

    pl_trainer_kwargs = {
        "callbacks": [EarlyStopping( **early_stopping_args,    )  ],
    }

    num_workers = 4
    scheduler = ASHAScheduler(max_t=1000,
                            grace_period=150,
                            reduction_factor=2,
                            metric="mse",
                            mode="min",)


    def tune_obj(config, df, horizon, freq):

        try:
            model = PatchTST(**(config )  )
            nf = NeuralForecast(models=[model], freq=freq)
            nf.fit(df=df, val_size=horizon, verbose=False)
            forecast_df = nf.predict_insample(step_size=1)
            error = mse(y=forecast_df.tail(horizon)['y'], y_hat=forecast_df.tail(horizon)['PatchTST'])
            metrics = {"mse": error,}
            ray.train.report(metrics)

        except Exception as e:  
            ray.train.report({"mse": np.nan})

    # run the hyperparameter tuning  
    analysis = tune.run(  
        tune.with_parameters(tune_obj, df=df, horizon=horizon, freq=freq),  
        num_samples=20,
        config=config,
        search_alg=HyperOptSearch(metric='mse', mode='min'),
        resources_per_trial={"cpu": 5, "gpu": 1 / num_workers}  
    )
    
    # Get the best hyperparameters  
    best_config = analysis.get_best_config(metric="mse", mode="min")

    storage_configs[feature] = best_config


# save on lake
with open(output_path + '/PatchTST_storage_configs.pickle', 'wb') as handle:
    pickle.dump(storage_configs, handle, protocol=pickle.HIGHEST_PROTOCOL)




# # test
# df = pd.read_parquet('data/feature_name_by_week_by_peak.parquet')
# X_test = df.query("unique_id == 'PrimeTime-ADV' ")
# y_test = X_test.tail(horizon)
# X_test = X_test.drop(X_test.tail(horizon).index)


# # use params to fit new model
# models = [PatchTST(**(best_config | pl_trainer_kwargs )  ) ]
# nf_best = NeuralForecast(models=models, freq=freq)
# nf_best.fit(df=X_test, )
# forecasts = nf_best.predict()
# forecasts = forecasts.reset_index().merge(y_test, how='inner', on=['ds', 'unique_id'])

# print('MAE', mae(forecasts['PatchTST'].values, forecasts['y'].values)  )


# # plot
# # Convert 'ds' to datetime
# forecasts['ds'] = pd.to_datetime(forecasts['ds'])

# # Set 'ds' as index
# forecasts.set_index('ds', inplace=True)

# # Plot NHITS and y over time
# plt.figure(figsize=(10, 6))
# plt.plot(forecasts.index, forecasts['PatchTST'], label='PatchTST')
# plt.plot(forecasts.index, forecasts['y'], label='y')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.title('PatchTST and y over time')
# plt.legend()
# plt.show()
