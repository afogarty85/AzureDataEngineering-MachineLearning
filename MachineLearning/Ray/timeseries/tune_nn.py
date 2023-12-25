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

# init Azure ML run context data
aml_context = Run.get_context()
data_path = aml_context.input_datasets['train_files']
output_path = aml_context.output_datasets['output_dir']


dataset_list = [
 '/feature_name_by_day_by_peak.parquet',
 '/feature_name_day_by_Braga_peak.parquet',
 '/feature_name_day_by_Kingsgate_peak.parquet',
 '/feature_name_day_by_Ruby Mountain_peak.parquet'
 ]


for data_set in dataset_list:

    # load data
    df = pd.read_parquet(data_path + data_set)

    # horizon
    horizon = 90

    # data freq
    freq = 'D'

    # set n_windows
    n_windows = 5 if data_set == '/feature_name_by_day_by_peak.parquet' else 2

    # drop test
    df = df.groupby('unique_id').apply(lambda x: x.iloc[:-horizon]).reset_index(drop=True)  


    config = {
        "learning_rate": tune.loguniform(1e-6, 5e-2),
        "max_steps": tune.choice([150, 200, 300, 400, 500, 600, 700, 900, 1000, 1200, 1300, 1500]),
        "input_size": tune.choice([60, 70, 80, 90, 100, 110, 120, 150, 182, 189, 196, 203, 210, 217, 224, 231, 238, 245, 252, 259, 266, 273, 280, 287, 294, ]),
        "batch_size": tune.choice([128, 512, 1024, 2056, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384, 17408, ]),
        "windows_batch_size": tune.choice([32, 48, 64, 128, 256, 512, 768]),
        "stride": tune.choice([4, 8, 12, 16]),
        "hidden_size": tune.choice([64, 128, 256, 512]),
        "linear_hidden_size": tune.choice([64, 128, 256, 512]),
        "n_heads": tune.choice([4, 8, 16, ]),
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
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=24)

    # run the hyperparameter tuning  
    analysis = tune.run(  
        tune.with_parameters(tune_obj, df=df, horizon=horizon, freq=freq),  
        num_samples=1000,
        config=config,
        scheduler=scheduler,
        local_dir=output_path,
        checkpoint_config=CheckpointConfig(
                    num_to_keep=1, checkpoint_frequency=0),
        storage_path=output_path,
        resume=False,
        search_alg=search_alg,
        max_concurrent_trials=24,
        progress_reporter=reporter,
        resources_per_trial=tune.PlacementGroupFactory([
                {"GPU": 1},
            ], strategy="SPREAD"),
    )

    # get the best hyperparameters  
    best_config = analysis.get_best_config(metric="mse", mode="min")
    print("Best Config:", best_config)

    # save on lake
    with open(output_path + f'{data_set}.pickle', 'wb') as handle:
        pickle.dump(best_config, handle, protocol=pickle.HIGHEST_PROTOCOL)


ray.shutdown()

#


