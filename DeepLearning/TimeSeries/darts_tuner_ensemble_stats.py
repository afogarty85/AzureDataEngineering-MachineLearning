import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import EarlyStopping
from darts import TimeSeries
from sklearn.metrics import mean_absolute_error, mean_squared_error
from darts.utils.losses import SmapeLoss, MAELoss, MapeLoss
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, smape, mae, rmse
from darts.models import *
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
import itertools
from argparse import ArgumentParser
import pickle
import datetime
torch.backends.cuda.matmul.allow_tf32 = True


# set parser requirements
parser = ArgumentParser(description="HyperOpt Tuning")
parser.add_argument(
    "--name",
    required=True,
    type=str,
    default='all',
    help="Name of model to run",
)
parser.add_argument(
    "--horizon",
    required=True,
    default=6,
    type=int,
    help="Timesteps to model",
)
parser.add_argument(
    "--n_series",
    required=True,
    default=49,
    type=int,
    help="Number of univariate series to record against",
)
parser.add_argument(
    "--n_models",
    required=True,
    type=int,
    help="Number of models to generate",
)
# initialize arg parser
args = parser.parse_args()

# available features to iterate over
cov_choices = ["uniqueFYQMilestones", "uniqueUsers", "uniqueProjects",
                'daysUntilMilestoneEnd', 'prevFeatureNamePeak', 'prevThreeFeatureNamePeak',
                'daysUntilNextMilestoneStart']

# filter cols
col_combos = list(itertools.chain(*[itertools.combinations(cov_choices, i+1) for i in range(len(cov_choices))]))

# horizon time steps
print(f'Established Horizon: {args.horizon} and Model: {args.name}')


def train_fn(params):
    '''
    HyperOpt tuning
    '''
    for key, value in params.items():
        try:
            if key not in ['decomposition_type', 'alpha_d', 'alpha_p']:
                params[key] = int(value)
        except Exception as e:
            params[key] = value

    # load df
    tseries = pd.read_parquet(r'/mnt/c/Users/afogarty/Desktop/darts/tseries_monthly_important.parquet')

    # drop test set
    tseries = tseries.drop(tseries.groupby(['featureName']).tail(args.horizon).index, axis=0).reset_index(drop=True)

    # add 1 so we can use mape loss if so
    if (tseries['featureNamePeak'] == 0.0).any():
        tseries['featureNamePeak'] = tseries['featureNamePeak'] + 1

    # build time series df
    series_multi = TimeSeries.from_group_dataframe(
        tseries,
        time_col="Date",
        group_cols="featureName",
        value_cols=["featureNamePeak"],
        freq='M',
        fill_missing_dates=True
    )

    # set f32 for way faster training
    series_multi = [s.astype(np.float32) for s in tqdm(series_multi)]

    # create other past covariates; 
    past_cov = TimeSeries.from_group_dataframe(
        tseries,
        time_col="Date",
        group_cols="featureName",
        value_cols=list(col_combos[118]),
        freq='M',
        fill_missing_dates=True
    )

    # set f32 for way faster training
    past_cov = [s.astype(np.float32) for s in tqdm(past_cov)]

    # Set aside n-time as a validation series
    train_set = [s[:-args.horizon] for s in series_multi]
    val_set = [s[-args.horizon:] for s in series_multi]

    if args.name == 'arima':

        print('AutoArima Model Triggered!')
        # subset params for kwargs
        mkwargs = {k: v for k, v in params.items() if k in ['d', 'D', 'max_p', 'max_q', 'max_P', 'max_Q', 
                                                            'max_d', 'max_D', 'start_p', 'start_q', 'start_P', 'start_Q',
                                                            'stationary', 'seasonal']
                                                            }

    elif args.name == 'ets':
        print('AutoETS Model Triggered!')
        # subset params for kwargs
        mkwargs = {k: v for k, v in params.items() if k in ['season_length', 'model']}

    elif args.name == 'ces':
        print('AutoCES Model Triggered!')
        # subset params for kwargs
        mkwargs = {k: v for k, v in params.items() if k in ['season_length', 'model']}

    elif args.name == 'theta':
        print('AutoTheta Model Triggered!')
        # subset params for kwargs
        mkwargs = {k: v for k, v in params.items() if k in ['season_length', 'decomposition_type']}

    elif args.name == 'croston':
        print('Croston Model Triggered!')
        # subset params for kwargs
        mkwargs = {k: v for k, v in params.items() if k in ['version', 'alpha_d', 'alpha_p', ]}


    # init model
    model = params['model_name'](**mkwargs,
                            )
    
    # unpack univariate series
    univariate_series = train_set[params['univariate_idx']]
    univariate_val_series = val_set[params['univariate_idx']]

    try:
        # fit
        model.fit(series=univariate_series)

        # get predictions
        preds = model.predict(n=args.horizon,
                            num_samples=1,  # 1 if no likelihood fn, else 500
                            )

    except Exception as e:
        print(f'Error: {e}, {params}')

    
    # RMSE on VAL
    sum_val = 0.0

    try:
        sum_val += mean_squared_error(univariate_val_series.values().flatten(),
                                      preds.values().flatten(),
                                      squared=False)  # if false, RMSE: punish larger deviations
    except Exception as e:
        sum_val += 99999999

    print(f'tried these params: {params.items()} and got this result {sum_val}')

    return {'loss': sum_val, 'status': STATUS_OK}




# config map to check/use
config_map = {
            "theta": {'params': {
                'model_name': hp.choice('model_name', [StatsForecastAutoTheta,]),
                'season_length': hp.choice('season_length', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
                'decomposition_type': hp.choice('decomposition_type', ['multiplicative', 'additive'])
                }
                },

            "arima": {'params': {
                'model_name': hp.choice('model_name', [StatsForecastAutoARIMA,]),
                'd': hp.choice('d', [None, 1, 2, 3, ]),
                'D': hp.choice('D', [None, 1, 2, 3, ]),
                'max_p': hp.quniform('max_p', 0, 10, 1),
                'max_P': hp.quniform('max_P', 0, 10, 1),
                'max_Q': hp.quniform('max_Q', 0, 10, 1),
                'max_d': hp.quniform('max_d', 0, 10, 1),
                'max_D': hp.quniform('max_D', 0, 10, 1),
                'start_p': hp.quniform('start_p', 0, 10, 1),
                'start_q': hp.quniform('start_q', 0, 10, 1),
                'start_Q': hp.quniform('start_Q', 0, 10, 1),
                'start_P': hp.quniform('start_P', 0, 10, 1),
                'stationary': hp.choice('stationary', [False, True]),
                'seasonal': hp.choice('seasonal', [False, True]),
                }
                },


            "croston": {'params': {
                            'model_name': hp.choice('model_name', [Croston,]),
                            'version': hp.choice('version', ['classic', 'optimized', 'sba', 'tsb']),
                            'alpha_d': hp.uniform('alpha_d', high=1, low=0),
                            'alpha_p': hp.uniform('alpha_p', high=1, low=0),
                            }
                            },

            "ces": {'params': {
                        'model_name': hp.choice('model_name', [StatsForecastAutoCES,]),
                        'season_length': hp.choice('season_length', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
                        'model': hp.choice('model', ['Z',]),
                        }
                        },                            

            "ets": {'params': {
                        'model_name': hp.choice('model_name', [StatsForecastAutoETS,]),
                        'season_length': hp.choice('season_length', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
                        'model': hp.choice('model', ['ZZZ',]),
                        }
                        },       

}

# tell the CLI user that they mistyped the model
if args.name not in config_map:
    raise ValueError("Unrecognized config")


# result storage
argmin_storage = {}
counter = 0



print('Starting trials!')
for i in range(0, args.n_models):

    # init a set of input lengths to use;
    for univariate_idx in range(0, args.n_series):
        # combine params
        params = {**config_map[args.name]['params'], **{'univariate_idx': univariate_idx}}

        # running these params:
        print(f'Running these params: {params}')

        # run
        trials = Trials()
        argmin = fmin(fn=train_fn,
                        space=params,
                        algo=tpe.suggest,
                        max_evals=150,
                        early_stop_fn=no_progress_loss(iteration_stop_count=35,
                                                        percent_increase=1.0),
                        trials=trials
                        )
        
        counter += 1
        
        # store argmin
        argmin_storage[f'{args.name}' + str(counter)] = trials.best_trial

        # update argmin_dict with aux params
        argmin_storage[f'{args.name}' + str(counter)]['misc']['vals']['model_name'] = args.name
        argmin_storage[f'{args.name}' + str(counter)]['misc']['vals']['univariate_idx'] = univariate_idx

        # report to CLI
        print(f'the argmin is: {argmin}')

# save the dict
with open(f"argmin_{args.name}", "wb") as f:
    pickle.dump(argmin_storage, f)

print(f'Total training complete! All argmins: {argmin_storage}')
