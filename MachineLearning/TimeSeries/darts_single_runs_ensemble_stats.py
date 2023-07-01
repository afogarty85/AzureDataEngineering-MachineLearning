import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from torch import nn
from typing import List, Tuple, Dict
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts import concatenate
from darts import TimeSeries
from darts.utils.losses import SmapeLoss, MAELoss
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, smape, mae, mase, rmse
from darts.utils.utils import SeasonalityMode, TrendMode, ModelMode
from darts.models import *
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
torch.backends.cuda.matmul.allow_tf32 = True
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import itertools
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, mean_pinball_loss
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})

# load df
tseries = pd.read_parquet(r'/mnt/c/Users/afogarty/Desktop/darts/tseries_monthly_important.parquet')

# add 1 so we can use mape loss if so
if (tseries['featureNamePeak'] == 0.0).any():
    tseries['featureNamePeak'] = tseries['featureNamePeak'] + 1

horizon = 6

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

# Set aside n-time as a validation series
train_set = [s[:-horizon] for s in series_multi]
val_set = [s[-horizon:] for s in series_multi]


argmin_paths = [
                # '/mnt/c/Users/afogarty/Desktop/darts/argmin_arima',
                # '/mnt/c/Users/afogarty/Desktop/darts/argmin_theta',
                # '/mnt/c/Users/afogarty/Desktop/darts/argmin_croston',  # no quantile support
                '/mnt/c/Users/afogarty/Desktop/darts/argmin_ets',
                # '/mnt/c/Users/afogarty/Desktop/darts/argmin_ces',  # no quantile support
                ]

# load argmin data and combine
argmin = {}

for apath in argmin_paths:
    t_arg = pd.read_pickle(apath)
    # combine
    argmin = {**argmin, **t_arg}

# if I  just want the best across all argmins
best_models = {}
for k, v in argmin.items():
    uni_idx = v['misc']['vals']['univariate_idx']
    curr_loss = v['result']['loss']
    curr_model = v['misc']['vals']['model_name']

    if uni_idx not in best_models.keys():
        best_models[uni_idx] = {'model': None, 'loss': 99999}

    if curr_loss < best_models[uni_idx]['loss']:
        best_models[uni_idx]['loss'] = curr_loss
        best_models[uni_idx]['model'] = k

# swap key with value for filtering
best_models = {v['model']: k for k, v in best_models.items()}

# filter argmin to just get the best model for each series
best_models_argmin = {k:v for k,v in argmin.items() if k in best_models.keys()}

# reorder
best_models_argmin = {key: argmin[key] for key in best_models.keys()}

featureOrder = np.concatenate([train_set[i].static_covariates.values[0] for i in range(len(train_set))])

# static trnsformer
static_transformer = StaticCovariatesTransformer()
train_set = static_transformer.fit_transform(train_set)

def generate_mkwargs(argmin):
    '''
    Take in argmin dict and return kwargs needed to run/generate models
    '''

    mkwargs = {}
    mparams = {}

    mkwarg_list = ['season_length', 'decomposition_type', 'version', 'alpha_d', 'alpha_p',
                'd', 'D', 'max_p', 'max_P', 'max_Q', 'max_d', 'max_D', 'start_p', 'start_P', 'start_Q',
                'stationary', 'seasonal']
    mparam_list = ['model_name']


    config_map = {'decomposition_type': ['multiplicative', 'additive'],
                'version': ['classic', 'optimized', 'sba', 'tsb'],
                'stationary': [False, True],
                'seasonal': [False, True],
                'theta': StatsForecastAutoTheta,
                'croston': Croston,
                'ets': StatsForecastAutoETS,
                'ces': StatsForecastAutoCES,
                'arima': StatsForecastAutoARIMA,
                    }


    # unpack argmin dict
    for k, v in argmin.items():
        t_vals = v['misc']['vals']
        # turn to all scalar
        t_vals = {k: int(v[0])
                    if k in ['season_length', 'alpha_d', 'alpha_p''alpha_d', 'alpha_p',
                'd', 'D', 'max_p', 'max_P', 'max_Q', 'max_d', 'max_D', 'start_p', 'start_P', 'start_Q'] else v
                    for k, v in t_vals.items() 
                }
        # filter
        t_vals = {k: v for k, v in t_vals.items() if k in mkwarg_list}
        
        mkwargs[k] = t_vals

    for k, v, in mkwargs.items():
        t_vals = v
        for tk, tv in t_vals.items():
            if tk in ['decomposition_type', 'version', 'stationary', 'seasonal']:
                t_vals[tk] = config_map[tk][t_vals[tk][0]]
        mkwargs[k] = t_vals


    # unpack argmin dict for mparams
    for k, v in argmin.items():
        t_vals = v['misc']['vals']
        t_vals = {k: v for k, v in t_vals.items() if k in mparam_list}
        # turn to all scalar
        t_vals = {k: v[0] if k not in ['model_name'] else v  for k, v in t_vals.items() }
        mparams[k] = t_vals

    # swap mparams
    for k, v in mparams.items():
        t_vals = v
        t_vals['model_name'] = config_map[t_vals['model_name']]
        mparams[k] = t_vals

    return mkwargs, mparams




#### Train using best model tagged to each series

mkwargs, mparams = generate_mkwargs(best_models_argmin)


# build n-models
model_list = [
    mparams[k]['model_name'](**v,
        ) for k, v in mkwargs.items() 
        ]

len(model_list)


# gen scaler
scaler = Scaler(BoxCoxTransformer(method='guerrero', sp=2))

smape_results = []
total_na = 0
pred_storage = []
# train on train + val
# predict on test

for i, m in enumerate(model_list):

    # get boxcox
    train_set_transformed = scaler.fit_transform(train_set[i])
    
    # fit on train + val
    m.fit(series=train_set_transformed)
    
    # get preds
    if m._is_probabilistic():
        preds = m.predict(horizon, num_samples=1000)
    else:
        preds = m.predict(horizon, num_samples=1)

    # inverse transform preds
    preds_inv = scaler.inverse_transform(preds)

    # get smape
    smape_ = smape(val_set[i], preds_inv)

    if pd.isnull(smape_):
        total_na += 1
    else:
        smape_results.append(smape_)

    # send to storage
    pred_storage.append(preds_inv)


# results
print(np.median(smape_results))  # 32
print(np.mean(smape_results))  # 41

# with boxcox:
#median: 40
#mean: 49


pd.DataFrame({'model': ['arima', 'theta', 'croston', 'ets', 'ces', 'best_available'],
              'median': [36.9, 33.5, 68.55, 34.43, 32.68, 47.46],
              'mean': [47.8, 41.99, 77.6, 45.47, 38.88, 60.82],
              'nan_series': [8, 0, 0, 0, 0, 0]}).sort_values(by=['median'], ascending=True)


# plot normally
plot_scalar = 8
fig, axs = plt.subplots(1,  figsize=(6, 6))
val_set[plot_scalar].plot(ax=axs, label='true')
pred_storage[plot_scalar].plot(ax=axs, low_quantile=0.5, high_quantile=0.85, label='pred')



#### Train each series using all models as an ensemble

smape_results = []
pinball_loss_results = []
# get ensemble of models for each tseries
for i in range(0, 49):
    # set dict
    series_argmin = {}

    # get models for series i
    for k, v, in argmin.items():
        if v['misc']['vals']['univariate_idx'] == i:
            series_argmin[k] = v

    # get mkwargs
    mkwargs, mparams = generate_mkwargs(series_argmin)

    # build n-models
    model_list = [
        mparams[k]['model_name'](**v,
            ) for k, v in mkwargs.items() 
            ]

    # gen model
    ensemble_model = RegressionEnsembleModel(forecasting_models=model_list,
                                         regression_train_n_points=2  # number of time steps to train the meta model
                                         )

    # get boxcox
    train_set_transformed = scaler.fit_transform(train_set[i])
    
    # fit on train + val
    ensemble_model.fit(series=train_set_transformed)
    
    # get preds
    if ensemble_model._is_probabilistic():
        preds = ensemble_model.predict(horizon, num_samples=1000)
    else:
        preds = ensemble_model.predict(horizon, num_samples=1)

    # inverse transform preds
    preds_inv = scaler.inverse_transform(preds)

    # get smape
    smape_ = smape(val_set[i], preds_inv)

    if pd.isnull(smape_):
        total_na += 1
    else:
        smape_results.append(smape_)

    # send to storage
    pred_storage.append(preds_inv)

    # gen pinball loss and store
    try:
        pinball_loss_ = mean_pinball_loss(val_set[i].values(), preds_inv.values(), alpha=0.95)
        pinball_loss_results.append(pinball_loss_)
    except Exception as e:
        print(i, e)
        


# ensemble results
print(np.median(smape_results))  # 55 @ 2 timesteps
print(np.mean(smape_results))  # 73
print(np.mean(pinball_loss_results))  # 23

