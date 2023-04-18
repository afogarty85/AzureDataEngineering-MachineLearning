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
from scipy.special import boxcox1p, inv_boxcox
from sklearn.metrics import mean_absolute_percentage_error
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})


def configure_callbacks():
    early_stop = EarlyStopping(monitor="train_loss", mode="min", patience=5, min_delta=0.05,)
    return [early_stop]


def trainer(series, target_transformer, horizon, past_covariates=None,  num_samples=500 ):

    # fit
    ensemble_model.fit(series=series,
            past_covariates=past_covariates,)

    # get predictions
    preds = ensemble_model.predict(series=series,
                        past_covariates=past_covariates,
                        n=horizon,
                        num_samples=num_samples,
                        )
    return preds



def ensemble_runner(tseries, argmin_paths, horizon, bagging=True, filter_argmins=False, filter_score=None):
    '''
    params:
        tseries; dataframe -- the dataframe of timeseries
        params; list -- paths to the argmin output(s) derived from ensemble_tuner.py
        horizon; int -- timesteps to predict
        col_sample; int -- set of features to use

    returns:
        model_list -- list of models for ensembling
    '''

    # add 1 so we can use mape loss if so
    if (tseries['featureNamePeak'] == 0.0).any():
        tseries['featureNamePeak'] = tseries['featureNamePeak'] + 1

    # load argmin data and combine
    argmin = {}

    for apath in argmin_paths:
        t_arg = pd.read_pickle(apath)
        # combine
        argmin = {**argmin, **t_arg}

    # paste all kwargs
    config_map = {'add_encoders': [None, 
                    {"datetime_attribute": {"past": ["year", "month"]},
                    "datetime_attribute": {"future": ["year", "month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},

                    {"datetime_attribute": {"past": ["month"]},
                    "datetime_attribute": {"future": ["month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},

                    {"datetime_attribute": {"past": ["year"]},
                    "datetime_attribute": {"future": ["year"]}, "transformer": Scaler(scaler=MaxAbsScaler())},

                    {"datetime_attribute": {"past": ["year", "month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                    {"datetime_attribute": {"past": ["month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                    {"datetime_attribute": {"past": ["year"]}, "transformer": Scaler(scaler=MaxAbsScaler())},


                    ],

                    'pooling_kernel_sizes': [ [(2,), (2,), (2,)],
                                                                            [(4,), (4,), (4,)],
                                                                            [(8,), (8,), (8,)],
                                                                            [(8,), (4,), (1,)],
                                                                            [(16,), (8,), (1,)],
                                                                            ],
                    'n_freq_downsample':  [ [(168,), (24,), (1,)],
                                                                            [(24,), (12,), (1,)],
                                                                            [(180,), (60,), (1,)],
                                                                            [(40,), (20,), (1,)],
                                                                            [(64,), (8,), (1,)],
                                                                            ],
                    'num_stacks': 3,
                    'num_layers': 2,
                    'num_blocks': 1,
                    'layer_widths': 512,
                    'add_relative_index': [True],
                    'nhits': NHiTSModel,
                    'nbeatsg': NBEATSModel,
                    'nbeatsi': NBEATSModel,
                    'tft': TFTModel,

                    }

    mkwargs = {}
    mparams = {}

    mkwarg_list = ['add_encoders', 'num_stacks', 'num_layers',
                'num_blocks', 'layer_widths', 'batch_size',
                'input_chunk_length', 'n_freq_downsample',
                'pooling_kernel_sizes', 'random_state', 'loss_fn', 'hidden_size',
                'optimizer_kwargs', 'lstm_layers', 'num_attention_heads', 'add_relative_index',]

    mparam_list = ['factor', 'patience', 'model_name']


    # unpack argmin dict
    for k, v in argmin.items():
        if filter_argmins and v['result']['loss'] >= filter_score:
            continue
        else:
            t_vals = v['misc']['vals']
            # turn to all scalar
            t_vals = {k: int(v[0])
                        if k not in ['input_chunk_length', 'loss_fn', 'optimizer_kwargs', 'random_state', 'model_name'] else 
                        v
                        for k, v in t_vals.items() 
                    }
            # add LR
            t_vals['optimizer_kwargs'] = {'lr': v['misc']['vals']['lr'][0]}
            t_vals = {k: v for k, v in t_vals.items() if k in mkwarg_list}
            mkwargs[k] = t_vals

    # replace vals 
    for k, v in mkwargs.items():
        for k_ in v:
            if k_ not in config_map.keys():
                continue
            elif k_ in ['pooling_kernel_sizes', 'n_freq_downsample', 'add_encoders', 'model_name']:
                mkwargs[k][k_] = config_map[k_][mkwargs[k][k_]]
            else:
                mkwargs[k][k_] = config_map[k_]


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

    # build n-models
    model_list = [
        mparams[k]['model_name'](**v,
                    # fixed params
                    output_chunk_length=horizon,
                    #likelihood=None,
                    likelihood=QuantileRegression(quantiles=[0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]),
                    pl_trainer_kwargs={"callbacks": configure_callbacks()},
                    lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    lr_scheduler_kwargs={'monitor': 'train_loss', 'factor': mparams[k]['factor'], 'patience': mparams[k]['patience']},
                    force_reset=True,
                    save_checkpoints=False,
                    n_epochs=100,
            ) for k, v in mkwargs.items()
            ]
    
    # bag with diff inits
    for k, v in mkwargs.items():
        t_vals = v
        # change seed
        t_vals['random_state'] = np.random.randint(low=1, high=1000)
        mkwargs[k] = t_vals

    bagging_list = [
        mparams[k]['model_name'](**v,
                    # fixed params
                    output_chunk_length=horizon,
                    #likelihood=None,
                    likelihood=QuantileRegression(quantiles=[0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]),
                    pl_trainer_kwargs={"callbacks": configure_callbacks()},
                    lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    lr_scheduler_kwargs={'monitor': 'train_loss', 'factor': mparams[k]['factor'], 'patience': mparams[k]['patience']},
                    force_reset=True,
                    save_checkpoints=False,
                    n_epochs=100,
            ) for k, v in mkwargs.items()
            ]
    
    if bagging:     
        # combine
        model_list = model_list + bagging_list

    print(f'Generated {len(model_list)} models')

    return model_list



# load df
tseries = pd.read_parquet(r'/mnt/c/Users/afogarty/Desktop/darts/tseries_monthly_important.parquet')

# set some choices
col_sample = 94  # top performers; 118, 13, 21, 80, 96, 94
horizon = 6

# available features to iterate over
cov_choices = ["uniqueFYQMilestones", "uniqueUsers", "uniqueProjects",
                'daysUntilMilestoneEnd', 'prevFeatureNamePeak', 'prevThreeFeatureNamePeak',
                'daysUntilNextMilestoneStart']

# filter cols
col_combos = list(itertools.chain(*[itertools.combinations(cov_choices, i+1) for i in range(len(cov_choices))]))

# build time series df
series_multi = TimeSeries.from_group_dataframe(
    tseries,
    time_col="Date",  # month end
    group_cols="featureName",  # individual time series are extracted by grouping `df` by `group_cols`
    value_cols=["featureNamePeak"],
    freq='M',
    fill_missing_dates=True
)

# set f32 for way faster training
series_multi = [s.astype(np.float32) for s in tqdm(series_multi)]

# create other past covariates; 
past_cov = TimeSeries.from_group_dataframe(
    tseries,
    time_col="Date",  # month end
    group_cols="featureName",  # individual time series are extracted by grouping `df` by `group_cols`
    value_cols=list(col_combos[col_sample]),
    freq='M',
    fill_missing_dates=True
)

# set f32 for way faster training
past_cov = [s.astype(np.float32) for s in tqdm(past_cov)]

# Set aside n-time as a validation series
train_set = [s[:-horizon] for s in series_multi]
val_set = [s[-horizon:] for s in series_multi]

# past cov split
past_train_cov = [s[:-horizon] for s in past_cov]
past_val_cov = [s[-horizon:] for s in past_cov]

# normalize 
target_transformer = Scaler(scaler=MaxAbsScaler())
# fit transform
train_set_transformed = target_transformer.fit_transform(train_set)
# transform
val_set_transformed = target_transformer.transform(val_set)

# transform past_cov
past_cov_scaler = Scaler(scaler=MaxAbsScaler())
# fit transform
past_train_cov_transformed = past_cov_scaler.fit_transform(past_train_cov)
# transform
past_val_cov_transformed = past_cov_scaler.transform(past_val_cov)

# static trnsformer
static_transformer = StaticCovariatesTransformer()
train_set_transformed = static_transformer.fit_transform(train_set_transformed)
val_set_transformed = static_transformer.transform(val_set_transformed)

# get feature order
featureOrder = np.concatenate([train_set[i].static_covariates.values[0] for i in range(len(train_set))])


# pickle paths
#model_name = 'nbeats'
argmin_paths = [
                #'/mnt/c/Users/afogarty/Desktop/darts/argmin_nbeatsi',
                #'/mnt/c/Users/afogarty/Desktop/darts/argmin_nbeatsg',
                '/mnt/c/Users/afogarty/Desktop/darts/argmin_tft',
                #'/mnt/c/Users/afogarty/Desktop/darts/argmin_nhits'
                ]

# generate models
model_list = ensemble_runner(
                             tseries=tseries,
                             argmin_paths=argmin_paths,
                             horizon=horizon,
                             bagging=True,
                             filter_argmins=False,
                             filter_score=155)

# init ensemble model
ensemble_model = RegressionEnsembleModel(forecasting_models=model_list,
                                        regression_train_n_points=6  # number of time steps to train the meta model
                                        )

# generate predictions
preds = trainer(series=train_set_transformed,
                    target_transformer=target_transformer,
                    horizon=horizon,
                    past_covariates=past_train_cov_transformed,
                    num_samples=500)  # 1 if no likelihood

# inverse scale
preds_inv = target_transformer.inverse_transform(preds)    

# getsmapes
smapes = smape(actual_series=val_set, pred_series=preds_inv)
print(f'the median smape for all series: {np.median(smapes)}')
print(f'the average smape for all series: {np.mean(smapes)}')

# check group loc
tseries_scalar = list(featureOrder).index('msimhdlsim')  # msimhdlsim

# MAPE/SMAPE for caliredrc
print("MAPE: {:.2f}%".format(mape(val_set[tseries_scalar], preds_inv[tseries_scalar])))  # 26
print("SMAPE: {:.2f}%".format(smape(val_set[tseries_scalar], preds_inv[tseries_scalar])))  # 31
print("MASE: {:.2f}%".format(mase(val_set[tseries_scalar], preds_inv[tseries_scalar], insample=train_set[tseries_scalar])) )    # 2.99

# plot
fig, axs = plt.subplots(1,  figsize=(6, 6))
val_set[tseries_scalar].plot(ax=axs, label='true')
preds_inv[tseries_scalar].plot(ax=axs, low_quantile=0.5, high_quantile=0.95, label='pred')



def generate_data(col_sample, horizon):

    # build time series df
    series_multi = TimeSeries.from_group_dataframe(
        tseries,
        time_col="Date",  # month end
        group_cols="featureName",  # individual time series are extracted by grouping `df` by `group_cols`
        value_cols=["featureNamePeak"],
        freq='M',
        fill_missing_dates=True
    )

    # set f32 for way faster training
    series_multi = [s.astype(np.float32) for s in tqdm(series_multi)]

    # create other past covariates; 
    past_cov = TimeSeries.from_group_dataframe(
        tseries,
        time_col="Date",  # month end
        group_cols="featureName",  # individual time series are extracted by grouping `df` by `group_cols`
        value_cols=list(col_combos[col_sample]),
        freq='M',
        fill_missing_dates=True
    )

    # set f32 for way faster training
    past_cov = [s.astype(np.float32) for s in tqdm(past_cov)]

    # Set aside n-time as a validation series
    train_set = [s[:-horizon] for s in series_multi]
    val_set = [s[-horizon:] for s in series_multi]

    # past cov split
    past_train_cov = [s[:-horizon] for s in past_cov]
    past_val_cov = [s[-horizon:] for s in past_cov]

    # normalize 
    target_transformer = Scaler(scaler=MaxAbsScaler())
    # fit transform
    train_set_transformed = target_transformer.fit_transform(train_set)
    # transform
    val_set_transformed = target_transformer.transform(val_set)

    # transform past_cov
    past_cov_scaler = Scaler(scaler=MaxAbsScaler())
    # fit transform
    past_train_cov_transformed = past_cov_scaler.fit_transform(past_train_cov)
    # transform
    past_val_cov_transformed = past_cov_scaler.transform(past_val_cov)

    # static trnsformer
    static_transformer = StaticCovariatesTransformer()
    train_set_transformed = static_transformer.fit_transform(train_set_transformed)
    val_set_transformed = static_transformer.transform(val_set_transformed)

    # get feature order
    featureOrder = np.concatenate([train_set[i].static_covariates.values[0] for i in range(len(train_set))])

    return train_set_transformed, val_set_transformed, past_train_cov_transformed, past_val_cov_transformed, target_transformer, featureOrder


# gen data
(train_set_transformed, 
    val_set_transformed, 
    past_train_cov_transformed,
    past_val_cov_transformed,
    target_transformer,
    featureOrder
    ) = generate_data(col_sample=col_sample, horizon=horizon)
