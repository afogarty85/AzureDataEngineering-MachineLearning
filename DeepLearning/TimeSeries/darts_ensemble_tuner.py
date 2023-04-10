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
from darts.metrics import mape, smape, mae
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


def configure_callbacks():
    early_stop = EarlyStopping(monitor="train_loss", mode="min", patience=7, min_delta=0.05,)
    return [early_stop]


def train_fn(params):
    '''
    HyperOpt tuning
    '''
    for key, value in params.items():
        try:
            if key not in ['lr', 'factor', 'dropout', 'n_freq_downsample', 'pooling_kernel_sizes']:
                params[key] = int(value)
        except Exception as e:
            params[key] = value

    # load df
    tseries = pd.read_parquet(r'/mnt/c/Users/afogarty/Desktop/darts/tseries_monthly.parquet')

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

    # past cov split
    past_train_cov = [s[:-args.horizon] for s in past_cov]
    past_val_cov = [s[-args.horizon:] for s in past_cov]

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

    # static transformer
    static_transformer = StaticCovariatesTransformer()
    train_set_transformed = static_transformer.fit_transform(train_set_transformed)
    val_set_transformed = static_transformer.transform(val_set_transformed)

    if args.name == 'nhits':

        print('NHiTS Model Triggered!')
        # subset params for kwargs
        mkwargs = {k: v for k, v in params.items() if k in ['add_encoders', 'num_stacks', 'num_layers',
                                                             'num_blocks', 'layer_widths', 'batch_size',
                                                             'input_chunk_length', 'n_freq_downsample',
                                                             'pooling_kernel_sizes', 'optimizer_kwargs']}

    elif args.name == 'nbeatsi':
        print('NBEATS-I Model Triggered!')
        # subset params for kwargs
        mkwargs = {k: v for k, v in params.items() if k in ['add_encoders', 'num_stacks', 'num_layers', 'num_blocks',
                                                            'layer_widths', 'trend_polynomial_degree', 'batch_size',
                                                            'generic_architecture', 'input_chunk_length', 'optimizer_kwargs']}

    elif args.name == 'nbeatsg':
        print('NBEATS-G Model Triggered!')
        # subset params for kwargs
        mkwargs = {k: v for k, v in params.items() if k in ['add_encoders', 'num_stacks', 'num_layers', 'num_blocks',
                                                            'layer_widths', 'batch_size', 'expansion_coefficient_dim',
                                                            'generic_architecture', 'input_chunk_length', 'optimizer_kwargs']}

    elif args.name == 'tft':
        print('TFT Model Triggered!')
        # subset params for kwargs
        mkwargs = {k: v for k, v in params.items() if k in ['add_encoders', 'lstm_layers', 'num_attention_heads',
                                                            'add_relative_index',
                                                             'batch_size', 'input_chunk_length', 'optimizer_kwargs']}


    # init model
    model = params['model'](**mkwargs,
                # fixed params
                output_chunk_length=args.horizon,
                likelihood=None,  # want to optimize against loss fns
                pl_trainer_kwargs={"callbacks": configure_callbacks()},
                lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
                lr_scheduler_kwargs={'monitor': 'train_loss', 'factor': params['factor'], 'patience': params['patience']},
                force_reset=True,
                n_epochs=100,
                random_state=params['random_state'],
        )

    try:
        # fit
        model.fit(series=train_set_transformed,
                past_covariates=past_train_cov_transformed,
                num_loader_workers=12,
                verbose=False)

    except Exception as e:
        print(f'Error: {e}, {params}')

    # get predictions
    preds = model.predict(series=train_set_transformed,
                        past_covariates=past_train_cov_transformed,
                        n=args.horizon,
                        num_samples=1,  # 1 if no likelihood fn, else 500
                        n_jobs=-1,
                        )
    
    # inverse scaler
    preds_inv = target_transformer.inverse_transform(preds)
    
    # MAE on VAL
    sum_val = 0.0
    preds_ = [preds_inv[i].values().flatten() for i in range(len(preds_inv))]
    vals = [val_set[i].values().flatten() for i in range(len(val_set))]
    vals = np.concatenate(vals)
    preds_ = np.concatenate(preds_)
    try:
        sum_val += mean_squared_error(vals, preds_, squared=False)  # if false, RMSE: punish larger deviations
    except Exception as e:
        sum_val += 99999999

    print(f'tried these params: {params.items()} and got this result {sum_val}')

    return {'loss': sum_val, 'status': STATUS_OK}



# set general search params
general_params = {
                #'col_sample': hp.quniform('col_sample', 0, len(col_combos)-1, 1),  # not cycling through features
                'batch_size': hp.quniform('batch_size', 32, 1024, 32),
                #'dropout': hp.uniform('dropout', 1e-2, 1e-1),  # no dropout for ensemble
                 # lr
                'lr': hp.uniform('lr', 1e-5, 1e-2),
                'patience': hp.quniform('patience', 1, 5, 1),
                'factor': hp.uniform('factor', 1e-2, 9e-1),
}

# config map to check/use
config_map = {
            "nhits": {'params': {
                'model': hp.choice('model', [NHiTSModel,]),
                # GFM Specific -- fixed to authors search
                'num_stacks': hp.choice('num_stacks', [3]),
                'num_blocks': hp.choice('num_blocks', [1]),
                'num_layers': hp.choice('num_layers', [2]),
                'layer_widths': hp.choice('layer_widths', [512]),
                # search
                'n_freq_downsample': hp.choice('n_freq_downsample', [ [(168,), (24,), (1,)],
                                                                           [(24,), (12,), (1,)],
                                                                           [(180,), (60,), (1,)],
                                                                           [(40,), (20,), (1,)],
                                                                           [(64,), (8,), (1,)],
                                                                           ]),
                'pooling_kernel_sizes': hp.choice('pooling_kernel_sizes', [ [(2,), (2,), (2,)],
                                                                           [(4,), (4,), (4,)],
                                                                           [(8,), (8,), (8,)],
                                                                           [(8,), (4,), (1,)],
                                                                           [(16,), (8,), (1,)],
                                                                           ]),
                # encoder
                'add_encoders': hp.choice('add_encoders', [None, 
                                                         {"datetime_attribute": {"past": ["year", "month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                                                         {"datetime_attribute": {"past": ["month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                                                         {"datetime_attribute": {"past": ["year"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                                                         ]),

                }
                },

            "nbeatsg": {'params': {
                'model': hp.choice('model', [NBEATSModel,]),
                # GFM Specific
                'num_stacks': hp.quniform('num_stacks', 1, 30, 1),
                'num_blocks': hp.quniform('num_blocks', 1, 4, 1),
                'num_layers': hp.quniform('num_layers', 1, 4, 1),
                'layer_widths': hp.quniform('layer_widths', 512, 768, 128),
                # NBEATS-G Specific
                'generic_architecture': hp.choice('generic_architecture', [True]),
                'expansion_coefficient_dim': hp.quniform('expansion_coefficient_dim', 1, 15, 1),  # only if generic=True                
                # encoder
                'add_encoders': hp.choice('add_encoders', [None, 
                                                         {"datetime_attribute": {"past": ["year", "month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                                                         {"datetime_attribute": {"past": ["month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                                                         {"datetime_attribute": {"past": ["year"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                                                         ]),
                }
                },    
                
            "nbeatsi": {'params': {
                'model': hp.choice('model', [NBEATSModel,]),
                # GFM Specific
                'num_stacks': hp.quniform('num_stacks', 1, 24, 2),
                'num_blocks': hp.quniform('num_blocks', 1, 8, 1),
                'num_layers': hp.quniform('num_layers', 1, 8, 1),
                'layer_widths': hp.quniform('layer_widths', 128, 1024, 32),
                # NBEATS-I Specific
                'generic_architecture': hp.choice('generic_architecture', [False]),
                'trend_polynomial_degree': hp.quniform('trend_polynomial_degree', 1, 8, 1),  # only if generic=False
                # encoder
                'add_encoders': hp.choice('add_encoders', [None, 
                                                         {"datetime_attribute": {"past": ["year", "month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                                                         {"datetime_attribute": {"past": ["month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                                                         {"datetime_attribute": {"past": ["year"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                                                         ]),
                }
                },

            "tft": {'params': {
                'model': hp.choice('model', [TFTModel,]),
                 # TFT Specific
                'hidden_size': hp.quniform('hidden_size', 64, 768, 32),
                'lstm_layers': hp.quniform('lstm_layers', 1, 4, 1),
                'num_attention_heads': hp.quniform('num_attention_heads', 1, 4, 1),
                'dropout': hp.uniform('dropout', 1e-2, 1e-1),
                'add_relative_index': hp.choice('add_relative_index', [True]),
                # encoder
                'add_encoders': hp.choice('add_encoders', [None, 
                                                         {"datetime_attribute": {"past": ["year", "month"]},
                                                          "datetime_attribute": {"future": ["year", "month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},

                                                         {"datetime_attribute": {"past": ["month"]},
                                                          "datetime_attribute": {"future": ["month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},

                                                         {"datetime_attribute": {"past": ["year"]},
                                                          "datetime_attribute": {"future": ["year"]}, "transformer": Scaler(scaler=MaxAbsScaler())},

                                                         {"datetime_attribute": {"past": ["year", "month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                                                         {"datetime_attribute": {"past": ["month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                                                         {"datetime_attribute": {"past": ["year"]}, "transformer": Scaler(scaler=MaxAbsScaler())},                                                         
                                                         ]),
                }
                },                
            'all': {'params': None}
}

# tell the CLI user that they mistyped the model
if args.name not in config_map:
    raise ValueError("Unrecognized config")


# result storage
argmin_storage = {}
counter = 0


print('Starting trials!')
for i in range(0, args.n_models):

    # init loss functions to optimize against
    for loss_ in [MAELoss(), SmapeLoss(), MapeLoss()]:

        # init a set of input lengths to use;
        for input_length_ in [2, 3, 4, 5, 6]:

            # random state for trial for ensemble variance
            random_state = np.random.randint(low=0, high=100000)

            # combine params
            params = {**config_map[args.name]['params'], **general_params}

            # add on params
            new_params = {'loss_fn': loss_,
                          'input_chunk_length': input_length_,
                          'optimizer_kwargs': {'lr': general_params['lr']}, 
                          'random_state': random_state
                          }
            
            # combine
            params = {**params, **new_params}

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
            argmin_storage[f'{args.name}' + str(counter)]['misc']['vals']['input_chunk_length'] = input_length_
            argmin_storage[f'{args.name}' + str(counter)]['misc']['vals']['loss_fn'] = loss_
            argmin_storage[f'{args.name}' + str(counter)]['misc']['vals']['random_state'] = random_state

            # report to CLI
            print(f'the argmin is: {argmin}')

# save the trials object
with open(f"hyperopt_trials_{args.name}", "wb") as f:
    pickle.dump(trials, f)

# save the dict
with open(f"argmin_{args.name}", "wb") as f:
    pickle.dump(argmin_storage, f)

print(f'Total training complete! All argmins: {argmin_storage}')
