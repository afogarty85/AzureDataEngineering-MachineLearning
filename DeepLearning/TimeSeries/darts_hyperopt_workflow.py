import random
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
from darts.utils.losses import SmapeLoss, MAELoss
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, smape, mae
from darts.models import *
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
import itertools
from argparse import ArgumentParser
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
# initialize arg parser
args = parser.parse_args()

# seed
random.seed(0)
np.random.seed(0)

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
            if key not in ['learning_rate', 'factor', 'dropout']:
                params[key] = int(value)
        except Exception as e:
            params[key] = value

    # load df
    tseries = pd.read_parquet(r'/mnt/c/Users/afogarty/Desktop/darts/tseries_monthly.parquet')

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
        value_cols=list(col_combos[params['col_sample']]),
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
                                                             'num_blocks', 'layer_widths', 'batch_size', 'input_chunk_length']}

    elif args.name == 'nbeats':
        print('NBEATS Model Triggered!')
        # subset params for kwargs
        mkwargs = {k: v for k, v in params.items() if k in ['add_encoders', 'num_stacks', 'num_layers', 'num_blocks',
                                                            'layer_widths', 'trend_polynomial_degree', 'batch_size',
                                                            'generic_architecture', 'input_chunk_length']}

    elif args.name == 'nbeatsg':
        print('NBEATSG Model Triggered!')
        # subset params for kwargs
        mkwargs = {k: v for k, v in params.items() if k in ['add_encoders', 'num_stacks', 'num_layers', 'num_blocks',
                                                            'layer_widths', 'batch_size', 'expansion_coefficient_dim',
                                                            'generic_architecture', 'input_chunk_length']}

    elif args.name == 'tft':
        print('TFT Model Triggered!')
        # subset params for kwargs
        mkwargs = {k: v for k, v in params.items() if k in ['add_encoders', 'lstm_layers', 'num_attention_heads', 'add_relative_index',
                                                            'batch_size', 'input_chunk_length']}


    # init model
    model = params['model'](**mkwargs,
                # fixed params
                output_chunk_length=args.horizon,
                loss_fn=MAELoss(),  # only used if Likelihood = None
                likelihood=QuantileRegression(quantiles=[0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]),
                pl_trainer_kwargs={"callbacks": configure_callbacks()},
                lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
                lr_scheduler_kwargs={'monitor': 'train_loss', 'factor': params['factor'], 'patience': params['patience']},
                force_reset=True,
                n_epochs=100,
                random_state=0,
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
                        num_samples=500,
                        n_jobs=-1,
                        )
    
    # smapes on val
    smapes = smape(val_set_transformed, preds)

    sum_val = 0.0
    sum_val += np.mean(smapes)

    print(f'tried these params: {params.items()} and got this result {sum_val}')

    return {'loss': sum_val, 'status': STATUS_OK}



# set general search params
general_params = {
                'col_sample': hp.quniform('col_sample', 0, len(col_combos)-1, 1),
                'batch_size': hp.quniform('batch_size', 32, 1024, 32),
                'dropout': hp.uniform('dropout', 1e-2, 1e-1),
                 # lr
                'learning_rate': hp.uniform('learning_rate', 1e-5, 1e-2),
                'patience': hp.quniform('patience', 1, 5, 1),
                'factor': hp.uniform('factor', 1e-2, 9e-1),
                'input_chunk_length': hp.quniform('input_chunk_length', 1, args.horizon*1.2, 1),
}

# config map to check/use
config_map = {
            "nhits": {'params': {
                'model': hp.choice('model', [NHiTSModel,]),
                # GFM Specific
                'num_stacks': hp.quniform('num_stacks', 1, 36, 2),
                'num_blocks': hp.quniform('num_blocks', 1, 12, 1),
                'num_layers': hp.quniform('num_layers', 1, 12, 1),
                'layer_widths': hp.quniform('layer_widths', 128, 1024, 32),
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
                'num_stacks': hp.quniform('num_stacks', 1, 36, 2),
                'num_blocks': hp.quniform('num_blocks', 1, 12, 1),
                'num_layers': hp.quniform('num_layers', 1, 12, 1),
                'layer_widths': hp.quniform('layer_widths', 128, 1024, 32),
                # NBEATS Specific
                'generic_architecture': hp.choice('generic_architecture', [True]),
                'expansion_coefficient_dim': hp.quniform('expansion_coefficient_dim', 1, 99, 1),  # only if generic=True                
                # encoder
                'add_encoders': hp.choice('add_encoders', [None, 
                                                         {"datetime_attribute": {"past": ["year", "month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                                                         {"datetime_attribute": {"past": ["month"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                                                         {"datetime_attribute": {"past": ["year"]}, "transformer": Scaler(scaler=MaxAbsScaler())},
                                                         ]),
                }
                },    
                
            "nbeats": {'params': {
                'model': hp.choice('model', [NBEATSModel,]),
                # GFM Specific
                'num_stacks': hp.quniform('num_stacks', 1, 36, 2),
                'num_blocks': hp.quniform('num_blocks', 1, 12, 1),
                'num_layers': hp.quniform('num_layers', 1, 12, 1),
                'layer_widths': hp.quniform('layer_widths', 128, 1024, 32),
                # NBEATS Specific
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
                'hidden_size': hp.quniform('hidden_size', 64, 1024, 32),
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

print('Starting trials!')
if args.name != 'all':

    # combine params
    params = {**config_map[args.name]['params'], **general_params}

    # run
    argmin = fmin(fn=train_fn,
                    space=params,
                    algo=tpe.suggest,
                    max_evals=700,
                    early_stop_fn=no_progress_loss(iteration_stop_count=200,
                                                    percent_increase=1.0)
                    )

    print(f'the argmin is: {argmin}')


else:
    argmin_storage = {}
    for m_ in ['nhits', 'nbeats', 'nbeatsg', 'tft']:
        # set args name
        args.name = m_
            
        # combine params
        params = {**config_map[args.name]['params'], **general_params}

        # run
        argmin = fmin(fn=train_fn,
                        space=params,
                        algo=tpe.suggest,
                        max_evals=700,
                        early_stop_fn=no_progress_loss(iteration_stop_count=150,
                                                        percent_increase=1.0)
                        )
        
        # store argmin
        argmin_storage[m_] = argmin

        print(f'the argmin is: {argmin}')

    print(f'Total training complete! All argmins: {argmin_storage}')













