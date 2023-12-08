from azureml.core import Run
import pyarrow.compute as pc
import ray
from ray import tune
from ray.train import ScalingConfig, CheckpointConfig
from ray.train.xgboost import XGBoostTrainer
from ray.tune.tuner import Tuner, TuneConfig
from ray.air.config import RunConfig
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import numpy as np
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
import xgboost
import argparse
from typing import Dict, Any  



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--num_devices", type=int, default=1, help="Number of devices to use."
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers to use."
    )
    parser.add_argument(
        "--num_trials", type=int, default=1, help="Number of tune trials."
    )    
    args = parser.parse_args()

    return args



# get args
args = parse_args()

ctx = ray.data.context.DatasetContext.get_current()
ctx.use_streaming_executor = True
ctx.execution_options.verbose_progress = True



print(f'Getting data...')
aml_context = Run.get_context()
data_path = aml_context.input_datasets['train_files']
output_path = aml_context.output_datasets['output_dir']


# load data
train_set = ray.data.read_parquet(data_path, ray_remote_args={"num_cpus": 0.25},
                            filter=pc.field('test_split').isin(['train'])) \
                            .drop_columns(['test_split', 'randomizedTestKey', 'condition_name_SI'])

valid_set = ray.data.read_parquet(data_path, ray_remote_args={"num_cpus": 0.25},
                            filter=pc.field('test_split').isin(['valid'])) \
                            .drop_columns(['test_split', 'randomizedTestKey', 'condition_name_SI', 'weight'])


run_config = RunConfig(
    storage_path=output_path,
    checkpoint_config=CheckpointConfig(
        # Checkpoint every iteration.
        checkpoint_frequency=1,
        # Keep all
        num_to_keep=None,
    )
)


# trainer
trainer = XGBoostTrainer(
    scaling_config=ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=True,
        placement_strategy='SPREAD',
        resources_per_worker={"GPU": args.num_devices}
    ),
    early_stopping_rounds=25,
    label_column="SYNDROME_SI",
    dmatrix_params={"train": {'weight': 'weight'}, },
    num_boost_round=1500,
    params={
        "booster": "gbtree",
        "objective": "binary:logistic",
        'device': 'cuda',
        'max_depth': 6,
        'eta': 0.01,        
        "tree_method": "gpu_hist",
        "eval_metric": ["error", "aucpr"],  # early stopping depends on last choice here
    },
    datasets={"train": train_set, "valid": valid_set},
    callbacks=[TuneReportCheckpointCallback({"valid-aucpr": "valid-aucpr"}, filename="model.json")],
    run_config=run_config,
    verbose_eval=True,
)

# result = trainer.fit()
# print(f" Result Metrics; {result.metrics}")

# Scheduler
scheduler = ASHAScheduler(
    time_attr='training_iteration',
    max_t=1000,
    grace_period=100,
    reduction_factor=2,
    metric='valid-aucpr',
    mode='max',
)

# RF -- sample rows, cols
# GBT -- use all data
param_space = {"params":
                    {
                    "booster": "gbtree",
                    'device': 'cuda',
                    "objective": "binary:logistic",
                    "tree_method": "hist",
                    'predictor': 'gpu_predictor',
                    "eval_metric": ["error", "aucpr"],  # early stopping depends on last choice here
                    'scale_pos_weight': 1.05,
                    "min_child_weight": tune.choice([1, 2, 3, 4]),
                    "subsample": tune.uniform(0.5, 1.0),
                    "colsample_bytree": tune.uniform(0.5, 1.0),
                    "eta": tune.loguniform(1e-4, 1e-1),
                    "gamma": tune.uniform(0, 5),  
                    "reg_lambda": tune.uniform(1, 10),  
                    "reg_alpha": tune.uniform(0, 1),                      
                    "max_depth": tune.randint(6, 12),
                    }
                }


tuner = Tuner(
    trainable=trainer,
    param_space=param_space,
    tune_config=TuneConfig(max_concurrent_trials=2,
                           num_samples=args.num_trials,
                           scheduler=scheduler),
    run_config=run_config,
)

# Execute tuning
result_grid = tuner.fit()

# Fetch the best result with its best hyperparameter config 
best_result = result_grid.get_best_result(metric='valid-aucpr', mode='max')
print("Best Result:", best_result)

# get checkpoint
checkpoint = best_result.checkpoint
print(f"Best Checkpoint: {checkpoint}")


# test set
ctx.execution_options.preserve_order = True  # keep idx order the same

test_set = ray.data.read_parquet(data_path, ray_remote_args={"num_cpus": 0.25},
                            filter=pc.field('test_split').isin(['test'])) \
                            .drop_columns(['test_split', 'randomizedTestKey', 'condition_name_SI', 'weight'])

class OfflinePredictor:
    def __init__(self):
        self._model = xgboost.Booster()
        self._model.load_model(checkpoint.path + '/model.json')

    def __call__(self, batch):
        inference_batch = pd.DataFrame(batch)
        labels = inference_batch['SYNDROME_SI'].values

        dmatrix = xgboost.DMatrix(inference_batch.drop(['SYNDROME_SI'], axis=1))
        outputs = self._model.predict(dmatrix)
        return {"prediction": outputs, "labels": labels}

# get preds
predicted_probabilities = test_set.map_batches(OfflinePredictor, num_gpus=1, batch_size=4096*4, compute=ray.data.ActorPoolStrategy(size=args.num_workers))
predicted_probabilities = predicted_probabilities.take_all()

# combine
preds = np.round([d['prediction'] for d in predicted_probabilities])
print(f"Found this set of preds: {preds[:50]}")

labels = [d['labels'] for d in predicted_probabilities]
print(f"Found this set of labels: {labels[:50]}")

# f1
my_f1 = f1_score(labels, preds, average=None)
print(f"F1: {my_f1}")

# classification report
print("Classification Report: \n")
print(classification_report(labels, preds, labels=[0, 1]))




#