import ray
from ray.train.xgboost import XGBoostTrainer, XGBoostPredictor
from ray.air.config import ScalingConfig
import pandas as pd
import numpy as np
import os
import psutil
import pyarrow.compute as pc
from azureml.core import Run
import gc
import pickle
np.set_printoptions(suppress=True)

os.environ['RAY_DATA_DISABLE_PROGRESS_BARS'] = "1"

# set data context
ctx = ray.data.DataContext.get_current()
ctx.use_push_based_shuffle = True
ctx.use_streaming_executor = True
ctx.execution_options.locality_with_output = True
ctx.execution_options.verbose_progress = True

if not ray.is_initialized():
    # init a driver
    ray.init()
    print('ray initialized')


# init Azure ML run context data
aml_context = Run.get_context()

# cluster available resources which can change dynamically
print(f"Ray cluster resources_ {ray.cluster_resources()}")


print(f'Getting data...')
train_set = ray.data.read_parquet(aml_context.input_datasets['RTE'],
                            use_threads=True,
                            filter=pc.field('test_split').isin(['train'])) \
                            .drop_columns(['test_split'])


# cats
cat_cols = ['INPUT__REG__m_cmd_merge_mode_0_',
            'INPUT__REG__m_cmd_merge_mode_1_',
            'INPUT__REG__m_hazard_type',
            'INPUT__YML__testname',
            'INPUT__REG__m_ddr_speed_variant']

train_set = train_set.drop_columns(cat_cols)

# add noise
train_set = train_set.add_column("noise", lambda df: np.random.normal(0, 1, df.shape[0]))

# split
_, train_set = train_set.train_test_split(0.6)

# params
params = {
                'booster': 'gbtree',
                'num_parallel_tree': 300,
                'objective': 'multi:softprob',
                'num_class': 67,
                'learning_rate': 1,
                'max_depth': 5,
                'grow_policy': 'lossguide',
                'sampling_method': 'gradient_based',
                'max_bin': 64,
                'eval_metric': 'mlogloss',
                'reg_lambda': 20,
                'reg_alpha': 3,
                'n_jobs': -1,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'colsample_bynode': 0.8,
                'tree_method': 'gpu_hist',
                "predictor": "auto",
                'subsample': 0.5,
                'validate_parameters': 1,
                }

# trainer -- random forest
trainer = XGBoostTrainer(
    scaling_config=ScalingConfig(num_workers=6,
                            use_gpu=True,  # if use_gpu=True, num_workers = num GPUs
                            _max_cpu_fraction_per_node=0.8,
                            resources_per_worker={"GPU": 1},
                            trainer_resources={"CPU": 1}
                                ),
    label_column="SYNDROME",
    params=params,
    datasets={"train": train_set},
    num_boost_round=1,
)



# fit
result = trainer.fit()

# get booster
booster = XGBoostPredictor.from_checkpoint(result.checkpoint)

# add col names
booster.model.feature_names = [x for x in train_set.columns() if x != 'SYNDROME']

# get feature importance
feature_important = booster.model.get_score(importance_type='gain')

# write
with open(aml_context.output_datasets['output1'] + '/feature_importance.pickle', 'wb') as handle:
    pickle.dump(feature_important, handle, protocol=pickle.HIGHEST_PROTOCOL)

