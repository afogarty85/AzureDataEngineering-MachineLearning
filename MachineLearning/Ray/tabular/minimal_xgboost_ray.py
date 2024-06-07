from ray.train import ScalingConfig, CheckpointConfig
from ray.train.xgboost import XGBoostTrainer
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import numpy as np
import argparse
from typing import Dict, Any  
import xgboost





# trainer
trainer = XGBoostTrainer(
    scaling_config=ScalingConfig(
        num_workers=1,
        use_gpu=True,
        placement_strategy='SPREAD',
        resources_per_worker={"GPU": 1}
    ),
    early_stopping_rounds=25,
    label_column="SYNDROME_SI",
    num_boost_round=1500,
    params={
        "booster": "gbtree",
        "objective": "binary:logistic",
        'device': 'cuda',
        'max_cat_to_onehot': 5,
        'max_depth': 6,
        'eta': 0.01,
        "tree_method": "hist",
        "eval_metric": ["error", "aucpr"],  # early stopping depends on last choice here
    },
    datasets={"train": train_set, "valid": valid_set},
    run_config=None,
    verbose_eval=True,
)

result = trainer.fit()
print(f" Result Metrics; {result}")



class Predict:

    def __init__(self, checkpoint):
        self.model = XGBoostTrainer.get_model(checkpoint)

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        dmatrix = xgboost.DMatrix(batch)
        return {"predictions": self.model.predict(dmatrix)}

scores = test_set.drop_columns(["SYNDROME_SI"]).map_batches(
    Predict, 
    fn_constructor_args=[result.checkpoint], 
    concurrency=1,
    batch_size=1024*4,
    num_gpus=1,
    batch_format="pandas"
)

predicted_labels = scores.map_batches(lambda df: df, batch_format="pandas").take_all()


all_preds = [d['predictions'] for d in predicted_labels]
all_labels = test_set.select_columns(['SYNDROME_SI']).take_all()
all_labels = [d['SYNDROME_SI'] for d in all_labels]


print(classification_report(y_true=all_labels, y_pred=all_preds))