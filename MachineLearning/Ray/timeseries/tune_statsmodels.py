from typing import List, Union, Callable, Dict, Type, Tuple, Any
import time
import ray
import itertools
import pandas as pd
import numpy as np
from collections import defaultdict
from statsforecast import StatsForecast
from statsforecast.models import ETS, AutoARIMA, _TS, AutoETS, ARIMA, Theta, AutoTheta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from azureml.core import Run
from tqdm.auto import tqdm

# init Azure ML run context data
aml_context = Run.get_context()

data_path = aml_context.input_datasets['data_files']
output_path = aml_context.output_datasets['output_dir']



ray.init(ignore_reinit_error=True)


def evaluate_models_with_cv(
    models: List[_TS],
    df: pd.DataFrame,
    label_column: str,
    metrics: Dict[str, Callable[[pd.Series, pd.Series], float]],
    freq: str = "D",
    cv: Union[int, TimeSeriesSplit] = 5,
) -> Dict[_TS, Dict[str, float]]:
    # Obtain CV train-test indices for each fold.
    if isinstance(cv, int):
        cv = TimeSeriesSplit(cv)
    train_test_indices = list(cv.split(df))

    # Put df into Ray object store for better performance.
    df_ref = ray.put(df)

    # Add tasks to be executed for each fold.
    fold_refs = []
    for model in models:
        fold_refs.extend(
            [
                train_and_evaluate_fold.remote(
                    model,
                    df_ref,
                    train_indices,
                    test_indices,
                    label_column,
                    metrics,
                    freq=freq,
                )
                for train_indices, test_indices in train_test_indices
            ]
        )

    fold_results = ray.get(fold_refs)

    # Split fold results into a list of CV splits-sized chunks.
    # Ray guarantees that order is preserved.
    fold_results_per_model = [
        fold_results[i : i + len(train_test_indices)]
        for i in range(0, len(fold_results), len(train_test_indices))
    ]

    # Aggregate and average results from all folds per model.
    # We go from a list of dicts to a dict of lists and then
    # get a mean of those lists.
    mean_results_per_model = []
    for model_results in fold_results_per_model:
        aggregated_results = defaultdict(list)
        for fold_result in model_results:
            for metric, value in fold_result.items():
                aggregated_results[metric].append(value)
        mean_results = {
            metric: np.mean(values) for metric, values in aggregated_results.items()
        }
        mean_results_per_model.append(mean_results)

    # Join models and their metrics together.
    mean_results_per_model = {
        models[i]: mean_results_per_model[i] for i in range(len(mean_results_per_model))
    }
    return mean_results_per_model

@ray.remote
def train_and_evaluate_fold(
    model: _TS,
    df: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    label_column: str,
    metrics: Dict[str, Callable[[pd.Series, pd.Series], float]],
    freq: str = "D",
) -> Dict[str, float]:
    try:
        # Create the StatsForecast object with train data & model.
        statsforecast = StatsForecast(
            df=df.iloc[train_indices], models=[model], freq=freq
        )
        # Make a forecast and calculate metrics on test data.
        # This will fit the model first automatically.
        forecast = statsforecast.forecast(len(test_indices))
        return {
            metric_name: metric(
                df.iloc[test_indices][label_column], forecast[model.__class__.__name__]
            )
            for metric_name, metric in metrics.items()
        }
    except Exception:
        # In case the model fit or eval fails, return None for all metrics.
        return {metric_name: None for metric_name, metric in metrics.items()}

def generate_configurations(search_space: Dict[Type[_TS], Dict[str, list]]) -> _TS:  
    # Convert dict search space into configurations - models instantiated with specific arguments.  
    for model, model_search_space in search_space.items():  
        kwargs, values = model_search_space.keys(), model_search_space.values()  
        # Get a product - all combinations in the per-model grid.  
        for configuration in itertools.product(*values):  
            yield model(**dict(zip(kwargs, configuration)))  
  
  
@ray.remote  
def evaluate_search_space_with_cv(  
    search_space: Dict[Type[_TS], Dict[str, list]],  
    df: pd.DataFrame,  
    label_column: str,  
    metrics: Dict[str, Callable[[pd.Series, pd.Series], float]],  
    eval_metric: str,  
    mode: str = "min",  
    freq: str = "D",  
    cv: Union[int, TimeSeriesSplit] = 5,  
) -> Tuple[str, Dict[str, Any]]:  
    assert eval_metric in metrics  
    assert mode in ("min", "max")  
  
    configurations = list(generate_configurations(search_space))  
    print(  
        f"Evaluating {len(configurations)} configurations with {cv.get_n_splits()} splits each, "  
        f"totalling {len(configurations)*cv.get_n_splits()} tasks..."  
    )  
    ret = evaluate_models_with_cv(  
        configurations, df, label_column, metrics, freq=freq, cv=cv  
    )  
    # get name
    unique_id = df['unique_id'].unique().item()
  
    # Sort the results by eval_metric  
    ret = sorted(ret.items(), key=lambda x: x[1][eval_metric], reverse=(mode == "max"))  
    print("Evaluation complete!")  
    return (unique_id, {'model': str(ret[0][0]), 'params': ret[0][0].__dict__})  




dataset_list = [
 '/feature_name_by_day_by_peak.parquet',
 '/feature_name_day_by_Braga_peak.parquet',
 '/feature_name_day_by_Kingsgate_peak.parquet',
 '/feature_name_day_by_Ruby Mountain_peak.parquet'
 ]


for data_set in dataset_list:
    print(f"Now on dataset: {data_set}")

    # store results
    result_store = {}

    # load data
    df = pd.read_parquet(data_path + data_set)

    # horizon
    horizon = 90

    # data freq
    freq = 'D'

    # drop test
    df = df.groupby('unique_id').apply(lambda x: x.iloc[:-horizon]).reset_index(drop=True)  

    # features
    feature_names = df['unique_id'].unique()

    # create a list to hold the future results  
    futures = []  
    
    # loop cv  
    for feature in tqdm(feature_names):  
        print(f"Now on feature name: {feature}")  
        sub_df = df.query(f"unique_id == '{feature}'")  
        future = evaluate_search_space_with_cv.remote(  
            search_space={  
                AutoARIMA: {"season_length": [10, 15, 20, 25, 30, 35, 40]},  
                AutoETS: {"season_length":  [10, 15, 20, 25, 30, 35, 40]},  
                AutoTheta: {"season_length":  [10, 15, 20, 25, 30, 35, 40]},  
            },  
            df=sub_df,  
            label_column="y",  
            metrics={"mse": mean_squared_error},  
            eval_metric="mse",  
            cv=TimeSeriesSplit(test_size=1),  
        )  
        futures.append(future)  
    
    # Get the results once all computations are done  
    results = ray.get(futures)  
    
    # Store the results  
    result_store = {result[0]: result[1] for result in results}  

    # write
    with open(output_path + f'{data_set}.pickle', 'wb') as handle:
        pickle.dump(result_store, handle, protocol=pickle.HIGHEST_PROTOCOL)









##############






