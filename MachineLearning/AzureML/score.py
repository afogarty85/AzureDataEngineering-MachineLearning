import pandas as pd
import numpy as np
import xgboost as xgb
import base64
import requests
import os
import logging
import json
import numpy as np
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from category_encoders import * 
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential



def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global bst
    global X_train
    global y_train
    print('Starting init...')

    # load trained model
    bst = xgb.Booster()
    bst.load_model("/var/azureml-app/train_data/xgboost_trained.json")

    # load X, y train
    X_train = pd.read_parquet(r'/var/azureml-app/train_data/X_train.parquet')
    y_train = pd.read_parquet(r'/var/azureml-app/train_data/y_train.parquet')

    print('Init Complete!')
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    """

    # get user input
    user_inputs = json.loads(raw_data)

    repo_name = user_inputs['repo_name']
    commit_id = user_inputs['commit_id']
    pipeline_name = user_inputs['pipeline_name']

    # do stuff
    # predicted_out is a dataframe

    return json.dumps({'data': predicted_out.to_dict(orient='records')})





#