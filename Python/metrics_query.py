import os
from datetime import timedelta, datetime
from azure.monitor.query import MetricsQueryClient, MetricAggregationType
from azure.identity import DefaultAzureCredential, ClientSecretCredential
import pandas as pd

# Microsoft.Synapse/workspaces
uri = '/subscriptions/<subscription_id>/resourceGroups/<rg_name>/providers/Microsoft.Synapse/workspaces/<synapse_workspace_name>'

# Microsoft.Synapse/workspaces/sqlPools
uri = '/subscriptions/<subscription_id>/resourceGroups/<rg_name>/providers/Microsoft.Synapse/workspaces/<synapse_workspace_name>/sqlPools/<synapse_workspace_dedicated_sql_pool_name>/'


# connect with a service principal
credential = ClientSecretCredential(tenant_id=TENANT, client_id=CLIENTID, client_secret=CLIENTSECRET)
client = MetricsQueryClient(credential)
metrics_uri = uri
response_active_queries = client.query_resource(
    metrics_uri,
    metric_names=["ActiveQueries"],
    timespan=timedelta(days=90),
    granularity=timedelta(days=1)
    )

# storage df
storage_df_activeQueries = pd.DataFrame()
# for the metric of interest
for metric in response_active_queries.metrics:
    # for its timestamp of interest
    for time_series_element in metric.timeseries:
        # for its value of interest at the timestamp
        for metric_value in time_series_element.data:
            # store it in a df
            temp_df = pd.DataFrame({'time': [metric_value.timestamp], 'total': [metric_value.total]})
            # concat it to the storage
            storage_df_activeQueries = pd.concat([storage_df_activeQueries, temp_df], axis=0)

# check results
storage_df_activeQueries['total'].sum()
storage_df_activeQueries.sort_values(['time'], ascending=False)
