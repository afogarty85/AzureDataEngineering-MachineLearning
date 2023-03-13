from pathlib import Path
from loguru import logger
import pandas as pd
from generate_data.scale_sql_pool.args import get_args
from datetime import timedelta
from azure.monitor.query import LogsQueryClient
from azure.identity import ClientSecretCredential
import adal
import time
import json
import requests


def check_current_pipeline_runs(args):
    # connect to logs
    credential = ClientSecretCredential(tenant_id=args.TENANT, client_id=args.CLIENTID, client_secret=args.CLIENTSECRET)
    client = LogsQueryClient(credential)

    # kusto query
    query = (f"""SynapseIntegrationPipelineRuns
    | where Start >= ago(8h) and _ResourceId contains '{args.WORKSPACE}'
    | project Status, PipelineName, RunId""")

    # some unneeded but forced params
    start_time = pd.Timestamp('today')
    duration = timedelta(days=1)

    # send response
    response = client.query_workspace(
        workspace_id=args.LOGANALYTICSID,  # log analytics client id
        query=query,
        timespan=(start_time, duration)
        )

    # build df
    for table in response:
        df = pd.DataFrame(table.rows, columns=[col for col in table.columns])

    # create flag
    df['flag'] = ( (df['Status'] != 'Succeeded') | (df['Status'] == 'InProgress') )
    # find runs that never succeeded and therefore still running
    df['flag'] = df.groupby(['RunId'])['flag'].transform('all').astype(int)

    # find open pipelines; but filter out UAT / BOM; we'll be done before the BOM!
    df = df.loc[~df['PipelineName'].str.contains('UAT|BOM|Scaler')]

    # find flag = 1
    df = df.loc[df['flag'] == 1]
    return df


def send_scale_request(args):
    # set to the right sql pool
    uri = f'https://management.azure.com/subscriptions/{args.SUBSCRIPTIONID}/resourceGroups/{args.RESOURCEGROUP}/providers/Microsoft.Synapse/workspaces/{args.WORKSPACE}/sqlPools/moad_sql?api-version=2021-06-01'
    authority_url = 'https://login.microsoftonline.com/' + args.TENANT
    context = adal.AuthenticationContext(authority_url)
    token = context.acquire_token_with_client_credentials(
        resource='https://management.azure.com/',
        client_id=args.CLIENTID,
        client_secret=args.CLIENTSECRET
    )

    # scale to
    if args.scale == 'up':
        scale = {'sku': {'name': 'DW1000c'} }
    elif args.scale == 'down':
        scale = {'sku': {'name': 'DW500c'} }

    # REST headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token['accessToken']
    }

    # send PATCH
    resp = requests.patch(url=uri, data=json.dumps(scale), headers=headers)
    return resp


if __name__ == '__main__':
    # get args
    args = get_args()
    # today's date
    dating = str(pd.Timestamp('today', tz='US/Pacific'))[:10] + '.log'
    # set the location for saving the log
    log_path = Path(f"./logs/logs_{dating}")
    # make the dir
    log_path.parent.mkdir(exist_ok=True)
    # initialize logging
    logger.add(log_path, rotation='1 week')

    # main ops
    # flag
    shape_flag = True

    # start loop
    while shape_flag:
        logger.info("Checking if its safe to scale...")

        # check the state of the pipelines
        df = check_current_pipeline_runs(args)

        # if nothing is going on,
        if df.shape[0] == 0:

            # end the loop
            shape_flag = False
            print('No current pipeline runs are in operation -- lets scale!')

            # and send the scale request
            resp = send_scale_request(args)
            print(f'PATCH results: {resp}, {resp.reason}')
            assert resp.reason == 'Accepted', 'patch request rejected; check permissions'

        else:
            # otherwise, keep checking for a window
            time.sleep(30)
            print(f'Waiting, we have these pipelines running: {df}')

    # signal complete
    logger.info("Scaling operations concluded")


#
