from azure.kusto.data.exceptions import KustoServiceError
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder, ClientRequestProperties
from azure.kusto.data._models import KustoResultTable
import pandas as pd

# kusto cluster
cluster = 'https://server.westus2.kusto.windows.net'

# kusto db
kustodb = 'dbnamehere'

# service principal: get AAD auth
kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(
    cluster, CLIENTID, CLIENTSECRET, TENANT
)

# personal access instead of SP
# kcsb=f"Data Source={cluster};AAD Federated Security=True"

# open a kusto client
client = KustoClient(kcsb)

# generate a query
query = f''' 
                TableName | take 5
                ''' 

# get response
response = client.execute(kustodb, query)
table = response.primary_results[0]
columns = [col.column_name for col in table.columns]
df = pd.DataFrame(table.raw_rows, columns=columns)