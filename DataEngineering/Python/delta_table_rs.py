from deltalake import DeltaTable
import pyarrow.dataset as ds  


storage_options = {"azure_client_id": CLIENTID,
                   "azure_client_secret": CLIENTSECRET,
                   "tenant_id": TENANT,
                   "account_name": "moaddevlake"}

url = 'az://moaddevlakefs/Delta/Gold/CHIE/StateTransitionDCM'
dt = DeltaTable('az://moaddevlakefs/Delta/Gold/CHIE/StateTransitionDCM', storage_options=storage_options)

# 
comparative_table = dt.to_pandas(filters=[("ResourceId", "==", "9a6c8efc-9573-5a17-8b30-48775253f405")])

# 2min 49s
condition = (ds.field("ResourceId") == "9a6c8efc-9573-5a17-8b30-48775253f405")
comparative_table = dt.to_pyarrow_dataset() \
                    .to_table(filter=condition) \
                    .to_pandas()
