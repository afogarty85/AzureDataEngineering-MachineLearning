# Defining the service principal credentials for the Azure storage account; spark will inherit this to get into the db as well
spark.conf.set("fs.azure.account.auth.type", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type",  "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id", dbutils.secrets.get(scope="scopename", key="dev-synapse-sqlpool-sp-id"))
spark.conf.set("fs.azure.account.oauth2.client.secret", dbutils.secrets.get(scope="scopename", key="dev-synapse-sqlpool-sp-secret"))
spark.conf.set("fs.azure.account.oauth2.client.endpoint", "https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47/oauth2/token")

# params
schema = 'dim'
table = 'Program'


# get df
df = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", "jdbc:sqlserver://chiescopename.sql.azuresynapse.net:1433;database=db_name;encrypt=true;trustServerCertificate=true;hostNameInCertificate=*.sql.azuresynapse.net;loginTimeout=30") \
  .option("useAzureMSI", "true") \
  .option("enableServicePrincipalAuth", "true") \
  .option("tempDir", "abfss://scopenamelakefs@scopenamelake.dfs.core.windows.net/temp") \
  .option("dbTable", f"{schema}.{table}") \
  .load()


# show
display(df)
