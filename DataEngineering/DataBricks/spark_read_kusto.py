from pyspark.sql import functions as F
from pyspark.sql import SparkSession
pyKusto = SparkSession.builder.appName("kustoPySpark").getOrCreate()

# Defining the service principal credentials for the Azure storage account; spark will inherit this to get into the db as well
spark.conf.set("fs.azure.account.auth.type", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type",  "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id", dbutils.secrets.get(scope="scopename", key="dev-synapse-sqlpool-sp-id"))
spark.conf.set("fs.azure.account.oauth2.client.secret", dbutils.secrets.get(scope="scopename", key="dev-synapse-sqlpool-sp-secret"))
spark.conf.set("fs.azure.account.oauth2.client.endpoint", "https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47/oauth2/token")

# connect params
kustoOptions = {"kustoCluster": "https://kustoserver.centralus.kusto.windows.net",
                "kustoDatabase": "kustoDB",
                "kustoAadAppId": dbutils.secrets.get(scope="scopename", key="dev-synapse-sqlpool-sp-id"),
                "kustoAadAppSecret": dbutils.secrets.get(scope="scopename", key="dev-synapse-sqlpool-sp-secret"),
                "kustoAadAuthorityID": "72f988bf-86f1-41af-91ab-2d7cd011db47"
               }

# query
kustoQuery = (
    f'''ResourceSnapshotV1
    | take 5
    '''
)

# params
kustoDf  = pyKusto.read.format("com.microsoft.kusto.spark.datasource"). \
            option("kustoCluster", kustoOptions["kustoCluster"]). \
            option("kustoDatabase", kustoOptions["kustoDatabase"]). \
            option("kustoAadAppId", kustoOptions["kustoAadAppId"]). \
            option("kustoAadAppSecret", kustoOptions["kustoAadAppSecret"]). \
            option("kustoAadAuthorityID", kustoOptions["kustoAadAuthorityID"]). \
            option("kustoQuery", f"{kustoQuery}").load()

# show
display(kustoDf)
