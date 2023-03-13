# Define the variables used for creating connection strings
adlsAccountName = "storagedevlake"
adlsContainerName = "storagedevlakefs"
#adlsFolderName = "RAW"
mountPoint = "/mnt/"

# Application (Client) ID
applicationId = dbutils.secrets.get(scope="storagedev", key="adls-client-id")
                                    
# Application (Client) Secret Key
authenticationKey = dbutils.secrets.get(scope="storagedev", key="adls-sp-secret")
                                    
# Directory (Tenant) ID
tenandId = dbutils.secrets.get(scope="storagedev", key="tenant")

# endpoint
endpoint = "https://login.microsoftonline.com/" + tenandId + "/oauth2/token"

source = "abfss://" + adlsContainerName + "@" + adlsAccountName + ".dfs.core.windows.net/" # + adlsFolderName

# Connecting using Service Principal secrets and OAuth
configs = {"fs.azure.account.auth.type": "OAuth",
           "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
           "fs.azure.account.oauth2.client.id": applicationId,
           "fs.azure.account.oauth2.client.secret": authenticationKey,
           "fs.azure.account.oauth2.client.endpoint": endpoint}
 
# Mount ADLS Storage to DBFS only if the directory is not already mounted
if not any(mount.mountPoint == mountPoint for mount in dbutils.fs.mounts()):
    dbutils.fs.mount(source=source,
                   mount_point=mountPoint,
                   extra_configs=configs)

# to unmount
#dbutils.fs.unmount("/mnt/")
