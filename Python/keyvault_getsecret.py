from azure.identity import ClientSecretCredential as ClientSecretCredentialS
from azure.keyvault.secrets import SecretClient
credential = ClientSecretCredentialS(TENANT, CLIENTID, CLIENTSECRET)
kv_client = SecretClient(vault_url='https://kvurl.vault.azure.net/', credential=credential)
kv_client.get_secret("secretkey").value