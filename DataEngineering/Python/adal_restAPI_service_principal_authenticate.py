import adal
import requests

# params to connect
tenant = 'tenant_vals'
target_client_id = '111111-1111-1111-1111-bead10eede4c'
self_client_id = 'our_client_id'
self_client_server = 'our_client_secret'  # use arg parse to pass in KeyVault

# build params
parameters = {
   # the client ID of the service principal we are connecting to
   "resource": target_resource_id,
   # your Azure Tenant
   "tenant" : tenant,
   "authorityHostUrl" : "https://login.microsoftonline.com",
   # our service principal that is authenticating against theirs
   "clientId" :self_client_id,
   "clientSecret" : self_client_server
   }

# establish authority url
authority_url = (parameters['authorityHostUrl'] + '/' +
                 parameters['tenant'])

# establish authority context
context = adal.AuthenticationContext(authority_url,
                                     validate_authority=parameters['tenant'] != 'adfs')

# get a token
token = context.acquire_token_with_client_credentials(parameters['resource'],
                                                      parameters['clientId'],
                                                      parameters['clientSecret']
                                                      )

# REST headers
headers = {
    'Accept': 'application/json',
    'Authorization': 'Bearer ' + token['accessToken']
    }

# API url to call
API_url = f'https://paramurlhere.com'

# get response
resp = requests.get(url=API_url, headers=headers)

if resp.ok:
    print('Response passed')
    # ...
