import pandas as pd  
import numpy as np  
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer  
from scipy.stats import entropy  
import joblib  
import pickle
import umap  
import hdbscan  
from azure.keyvault.secrets import SecretClient
import pyodbc
import ray
from ray import tune  
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score  
from sklearn.model_selection import GroupShuffleSplit  
from concurrent.futures import ProcessPoolExecutor, as_completed  
from azure.identity import DefaultAzureCredential
import json
from sklearn.preprocessing import FunctionTransformer  



def generate_inversions(df, group_key):  
    ''' apply before feature engineering '''
    # Get the indices for each group  
    grouped = df.groupby(group_key).indices  
      
    # Initialize an empty DataFrame for the inversions  
    inverted_df_list = []  
      
    # Loop through each group and reverse the order of rows  
    for indices in grouped.values():  
        if len(indices) >= 2:  # Only consider groups with size 2 or greater
            if df.iloc[indices]['MinutesInProduction'].sum() == 0:
                # lets not add in even more 0 repairs
                continue
            inverted_df_list.append(df.iloc[indices[::-1]])  
      
    # Concatenate all inverted dataframes  
    inversions = pd.concat(inverted_df_list, ignore_index=True)  
    return inversions  

def specialization_entropy(error_codes):  
    # Count occurrences of each error code  
    value_counts = error_codes.value_counts()  
    # Normalize counts to sum to 1 (probability distribution)  
    probabilities = value_counts / value_counts.sum()  
    # Calculate entropy of the distribution  
    return entropy(probabilities)  
  
def generate_window_slices(df, group_key, window_size=2):
    ''' apply before feature engineering '''
    # Ensure the DataFrame is sorted within each group  
    df = df.sort_values(by=[group_key] + (['CreatedDate'] if 'CreatedDate' in df.columns else []) )  
  
    # Compute the group sizes and filter out those groups with size less than or equal to window_size  
    group_sizes = df.groupby(group_key).size()  
    eligible_groups = group_sizes[group_sizes > window_size].index  
  
    # Keep only rows belonging to eligible groups  
    eligible_df = df[df[group_key].isin(eligible_groups)]  
  
    # Create a group label array  
    group_labels = eligible_df[group_key].values  
  
    # Compute the sliding window views of the indices and group labels  
    indices = np.arange(len(eligible_df))  
    rolling_indices = np.lib.stride_tricks.sliding_window_view(indices, window_shape=window_size)  
    rolling_group_labels = np.lib.stride_tricks.sliding_window_view(group_labels, window_shape=window_size)  
  
    # Find windows that do not cross group boundaries  
    valid_windows_mask = np.all(rolling_group_labels == np.expand_dims(rolling_group_labels[:, 0], axis=1), axis=1)  
    valid_windows = rolling_indices[valid_windows_mask]  
  
    # Flatten the valid windows and retrieve the corresponding rows from the dataframe  
    valid_indices = valid_windows.ravel()  
    window_slices_df = eligible_df.iloc[valid_indices].reset_index(drop=True)  
  
    return window_slices_df  



# Load the data
df = pd.read_parquet('./data/azrepair_rl.parquet',
                      columns=['TicketId', 'InFleet', 'HoursInProduction', 'MinutesInProduction',
                          'ResourceId', 'FaultCode', 'DiscreteSkuMSFId', 'RobertsRules',
                          'CreatedDate', 'Cluster',
                          'SP6RootCaused', 'MsfNumber',
                          'FirstTimeStamp', 'KickedFlag', 'WorkEndDate',
                          'UserUpdatedBy', 'TicketAgility', 'TopFCHops',
                          'Censored', 'DCMStatesDuringTicket',
                      ])

df.query("ResourceId == 'e329945a-ef85-44e4-bbae-a51d3499b2d4' ")
df.query("MinutesInProduction.isna()")['ResourceId'].sample(n=2)
point_wise = True  # False for pairwise


# Limited Processing
# ====================================================

# Convert date columns to datetime  
df['CreatedDate'] = pd.to_datetime(df['CreatedDate'], format='mixed', utc=True)  
df['WorkEndDate'] = pd.to_datetime(df['WorkEndDate'], format='mixed', utc=True)  
df['FirstTimeStamp'] = pd.to_datetime(df['FirstTimeStamp'])

# filter
df = df.query("RobertsRules.notna() ")
df = df[df['RobertsRules'].str.contains('Replace', na=False)]

# sort
df = df.sort_values(by=['ResourceId', 'CreatedDate']).reset_index(drop=True)

# drop multi-ticket repairs that get assigned same outcomes for same repair, faultcode
df = df.drop_duplicates(subset=['ResourceId', 'FaultCode', 'MinutesInProduction', 'RobertsRules'])



def find_chained_repairs(df):
    df = df.copy()
    # Compute the time differences within each group    
    df['time_diff_forward'] = df.groupby('ResourceId')['CreatedDate'].diff(-1).abs()    
    df['time_diff_backward'] = df.groupby('ResourceId')['CreatedDate'].diff().abs()    
        
    # Set the sequence flag if the time difference is less than or equal to 24 hours    
    df['sequence_flag'] = ((df['time_diff_forward'] <= pd.Timedelta(hours=24)) |    
                        (df['time_diff_backward'] <= pd.Timedelta(hours=24))).astype(int)    
        
    # Drop the temporary columns used for computing differences    
    df.drop(columns=['time_diff_forward', 'time_diff_backward'], inplace=True)    

    # Compute the backward time differences within each group  
    df['time_diff_backward'] = df.groupby('ResourceId')['CreatedDate'].diff()  
    
    # Identify where a new sequence should start  
    new_sequence = (df['time_diff_backward'] > pd.Timedelta(hours=24)) | df['time_diff_backward'].isnull()  
    
    # Cumulatively sum the new_sequence flags to get unique sequence identifiers, per 'ResourceId'  
    df['sequence_id'] = new_sequence.groupby(df['ResourceId']).cumsum()  
    
    # Drop the temporary column used for computing differences  
    df.drop(columns=['time_diff_backward'], inplace=True)  
    return df

# find sequences
df = find_chained_repairs(df)

# go json
df['RobertsRules'] = df['RobertsRules'].map(json.loads)  

# combine JSON arrays in RobertsRules  
def combine_jsons(series):  
    combined_json_array = []  
    for json_array in series:  
        combined_json_array.extend(json_array)  
    return combined_json_array  

# filter
df_sequences = df[df['sequence_flag'] == 1]  

# combine jsons
combined_jsons = df_sequences.groupby(['ResourceId', 'sequence_id'])['RobertsRules'] \
                            .apply(combine_jsons) \
                            .reset_index(name='CombinedRobertsRules')  
  
# merge to main
df = df.merge(combined_jsons, on=['ResourceId', 'sequence_id'], how='left')  
  
# rows where sequence_flag == 1, replace RobertsRules with the CombinedRobertsRules  
df.loc[df['sequence_flag'] == 1, 'RobertsRules'] = df['CombinedRobertsRules']  
  
# drop all but the last row of each group where sequence_flag == 1  
df = df.drop_duplicates(subset=['ResourceId', 'sequence_id'], keep='last') \
        .drop(columns=['CombinedRobertsRules'])

# Fill missing values
df['MinutesInProduction'] = df['MinutesInProduction'].fillna(0)  

# clean
df['TopFCHops'] = df['TopFCHops'].replace('', np.nan)

# tends to be tickets resolved today and mtx not yet updated:4
df = df.query("Censored.notna()").reset_index(drop=True)  # why 53 obs with nan?

# stateless transformations
# ====================================================

# device age
df['DeviceAge'] = (df['CreatedDate'] - df['FirstTimeStamp']).dt.total_seconds() / 3600

# sev repair
df['SevRepair'] = df['RobertsRules'].astype('string').str.contains("Motherboard|CPU|GPU|FPGA").astype(int)  

# boolean column encode 
df['SP6RootCaused'] = df['SP6RootCaused'].map({"true": 1, "false": 0}).fillna(0)  

# assign embeddings
with open('/mnt/c/Users/afogarty/Desktop/AZRepairRL/data/RobertsRules_embeddings.pickle', 'rb') as handle:
    embedding_dict = pickle.load(handle)

# go str
df['RobertsRules'] = df['RobertsRules'].map(lambda x: json.dumps(x, separators=(',', ':')))  

# dense rep
df['RobertsRulesEmbeddings'] = df['RobertsRules'].map(embedding_dict) # fails



# stateless joins
# ====================================================

# get secs
KVUri = "https://moaddev6131880268.vault.azure.net"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)

# secret vals
CLIENTID = client.get_secret("dev-synapse-sqlpool-sp-id").value
CLIENTSECRET = client.get_secret("dev-synapse-sqlpool-sp-secret").value
TENANT = client.get_secret("tenant").value

def retrieve_sql(CLIENTID, CLIENTSECRET, query):
    # client id / secret
    SERVER='chiemoaddev'
    s2="Driver={ODBC Driver 17 for SQL Server};Server=tcp:%s.sql.azuresynapse.net,1433;Database=moad_sql;UID=%s;PWD=%s;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=15;Authentication=ActiveDirectoryServicePrincipal" % (
        SERVER, CLIENTID, CLIENTSECRET)
    # connect to the sql server
    cnxn=pyodbc.connect(s2)
    # get sql data
    df = pd.read_sql(query, cnxn)
    return df

# generate a server set
query = f"""
    select
P.ID,
P.Name,
MPA.M_ALIAS as DiscreteSkuMSFId
from [raw].[OnePDMPart] AS P
JOIN [raw].[OnePDMItemClass] AS IC ON IC.[id] = P.[ITEM_CLASS_ID]
LEFT JOIN [raw].[OnePDMMPartAlias] as MPA on MPA.SOURCE_ID = P.ID
WHERE IC.[Label] = 'Server'
AND P.IS_CURRENT = 1
"""

# get df
server_df = retrieve_sql(CLIENTID, CLIENTSECRET, query)
pattern = r'(MSF-\d{6})'
extracted_series = server_df['Name'].str.extract(pattern, expand=False)
  
server_df['DiscreteSkuMSFId'] = np.where(  
    server_df['DiscreteSkuMSFId'].isna(),  
    extracted_series,  
    server_df['DiscreteSkuMSFId']  
)  

# reduce to what we have
server_df = server_df.merge(df[['DiscreteSkuMSFId']].drop_duplicates(), how='inner', on=['DiscreteSkuMSFId'])

# get raw bom # select * from raw.onepdmpartbom
query = f"""select * from raw.onepdmpartbom"""
raw_bom = retrieve_sql(CLIENTID, CLIENTSECRET, query)
raw_bom = raw_bom.set_index("SOURCE_ID")

raw_bom.query("RELATED_ID == 'C0125CF6865E498E99225ED1CA7B483E' ")  # this is the server

# max server count
max_server_count = raw_bom[raw_bom['RELATED_ID'].isin(server_df['ID'].values)].groupby(['RELATED_ID'], as_index=False)['QUANTITY'].max()
max_server_count = max_server_count.rename(columns={'RELATED_ID': "serverID", 'QUANTITY': 'NumBlades'})

# parent child recursion
def recur(df, init_parent, parents, DiscreteSkuMSFId=None,  parentChild=None, step=0, t_df=None, nextExtras=None):
    '''
    Recursively search and generate child / parent relationships

    params::
    df - DataFrame - the DF we are extracting nested data from
    init_parent - list - a single parent ID wrapped in a list
    parents - list - a list containing n-many new parents (childs of init_parent); to start, we use init_parent here
    parentChild - list - captures the parents of the downstream children
    step - int - capturing the depth of search
    t_df - DataFrame - storage container to yield; nothing to enter to start
    nextExtras -- dict - conditional item that will tell us to look for the right amount of new parent parts

    '''
    if len(parents) == 0:
        # end
        return

    if step >= 1:
        # give me the df
        yield t_df

    # generate some info of interest; set parent column to index for faster searching
    curr_pull = df[df.index.isin(parents)]

    if (nextExtras is not None) and (step > 1):
        # if we have duplicate childParts that are now parents, .isin will treat them uniquely, so
        for k, v in nextExtras.items():
            # extract set
            tdf_ = curr_pull[curr_pull.index.isin([k])][["ID", "RELATED_ID", "SUBSTITUTES", "QUANTITY", "SORT_ORDER"]]
            # multiply n times
            tdf_ = pd.concat(tdf_ for i in range(0, v-1))
            if len(tdf_) > 0:
                # add to current data
                curr_pull = pd.concat([curr_pull, tdf_], axis=0)

    # set the next parents
    nextParents = curr_pull.RELATED_ID.values
    # set current children
    currentChildren = curr_pull.RELATED_ID.values
    # the parents are set as the index
    currentParents = curr_pull.index.values
    # get quantities
    currentQuantities = curr_pull.QUANTITY.values
    # get subs
    currentSubstitutes = curr_pull.SUBSTITUTES.values
    # sortOrder
    currentSortOrder = curr_pull.SORT_ORDER.values

    # prepare a dict to send into next iteration
    if curr_pull.groupby(['RELATED_ID']).filter(lambda x: len(x) > 1).shape[0] > 0:
        sub_search = curr_pull.groupby(['RELATED_ID']).filter(lambda x: len(x) > 1)
        nextExtras = sub_search.groupby(['RELATED_ID']).size().to_dict()
    else:
        nextExtras = None

    # agg data
    t_df = pd.DataFrame({
                        'DiscreteSkuMSFId': DiscreteSkuMSFId,
                        'parentPartID': [init_parent] * len(nextParents),
                        'childPartID': currentChildren,
                        'childParentPartID': currentParents,
                        'Quantities': currentQuantities.astype(int),
                        'Substitutes': currentSubstitutes.astype(int),
                        'sortOrder': currentSortOrder.astype(int),
                        'Level': [step] * len(nextParents)
                        })

    # yield from
    yield from recur(df=df, init_parent=init_parent, parents=nextParents, DiscreteSkuMSFId=DiscreteSkuMSFId, parentChild=currentParents, step=step+1, t_df=t_df, nextExtras=nextExtras)


# list of parents
parents = server_df[['ID', 'DiscreteSkuMSFId']].drop_duplicates()['ID'].values
discrete_sku_msfs = server_df[['ID', 'DiscreteSkuMSFId']].drop_duplicates()['DiscreteSkuMSFId'].values

# The worker function to be executed in parallel  
def worker(sub_parents, discrete_sku_msfs, raw_bom):  
    storage = []  
    for parent, msf in zip(sub_parents, discrete_sku_msfs):  
        out = recur(df=raw_bom, init_parent=[parent], parents=[parent], DiscreteSkuMSFId=msf, parentChild=None, step=1, t_df=None)  
        for x in out:  
            storage.append(x)  
    return storage  
  
# The main function that sets up parallel processing  
def main(server_df, raw_bom):  
    parents = server_df[['ID', 'DiscreteSkuMSFId']].drop_duplicates()['ID'].values  
    discrete_sku_msfs = server_df[['ID', 'DiscreteSkuMSFId']].drop_duplicates()['DiscreteSkuMSFId'].values  
  
    # Split the list of parents into chunks for parallel processing  
    n_splits = 8
    split_parents = np.array_split(parents, n_splits)  
    split_discrete_sku_msfs = np.array_split(discrete_sku_msfs, n_splits)  
  
    with ProcessPoolExecutor() as executor:  
        # Schedule the worker function for each chunk of parents  
        futures = [executor.submit(worker, sub_parents, sub_msfs, raw_bom) for sub_parents, sub_msfs in zip(split_parents, split_discrete_sku_msfs)]  
  
        # Collect the results as they complete  
        storage = []  
        for future in as_completed(futures):  
            storage.extend(future.result())  
  
    # Concatenate the results  
    server_bom = pd.concat(storage).reset_index(drop=True)  
    return server_bom  

# parallel server bom
server_bom = main(server_df, raw_bom)  
print(server_bom.shape)

# process and agg
server_bom['parentPartID'] = server_bom['parentPartID'].explode()
server_bom = server_bom.rename(columns={"parentPartID": "serverID"})

# join max server
server_bom = server_bom.merge(max_server_count, how='left', on=['serverID'])

msf_agg = server_bom.groupby(['DiscreteSkuMSFId'], as_index=True).agg({"Quantities": "sum",
                                                      "Substitutes": "sum",
                                                      "Level": "max",
                                                      "DiscreteSkuMSFId": len,
                                                      'NumBlades': "max"})

msf_agg = msf_agg.rename(columns={"Quantities": "NumBOMParts",
                                  "Substitutes": "NumBOMSubstitutes",
                                  "Level": "MaxBOMDepth",
                                  "DiscreteSkuMSFId": "NumBOMRows",}) \
                .reset_index()

msf_agg.to_parquet('./data/msf_agg.parquet')

# join to main df
df = df.merge(msf_agg, how='left', on=['DiscreteSkuMSFId'])


# get server characteristics
query = f"""
    select partID, BusinessGroup, CFMAirFlow, Depth, Weight, TargetWorkload, Width,
    P.Name
    from dim.ServerCharacteristic as SC
    join raw.onepdmpart as P on P.ID = SC.partID
"""

# get df
server_characteristic = retrieve_sql(CLIENTID, CLIENTSECRET, query)
server_characteristic = server_characteristic.replace("<missing>", np.nan)

# get msf
pattern = r'(MSF-\d{6})'
server_characteristic['DiscreteSkuMSFId'] = server_characteristic['Name'].str.extract(pattern)
server_characteristic = server_characteristic.dropna(subset=['DiscreteSkuMSFId'])

df = df.merge(server_characteristic, how='left', on=['DiscreteSkuMSFId'])


# test/train split   
# ===================================================

# Group by 'FaultCode' and 'DiscreteSkuMSFId' and filter out single-instance groups  
single_instances = df.groupby(['FaultCode', 'DiscreteSkuMSFId']).filter(lambda x: len(x) == 1)  
multi_instances = df.groupby(['FaultCode', 'DiscreteSkuMSFId']).filter(lambda x: len(x) > 1)  
  
# Add single-instance groups directly to the training set  
train = single_instances  
  
# Initialize sets to track 'ResourceId' allocations  
train_resource_ids = set(train['ResourceId'])  
valid_resource_ids = set()  
test_resource_ids = set()  

# Use GroupShuffleSplit to split the multi-instance groups while keeping 'ResourceId' unique  
gss = GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=42) 
for train_idx, test_idx in gss.split(multi_instances, groups=multi_instances['ResourceId']):  
    # Update the training set and the set of allocated 'ResourceId's  
    train_multi = multi_instances.iloc[train_idx]  
    train = pd.concat([train, train_multi])  
    train_resource_ids.update(train_multi['ResourceId'])  
  
    # Split the test_idx further into validation and test, ensuring unique 'ResourceId'  
    test_valid_multi = multi_instances.iloc[test_idx]  
    unique_resource_ids = np.array(list(set(test_valid_multi['ResourceId']) - train_resource_ids))  
    np.random.seed(42)  
    np.random.shuffle(unique_resource_ids)  
    midpoint = len(unique_resource_ids) // 2  
  
    # Allocate 'ResourceId's to validation and test sets  
    valid_resource_ids.update(unique_resource_ids[:midpoint])  
    test_resource_ids.update(unique_resource_ids[midpoint:])  
  
# Now, separate the validation and test sets based on allocated 'ResourceId's  
valid = test_valid_multi[test_valid_multi['ResourceId'].isin(valid_resource_ids)]  
test = test_valid_multi[test_valid_multi['ResourceId'].isin(test_resource_ids)]  
  
# Assertions to check for leakage are not necessary here as we've tracked 'ResourceId' allocations  
# However, we can still perform them as a safety check  
  
assert set(train['ResourceId']).isdisjoint(valid['ResourceId']), "Leakage detected between train and valid sets."  
assert set(train['ResourceId']).isdisjoint(test['ResourceId']), "Leakage detected between train and test sets."  
assert set(valid['ResourceId']).isdisjoint(test['ResourceId']), "Leakage detected between valid and test sets."  

print(f"Train shape: {train.shape}")  
print(f"Validation shape: {valid.shape}")  
print(f"Test shape: {test.shape}")

# add split
train['test_split'] = 'train'
valid['test_split'] = 'valid'
test['test_split'] = 'test'



if point_wise:
    # generate inversions and slices
    inverted_data = generate_inversions(train, 'ResourceId')  
    window_slices_data = generate_window_slices(train, 'ResourceId', window_size=2)  

    # rename
    train = train.rename(columns={"ResourceId": "ResourceIdOriginal"})
    inverted_data = inverted_data.rename(columns={"ResourceId": "ResourceIdOriginal"})
    window_slices_data = window_slices_data.rename(columns={"ResourceId": "ResourceIdOriginal"})

    # label
    inverted_data['ResourceId'] = inverted_data['ResourceIdOriginal'] + '_inverted'  
    window_slices_data['ResourceId'] = window_slices_data['ResourceIdOriginal']  + '_window_slice'
    train['ResourceId'] = train['ResourceIdOriginal'] + '_full'

    # give sequence label
    inverted_data['sequence_type'] = '_inverted'  
    window_slices_data['sequence_type'] = '_window_slice'
    train['sequence_type'] = '_full'

    # join
    train = pd.concat([train, window_slices_data, inverted_data], axis=0).reset_index(drop=True)
    print("Generated train-test split")


# Stateful Device Transformations
# ====================================================

def calculate_most_common_action(df, group_by_cols, action_col, new_col_name, last_common_actions=None):  
    df = df.copy()  
  
    # Shift the actions within each group to avoid including the current step's action  
    df['Shifted_Action'] = df.groupby(group_by_cols)[action_col].shift()  
  
    # Calculate the cumulative count of each action within each group  
    action_counts = df.groupby(group_by_cols + ['Shifted_Action']).cumcount() + 1  
  
    # Store the action counts in a separate DataFrame  
    action_counts_df = df[group_by_cols + ['Shifted_Action']].copy()  
    action_counts_df['Action_Counts'] = action_counts  
  
    # If processing non-training data, initialize with the last known common actions from the training set  
    if last_common_actions is not None:  
        action_counts_df = action_counts_df.merge(  
            last_common_actions,  
            on=group_by_cols,  
            how='left'  
        )  
        action_counts_df['Action_Counts'] = action_counts_df[['Action_Counts', 'Last_Action_Counts']].max(axis=1)  
  
    # For each row, find the action with the maximum cumulative count up to that point  
    most_common_action = action_counts_df.groupby(group_by_cols + ['Action_Counts'])['Shifted_Action'].transform('last')  
    df[new_col_name] = most_common_action  
  
    # Drop the helper columns  
    df.drop(columns=['Shifted_Action'], inplace=True)  
  
    # If processing training data, store the last known common actions for each group  
    if last_common_actions is None:  
        last_common_actions = action_counts_df.groupby(group_by_cols)['Action_Counts'].last().reset_index()  
        last_common_actions.rename(columns={'Action_Counts': 'Last_Action_Counts'}, inplace=True)  
  
    return df, last_common_actions  


def add_previous_work_end_date(df, group_col='ResourceId', shift_col='WorkEndDate'):  
    df = df.copy()  
    df['PreviousWorkEndDate'] = df.groupby(group_col)[shift_col].shift()
    df['PreviousWorkEndDate'] = pd.to_datetime(df['PreviousWorkEndDate'])
    return df  
  
def add_hours_last_repair(df, date_col='CreatedDate', prev_date_col='PreviousWorkEndDate'):  
    df = df.copy()  
    df['HoursLastRepair'] = (df[date_col] - df[prev_date_col]).dt.total_seconds() / 3600
    df['HoursLastRepair'] = df['HoursLastRepair'].fillna(0)
    return df  
  
def add_device_age(df, created_col='CreatedDate', first_time_col='FirstTimeStamp', max_time_in_train=None):  
    df = df.copy()  
    if max_time_in_train is None:  # Only calculate this for the train set  
        max_time_in_train = (df[created_col] - df[first_time_col]).dt.total_seconds().max() / 3600  
    df['DeviceAge'] = df['DeviceAge'].fillna(max_time_in_train)  
    return df, max_time_in_train  
  
def add_number_previous_repairs(df, group_col='ResourceId', count_col='FaultCode', last_repair_counts=None):  
    df = df.copy()  
    df['NumberPrevRepairs'] = df.groupby(group_col)[count_col].cumcount() + 1  
    if last_repair_counts is not None:  # Only for the test set  
        df['NumberPrevRepairs'] += df[group_col].map(last_repair_counts).fillna(0).astype(int)  
    return df  

def calculate_average_embedding(df, embedding_col='RobertsRulesEmbeddings'):  
    non_null_embeddings = np.stack(df.drop_duplicates(subset=['RobertsRules']).dropna(subset=[embedding_col])[embedding_col])  
    average_embedding = non_null_embeddings.mean(axis=0).tolist()  
    return average_embedding  



def impute_null_embeddings(df, embedding_col='RobertsRulesEmbeddings', average_embedding=None):  
    df = df.copy()  
    if average_embedding is None:  
        average_embedding = calculate_average_embedding(df, embedding_col)  
    df.loc[df[embedding_col].isna(), embedding_col] = df.loc[df[embedding_col].isna(), embedding_col].apply(lambda x: average_embedding)  
    return df, average_embedding  
  
def truncate_embeddings(df, embedding_col='RobertsRulesEmbeddings', length=256):  
    df = df.copy()  
    df[embedding_col] = [emb[:length] if isinstance(emb, list) else emb for emb in df[embedding_col]]  
    return df  

def calculate_cumulative_average_fit(df, group_col, target_col):  
    df = df.copy()  
    df['CumSum'] = df.groupby(group_col)[target_col].cumsum() - df[target_col]  
    df['ExpandingCount'] = df.groupby(group_col).cumcount()  
    df['CumulativeAvgSurvival'] = df['CumSum'] / df['ExpandingCount']  
    df['CumulativeAvgSurvival'] = df.groupby(group_col)['CumulativeAvgSurvival'].fillna(method='ffill')  
  
    # Get the last cumulative values for each group  
    last_cumsum = df.groupby(group_col)['CumSum'].last()  
    last_count = df.groupby(group_col)['ExpandingCount'].last() + 1  
  
    # Drop the intermediate columns  
    df = df.drop(columns=['CumSum', 'ExpandingCount'])  
  
    return df, last_cumsum, last_count  

def calculate_cumulative_average_transform(df, group_col, target_col, last_cumsum, last_count):  
    df = df.copy()  
    df['CumSum'] = df.groupby(group_col)[target_col].cumsum() + df[group_col].map(last_cumsum) - df[target_col]  
    df['ExpandingCount'] = df.groupby(group_col).cumcount() + df[group_col].map(last_count)  
    df['CumulativeAvgSurvival'] = df['CumSum'] / df['ExpandingCount']  
    df['CumulativeAvgSurvival'] = df.groupby(group_col)['CumulativeAvgSurvival'].fillna(method='ffill')  

    # Drop the intermediate columns  
    df = df.drop(columns=['CumSum', 'ExpandingCount'])  
  
    return df  

def calculate_cumulative_frequency_fit(df, group_cols):  
    df = df.copy()  
    df['CumulativeFCFreq'] = df.groupby(group_cols).cumcount() + 1  
    last_fc_counts = df.groupby(group_cols)['CumulativeFCFreq'].last()  
    return df, last_fc_counts  

def calculate_cumulative_frequency_transform(df, group_cols, last_fc_counts):  
    df = df.copy()  
    df['CumulativeFCFreq'] = df.groupby(group_cols).cumcount() + 1  
    df.set_index(group_cols, inplace=True)  
    df['CumulativeFCFreq'] += last_fc_counts.reindex(df.index, fill_value=0).astype(int)  
    df.reset_index(inplace=True)  
    return df  



def calculate_fc_entropy(train_df, specialization_entropy):  
    '''
    high entropy value indicates a more uniform distribution of fault codes, 
    suggesting that there is no single or small set of fault codes that occur significantly more often than the others 
    '''    
    entropy_scores = train_df.groupby('DiscreteSkuMSFId')['FaultCode'].apply(specialization_entropy)  
    return entropy_scores.to_dict()  # Return entropy scores as a dictionary  
  
def apply_entropy_scores(df, entropy_scores):  
    # Map the entropy scores to the DataFrame based on DiscreteSkuMSFId  
    df['FCMSFEntropy'] = df['DiscreteSkuMSFId'].map(entropy_scores)  
    return df  


def device_pipeline(df, is_train=True, max_time_in_train=None, last_repair_counts=None,    
                    average_embedding=None, last_fc_counts=None, last_cumsum=None,  
                    last_count=None, last_common_actions=None, entropy_scores=None):    
    df = df.sort_values(['ResourceId', 'CreatedDate'])  # Ensure the data is sorted    
    
    # common transformations
    if 'WorkEndDate' in df.columns:
        df = add_previous_work_end_date(df)    
        df = add_hours_last_repair(df)

    # train specific        
    if is_train:
        entropy_scores = calculate_fc_entropy(df, specialization_entropy)  

        df, max_time_in_train = add_device_age(df)    
        df = add_number_previous_repairs(df)    
        df, average_embedding = impute_null_embeddings(df)  # Calculate and impute for the train set    
        df, last_fc_counts = calculate_cumulative_frequency_fit(df, ['ResourceId', 'FaultCode'])    
        df, last_common_actions = calculate_most_common_action(df, ['DiscreteSkuMSFId', 'FaultCode'], 'RobertsRules', 'FCMSFRepair')  
        group_col, target_col = 'UserUpdatedBy', 'TicketAgility'  # Example group and target columns  
        df, last_cumsum, last_count = calculate_cumulative_average_fit(df, group_col, target_col)

    else:    
        df, _ = add_device_age(df, max_time_in_train=max_time_in_train)    
        df = add_number_previous_repairs(df, last_repair_counts=last_repair_counts)    
        df, _ = impute_null_embeddings(df, average_embedding=average_embedding)  # Impute using train set's average for the test set    
        df = calculate_cumulative_frequency_transform(df, ['ResourceId', 'FaultCode'], last_fc_counts)    
        df, _ = calculate_most_common_action(df, ['DiscreteSkuMSFId', 'FaultCode'], 'RobertsRules', 'FCMSFRepair', last_common_actions=last_common_actions)  
        group_col, target_col = 'UserUpdatedBy', 'TicketAgility'  # Example group and target columns  
        df = calculate_cumulative_average_transform(df, group_col, target_col, last_cumsum, last_count)  
    
    df = truncate_embeddings(df)  # Truncate embeddings for both train and test sets
    df = apply_entropy_scores(df, entropy_scores)
    
    return df, max_time_in_train, average_embedding, last_fc_counts, last_cumsum, last_count, last_common_actions, entropy_scores

df, last_common_actions = calculate_most_common_action(df, ['DiscreteSkuMSFId', 'FaultCode'], 'RobertsRules', 'FCMSFRepair')  
last_common_actions.to_parquet('./data/last_common_actions.parquet')

# Fit the pipeline on the training set  
train, max_time_in_train, train_average_embedding, last_fc_counts, last_cumsum, last_count, last_common_actions, entropy_scores = device_pipeline(train, is_train=True)  
joblib.dump(train_average_embedding, '/mnt/c/Users/afogarty/Desktop/AZRepairRL/scalars_dict/train_average_embedding.joblib')  
joblib.dump(entropy_scores, '/mnt/c/Users/afogarty/Desktop/AZRepairRL/scalars_dict/entropy_scores.joblib')  

# Save the stateful variables for use with the validation/test sets  
last_repair_counts = train.groupby('ResourceId')['FaultCode'].count()  

# Transform the validation set using the pipeline  
valid, _, _, _, _, _, _, _ = device_pipeline(valid, is_train=False, max_time_in_train=max_time_in_train,  
                                          last_repair_counts=last_repair_counts,  
                                          average_embedding=train_average_embedding,  
                                          last_fc_counts=last_fc_counts,  
                                          last_cumsum=last_cumsum,  
                                          last_count=last_count,  
                                          last_common_actions=last_common_actions,
                                          entropy_scores=entropy_scores)  
  
# Transform the test set using the pipeline  
test, _, _, _, _, _, _, _ = device_pipeline(test, is_train=False, max_time_in_train=max_time_in_train,  
                                         last_repair_counts=last_repair_counts,  
                                         average_embedding=train_average_embedding,  
                                         last_fc_counts=last_fc_counts,  
                                         last_cumsum=last_cumsum,  
                                         last_count=last_count,  
                                         last_common_actions=last_common_actions,
                                         entropy_scores=entropy_scores)  


# top fc hops
train.groupby('FaultCode')['TopFCHops'].first().reset_index().dropna().to_parquet('./data/top_fc_hops.parquet')

print("Device transformations complete!")



# Stateful Technician Transformations
# ====================================================


def calculate_technician_mttr(train_df):      
    mttr_mean = train_df.groupby('UserUpdatedBy')['TicketAgility'].mean()      
    return mttr_mean      
      
def calculate_technician_specialization(train_df, specialization_entropy):      
    specialization_scores = train_df.groupby('UserUpdatedBy')['FaultCode'].apply(specialization_entropy)      
    return specialization_scores      
      
def calculate_technician_success_rate(train_df):      
    success_rate = train_df.groupby('UserUpdatedBy')['MinutesInProduction'].mean()      
    return success_rate      
      
def calculate_repairs_per_device(train_df):      
    repairs_per_device_train = train_df.groupby(['UserUpdatedBy', 'ResourceId']).size()      
    return repairs_per_device_train      
      
def calculate_repair_action_diversity(train_df):      
    repair_action_diversity = train_df.groupby('UserUpdatedBy')['RobertsRules'].nunique()      
    return repair_action_diversity      
      
def calculate_technician_experience_score(train_df):      
    experience_score = train_df.groupby('UserUpdatedBy')['FaultCode'].count()      
    return experience_score      


# The pipeline function  
def technician_pipeline(df, is_train=True, technicians_data=None, specialization_entropy=None):  
    if is_train:  
        # Initialize technicians_data dictionary if not provided  
        if technicians_data is None:  
            technicians_data = {}  
          
        # Calculate and store the technician features for the training set  
        technicians_data['MTTR'] = calculate_technician_mttr(df)  
        technicians_data['SpecializationScore'] = calculate_technician_specialization(df, specialization_entropy)  
        technicians_data['SuccessRate'] = calculate_technician_success_rate(df)  
        technicians_data['RepairActionDiversity'] = calculate_repair_action_diversity(df)  
        technicians_data['ExperienceScore'] = calculate_technician_experience_score(df)  
        technicians_data['RepairsPerDevice'] = calculate_repairs_per_device(df)  
          
        # Compute the median value of 'RepairsPerDevice' from training data for use in validation/test sets  
        technicians_data['RepairsPerDeviceMedian'] = technicians_data['RepairsPerDevice'].median()  
    else:  
        if technicians_data is None:  
            raise ValueError("technicians_data must be provided when is_train is False")  
  
    # Add calculated features to the DataFrame  
    for feature_name in ['MTTR', 'SpecializationScore', 'SuccessRate', 'RepairActionDiversity', 'ExperienceScore']:  
        mapped_feature = df['UserUpdatedBy'].map(technicians_data[feature_name])  
        df[feature_name] = mapped_feature  
  
    # Handle 'RepairsPerDevice' separately as it involves a MultiIndex  
    if 'RepairsPerDevice' in technicians_data:  
        # Ensure the DataFrame has 'UserUpdatedBy' and 'ResourceId' columns  
        if 'UserUpdatedBy' not in df or 'ResourceId' not in df:  
            raise KeyError("DataFrame must contain 'UserUpdatedBy' and 'ResourceId' columns for mapping 'RepairsPerDevice'.")  
  
        # Set the index for mapping 'RepairsPerDevice'  
        df = df.set_index(['UserUpdatedBy', 'ResourceId'])  
          
        # Use the median value of 'RepairsPerDevice' from training data as the default value  
        repairs_per_device_median = technicians_data['RepairsPerDeviceMedian']  
        df['RepairsPerDevice'] = df.index.map(  
            lambda x: technicians_data['RepairsPerDevice'].get(x, repairs_per_device_median)  
        )  
          
        # Reset the index after mapping  
        df.reset_index(inplace=True)  
    else:  
        df['RepairsPerDevice'] = np.nan  
  
    return df, technicians_data  



  
# # Fit the pipeline on the training set  
# # Ensure that the `specialization_entropy` function is passed to the pipeline  
# train, technicians_data = technician_pipeline(train, is_train=True, specialization_entropy=specialization_entropy)  

# # Transform the test set using the calculated data from the training set  
# valid, _ = technician_pipeline(valid, is_train=False, technicians_data=technicians_data, specialization_entropy=specialization_entropy)  
# test, _ = technician_pipeline(test, is_train=False, technicians_data=technicians_data, specialization_entropy=specialization_entropy)  


# technician_feature_columns = [  
#     'MTTR',  
#     'SpecializationScore',  
#     'SuccessRate',  
#     'RepairActionDiversity',  
#     'ExperienceScore',  
#     'RepairsPerDevice'  
# ]

# train[technician_feature_columns]



# Cluster
# ====================================================

# Best Parameters: {'n_neighbors': 21, 'min_dist': 0.08772595184093716, 'n_components': 50, 'spread': 1.3323248474184848,
#                    'metric': 'cosine', 'min_cluster_size': 5, 'min_samples': 4, 'cluster_selection_epsilon': 0.04084128238044429, 
#                    'cluster_selection_method': 'eom'}

def fit_umap_model(train_embeddings):  
    umap_model = umap.UMAP(  
        n_neighbors=21, min_dist=0.08772595184093716, n_components=50,  
        metric='cosine', spread=1.3323248474184848, random_state=42  
    )  
    umap_embeddings_train = umap_model.fit(train_embeddings)  
    return umap_model, umap_embeddings_train.embedding_  
  
def fit_hdbscan_model(umap_embeddings_train):  
    clusterer = hdbscan.HDBSCAN(  
        min_cluster_size=5, min_samples=4, cluster_selection_epsilon=0.04084128238044429,  
        metric='euclidean', cluster_selection_method='eom', prediction_data=True  
    )  
    clusterer.fit(umap_embeddings_train)  
    return clusterer  

def transform_embeddings_and_predict_clusters(umap_model, clusterer, embeddings_2d):  
    umap_embeddings = umap_model.transform(embeddings_2d)  
    labels, strengths = hdbscan.approximate_predict(clusterer, umap_embeddings)  
    return labels  

def clustering_pipeline(df, is_train=True, umap_model=None, clusterer=None, cluster_mapping=None):  
    # Drop duplicates to get one embedding per unique value in 'RobertsRules'  
    unique_df = df.drop_duplicates(subset='RobertsRules')[['RobertsRules', 'RobertsRulesEmbeddings']].dropna(subset=['RobertsRulesEmbeddings'])  
    embeddings_2d = np.stack(unique_df['RobertsRulesEmbeddings'].values)  
  
    if is_train:  
        # Fit UMAP and HDBSCAN on training data  
        umap_model, umap_embeddings_train = fit_umap_model(embeddings_2d)  
        clusterer = fit_hdbscan_model(umap_embeddings_train)  
        train_labels = clusterer.labels_
        print(f"Generated this many labels: {max(clusterer.labels_)}")
          
        # Handle noise by assigning its own category  
        max_value = train_labels.max()  
        train_labels[train_labels == -1] = max_value + 1  
          
        # Create a mapping from unique identifiers to their cluster labels  
        cluster_mapping = {identifier: label for identifier, label in zip(unique_df['RobertsRules'].values, train_labels)}  
        df['ClusterLabel'] = df['RobertsRules'].map(cluster_mapping)  
          
        # Save the cluster mapping  
        with open('./data/embedding_cluster_labels.pickle', 'wb') as handle:  
            pickle.dump(cluster_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    else:  
        # Transform embeddings and predict clusters for valid/test data  
        test_labels = transform_embeddings_and_predict_clusters(umap_model, clusterer, embeddings_2d)  
          
        # Create a mapping from unique identifiers to their cluster labels  
        test_cluster_mapping = {identifier: label for identifier, label in zip(unique_df['RobertsRules'].values, test_labels)}  
        df['ClusterLabel'] = df['RobertsRules'].map(test_cluster_mapping)  
          
        # Assign all unmapped to noise category  
        noise_category = max(cluster_mapping.values()) + 1  
        df['ClusterLabel'] = df['ClusterLabel'].fillna(noise_category)  
        df['ClusterLabel'] = np.where(df['ClusterLabel'] == -1, noise_category, df['ClusterLabel'])  
  
    return df, umap_model, clusterer, cluster_mapping  

# Apply the pipeline to the training data  
train, umap_model, clusterer, cluster_mapping = clustering_pipeline(train, is_train=True)  

# dump
joblib.dump(umap_model, '/mnt/c/Users/afogarty/Desktop/AZRepairRL/scalars_dict/umap_model.joblib')  
joblib.dump(clusterer, '/mnt/c/Users/afogarty/Desktop/AZRepairRL/scalars_dict/clusterer.joblib')  
joblib.dump(cluster_mapping, '/mnt/c/Users/afogarty/Desktop/AZRepairRL/scalars_dict/cluster_mapping.joblib')  


# Apply the pipeline to the validation data  
valid, _, _, _ = clustering_pipeline(valid, is_train=False, umap_model=umap_model, clusterer=clusterer, cluster_mapping=cluster_mapping)  
  
# Apply the pipeline to the test data  
test, _, _, _ = clustering_pipeline(test, is_train=False, umap_model=umap_model, clusterer=clusterer, cluster_mapping=cluster_mapping)  



# ray.init(num_cpus=20, num_gpus=1, ignore_reinit_error=True)
# # Drop duplicates to get one embedding per unique value in 'RobertsRules'  
# unique_train_df = train.drop_duplicates(subset='RobertsRules')[['RobertsRules', 'RobertsRulesEmbeddings']].dropna(subset=['RobertsRulesEmbeddings'])

# # Convert the 'RobertsRulesEmbeddings' column to a 2D NumPy array  
# embeddings_2d = np.stack(unique_train_df['RobertsRulesEmbeddings'].values)  
  
# # Verify the shape of the resulting array  
# print(embeddings_2d.shape) 

# embeddings_2d.mean()  # 0.0008316212345613593
# embeddings_2d.std()  # 0.02505283353680551

# # send to ray
# embeddings_ref = ray.put(embeddings_2d)  

# # Evaluation function for clustering  
# def evaluate_clustering(config):  
#     min_desired_clusters = 30
#     max_desired_clusters = 150
#     silhouette_weight = 2  # Increase the weight of the silhouette score  
#     cluster_factor_weight = 0.5  # Decrease the weight of the cluster factor  
#     density_factor_weight = 0.5  # Decrease the weight of the density factor  
      
#     embeddings = ray.get(embeddings_ref)      
      
#     # Perform dimensionality reduction with UMAP  
#     umap_model = umap.UMAP(  
#         n_neighbors=config['n_neighbors'],  
#         min_dist=config['min_dist'],  
#         n_components=config['n_components'],  
#         spread=config['spread'],  
#         metric=config['metric'],  
#         random_state=42  
#     )  
#     embedding = umap_model.fit_transform(embeddings)  
      
#     # Perform clustering with HDBSCAN  
#     clusterer = hdbscan.HDBSCAN(  
#         min_cluster_size=config['min_cluster_size'],  
#         min_samples=config['min_samples'],
#         metric='euclidean',
#         cluster_selection_epsilon=config['cluster_selection_epsilon'],  
#         cluster_selection_method=config['cluster_selection_method'],
#     )  
#     clusterer.fit(embedding)  
      
#     # Calculate the silhouette score  
#     labels = clusterer.labels_  
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  
#     cluster_sizes = [sum(labels == label) for label in set(labels) if label != -1]  
      
#     if n_clusters >= min_desired_clusters and n_clusters <= max_desired_clusters:    
#         silhouette = silhouette_score(embedding, labels)    
#         density_factor = 1 / np.mean(cluster_sizes) if cluster_sizes else 0    
          
#         # Apply a logarithmic scaling to cluster_factor  
#         cluster_factor = np.log1p(n_clusters / min_desired_clusters) ** cluster_factor_weight  
    
#         # Adjust the combined score calculation    
#         combined_score = (silhouette ** silhouette_weight) * cluster_factor * (density_factor ** density_factor_weight)    
          
#         ray.train.report({  
#             'silhouette': silhouette,  
#             "n_clusters": n_clusters,  
#             "cluster_factor": cluster_factor,  
#             "density_factor": density_factor,  
#             'combined_score': combined_score  
#         })  
#     else:  
#         ray.train.report({  
#             'silhouette': float('-inf'),  
#             "n_clusters": 0,  
#             "cluster_factor": 0,  
#             "density_factor": 0,  
#             'combined_score': float('-inf')  
#         })  
  
# # Define the search space for the hyperparameters  
# search_space = {    
#     # UMAP parameters    
#     'n_neighbors': tune.randint(8, 25),  # Narrowed range around previously good values  
#     'min_dist': tune.uniform(0.04, 0.1),  # Granular exploration in a promising range  
#     'n_components': tune.choice([50, 75, 100, 125, 150]),  # Focus on higher end of range  
#     'spread': tune.uniform(1.3, 1.5),  # Tightened range based on previous results  
#     'metric': tune.choice(['cosine']),  # No change, as only one metric was given  
    
#     # HDBSCAN parameters    
#     'min_cluster_size': tune.randint(5, 31),  # No change, as only one value was given  
#     'min_samples': tune.randint(4, 10),  # Explore higher range for potentially better DBI  
#     'cluster_selection_epsilon': tune.uniform(0.02, 0.05),  # More focused range  
#     'cluster_selection_method': tune.choice(['eom']),  # No change, as only one method was given  
# }  

# search_alg = HyperOptSearch(metric='combined_score', mode='max', points_to_evaluate=None)

# # Run the hyperparameter search  
# analysis = tune.run(  
#     evaluate_clustering,  
#     config=search_space,
#     raise_on_failed_trial=False,
#     search_alg=search_alg,
#     max_concurrent_trials=8,
#     verbose=1,
#     num_samples=200,  # Number of trials to sample from the search space  
#     resources_per_trial={"cpu": 1, "gpu": 0},  # Adjust based on your system's resources  
# )  
  
# # Get the best hyperparameters  
# best_trial = analysis.get_best_trial(metric='combined_score', mode='max')  
# best_params = best_trial.config  
# best_silhouette_score = best_trial.last_result['combined_score']  
  
# print(f"Best Score: {best_silhouette_score}")  
# print(f"Best Parameters: {best_params}")  




# Organize Data
# ====================================================


# Define columns by type
cat_cols = ['FaultCode',
            'DiscreteSkuMSFId',
            'RobertsRules',
            'Cluster',
            'TopFCHops',
            'ClusterLabel',
            'FCMSFRepair',
            ] + ['BusinessGroup', 'CFMAirFlow', 'Depth', 'Weight', 'TargetWorkload', 'Width']

binary_cols = [
    'SP6RootCaused', 'SevRepair', 'KickedFlag', 'InFleet', 'Censored',
]
cont_cols = ['DeviceAge', 'FCMSFEntropy',
             'NumBOMParts', 'NumBOMSubstitutes', 'MaxBOMDepth', 'NumBOMRows', 'NumBlades', 
             'HoursLastRepair', 'NumberPrevRepairs', 'CumulativeAvgSurvival', 'CumulativeFCFreq',
             #"MTTR", "SpecializationScore", "SuccessRate", "RepairActionDiversity", "ExperienceScore", "RepairsPerDevice", 
             ]

label_cols = ['MinutesInProduction']

# Cat NULL Check
# ====================================================

missing_cols_train = train[cat_cols].isna().any()
missing_cols_valid = valid[cat_cols].isna().any()
missing_cols_test = test[cat_cols].isna().any()

# combine
missing_cols = missing_cols_train | missing_cols_test | missing_cols_valid
missing_cols_list = missing_cols[missing_cols].index.tolist()  


# Cont NULL Check
# ====================================================

train[cont_cols].isna().sum()
valid[cont_cols].isna().sum()
test[cont_cols].isna().sum()

# Define a special code that is outside the normal range of the data for 'HoursLastRepair'  
LONG_TIME_VALUE = train['HoursLastRepair'].max() * 10  

# Fill missing values for 'HoursLastRepair' with the special code  
train['HoursLastRepair'].fillna(LONG_TIME_VALUE, inplace=True)
valid['HoursLastRepair'].fillna(LONG_TIME_VALUE, inplace=True)  
test['HoursLastRepair'].fillna(LONG_TIME_VALUE, inplace=True)  

train['MinutesInProduction'].isna().sum()


# Binary/Other NULL Check
# ====================================================

train[binary_cols].isna().sum()
valid[binary_cols].isna().sum()
test[binary_cols].isna().sum()

train['RobertsRulesEmbeddings'].isna().sum()
valid['RobertsRulesEmbeddings'].isna().sum()
test['RobertsRulesEmbeddings'].isna().sum()


# Encode
# ====================================================


# Initialize dictionaries  
encoders_dict = {}  
EMBEDDING_TABLE_SHAPES = {}  
PADDING_INDEX = 0  
embedding_size = 25  # Replace with the actual embedding size you want  

for column in cat_cols:  
    # Replace missing values with a unique string  
    train[column] = train[column].fillna('NAN_NULL')  
    valid[column] = valid[column].fillna('NAN_NULL')  
    test[column] = test[column].fillna('NAN_NULL')  
      
    # Convert all non-missing values to strings  
    train[column] = train[column].astype(str)  
    valid[column] = valid[column].astype(str)  
    test[column] = test[column].astype(str)  
      
    # Initialize the encoder without specifying handle_unknown  
    encoder = OrdinalEncoder()  
    encoder.fit(train[[column]])  
      
    # Determine the number of unique categories  
    num_unique_categories = len(encoder.categories_[0])  
  
    # Create a dictionary mapping each category to its encoded value  
    category_mapping = {category: index for index, category in enumerate(encoder.categories_[0], start=1)}  
      
    # Define a function to encode categories, with a special index for unseen categories  
    def encode_category(category):  
        return category_mapping.get(category, num_unique_categories + 1)  
      
    # Apply the encoding function to each column  
    train[column] = train[column].apply(encode_category)  
    valid[column] = valid[column].apply(encode_category)  
    test[column] = test[column].apply(encode_category)  
      
    # Check for unseen categories represented as 0 after incrementing  
    assert 0 not in train[column].unique(), "Encoded training data contains 0 after incrementing, which is reserved for padding."  
    assert 0 not in valid[column].unique(), "Encoded validation data contains 0 after incrementing, which is reserved for padding."  
    assert 0 not in test[column].unique(), "Encoded test data contains 0 after incrementing, which is reserved for padding."  
      
    # Store the fitted encoder in the dictionary  
    encoders_dict[column] = encoder  
      
    # Define embedding table shapes  
    # The embedding table should have a row for each unique category plus one for unknown categories  
    unknown_category_index = num_unique_categories + 1  
    EMBEDDING_TABLE_SHAPES[column] = (unknown_category_index + 1, embedding_size)  

EMBEDDING_TABLE_SHAPES

# storage
imputers_dict = {}

# Assuming cont_cols is a list of your continuous column names  
missing_cols_test = test[cont_cols].isna().any()  
missing_cols_valid = valid[cont_cols].isna().any()  
  
# Combine the boolean Series using bitwise OR to identify columns with missing values in either DataFrame  
missing_cols = missing_cols_test | missing_cols_valid  
  
# missing_cols will be a boolean Series where True indicates missing values in a column  
# To get just the names of the columns with missing values, you can use:  
missing_cols_list = missing_cols[missing_cols].index.tolist()  

# Now, missing_cols contains the names of continuous columns that have NULL values  
print("Columns with missing values:", missing_cols)  

# Loop over the features with missing values  
for feature in missing_cols_list:  
    # Choose an imputation strategy  
    # For simplicity, we are using median imputation here  
    imputer = SimpleImputer(strategy='median')  
  
    # Fit the imputer on the training data and transform both training and test sets  
    imputer.fit(train[[feature]])  
    train[feature] = imputer.transform(train[[feature]])
    valid[feature] = imputer.transform(valid[[feature]])  
    test[feature] = imputer.transform(test[[feature]])  
  
    # Store the fitted imputer  
    imputers_dict[feature] = imputer  


# Dictionary to store the scalers for each continuous column    
scalers_dict = {}  

# Loop over the continuous columns to fit and transform using separate scalers    
for column in cont_cols:    

    # For other columns, use the regular StandardScaler    
    scaler = StandardScaler()    
        
    # Fit the scaler on the training data for the column  
    scaler.fit(train[[column]])    
        
    # Scale the training data for the column  
    train[column] = scaler.transform(train[[column]])  
        
    # Scale the test data for the column
    valid[column] = scaler.transform(valid[[column]])  
    test[column] = scaler.transform(test[[column]])  
        
    # Store the fitted scaler in the dictionary    
    scalers_dict[column] = scaler  


# Create the log1p transformer  
log1p_scaler = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)  
  
# Apply the log1p transform to all data  
train[label_cols] = log1p_scaler.fit_transform(train[label_cols])  
valid[label_cols] = log1p_scaler.transform(valid[label_cols])  
test[label_cols] = log1p_scaler.transform(test[label_cols])  
  
# Store the fitted transformer  
scalers_dict['MinutesInProduction'] = log1p_scaler  


# resort inversion
# Identify the groups that contain "_inverted" in the specified column  
groups_to_invert = train[train['ResourceId'].str.contains('_inverted')]['ResourceId'].unique()  
  
# Create a boolean mask for rows that belong to groups that should be inverted  
mask = train['ResourceId'].isin(groups_to_invert)  
  
# Invert the order of rows for groups that need inversion  
train.loc[mask, 'GroupSortHelper'] = train[mask].groupby('ResourceId').cumcount(ascending=False)  
train.loc[~mask, 'GroupSortHelper'] = train[~mask].groupby('ResourceId').cumcount()  
  
# Sort the DataFrame based on the helper column and GroupColumn  
train = train.sort_values(by=['ResourceId', 'GroupSortHelper'])  
  
# Drop the helper column as it's no longer needed  
train = train.drop(columns=['GroupSortHelper']).reset_index(drop=True)  


# save
if point_wise:
    train[cat_cols + binary_cols + cont_cols + label_cols + ['ResourceId', 'ResourceIdOriginal', 'sequence_type', 'test_split', 'CreatedDate', 'RobertsRulesEmbeddings']] \
                        .to_parquet('./data/seq_azrepair_train.parquet')

    valid[cat_cols + binary_cols + cont_cols + label_cols + ['ResourceId', 'test_split', 'CreatedDate', 'RobertsRulesEmbeddings']] \
                        .to_parquet('./data/seq_azrepair_valid.parquet')

    test[cat_cols + binary_cols + cont_cols + label_cols + ['ResourceId', 'test_split', 'CreatedDate', 'RobertsRulesEmbeddings' ]] \
                        .to_parquet('./data/seq_azrepair_test.parquet')
    



if not point_wise:


    train[cat_cols + binary_cols + cont_cols + label_cols + ['ResourceId', 'test_split', 'CreatedDate', 'RobertsRulesEmbeddings']] \
                        .to_parquet('./data/pairwise_azrepair_train.parquet')

    valid[cat_cols + binary_cols + cont_cols + label_cols + ['ResourceId', 'test_split', 'CreatedDate', 'RobertsRulesEmbeddings']] \
                        .to_parquet('./data/pairwise_azrepair_valid.parquet')

    test[cat_cols + binary_cols + cont_cols + label_cols + ['ResourceId', 'test_split', 'CreatedDate', 'RobertsRulesEmbeddings' ]] \
                        .to_parquet('./data/pairwise_azrepair_test.parquet')    


# Save scalers to a file  
joblib.dump(scalers_dict, '/mnt/c/Users/afogarty/Desktop/AZRepairRL/scalars_dict/scalars_dict.joblib')  
joblib.dump(encoders_dict, '/mnt/c/Users/afogarty/Desktop/AZRepairRL/scalars_dict/encoders_dict.joblib')  
joblib.dump(imputers_dict, '/mnt/c/Users/afogarty/Desktop/AZRepairRL/scalars_dict/imputers_dict.joblib')  

print(EMBEDDING_TABLE_SHAPES)




