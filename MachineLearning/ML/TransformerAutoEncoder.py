#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch  
import torch.nn as nn  
from tqdm.auto import tqdm
from torch.utils.data import Dataset  
import torch 
import pandas as pd 
from torch.utils.data import DataLoader  
from accelerate import Accelerator
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from torch.optim.lr_scheduler import _LRScheduler  
import math
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import nmslib
torch.backends.cuda.matmul.allow_tf32 = True


# In[16]:


# Central Question: what drives the repeat?

# load mtx
df = pd.read_parquet('data/mtx_full_for_embeds.parquet')
df = df.sort_values(by=['ResourceId', 'FromTimeStamp'], ascending=True).reset_index(drop=True)  

# convert arr
df['RepeatedFaultCodes'] = df['RepeatedFaultCodes'].apply(lambda x: ','.join(map(str, x)))  
df['RepeatedFaultCodes'] = df['RepeatedFaultCodes'].replace('', np.nan)

# reduce cardinality
# Calculate the frequency of each category  
frequency = df['RepeatedFaultCodes'].value_counts(normalize=True)  

# Identify categories that are below the threshold  
rare_categories = frequency[frequency < 0.0001].index  

# result
print(f"Number unique RFC Cats: {df['RepeatedFaultCodes'].nunique() - len(rare_categories)}")

# Replace rare categories with 'Other'  
df['RepeatedFaultCodes'] = df['RepeatedFaultCodes'].apply(lambda x: 'Other' if x in rare_categories else x)  

# Generate timesteps  
df['TimeStep'] = df.groupby(['ResourceId']).cumcount() + 1  


# In[17]:


# Set test/train split    
np.random.seed(42)    
unique_devices = df['ResourceId'].unique()    
np.random.shuffle(unique_devices)    
split_idx = int(len(unique_devices) * 0.9)    
train_devices = unique_devices[:split_idx]    
test_devices = unique_devices[split_idx:]    
device_split_mapping = {device: 'train' for device in train_devices}    
device_split_mapping.update({device: 'test' for device in test_devices})    
df['test_split'] = df['ResourceId'].map(device_split_mapping)


# split
df = df.drop(['FromTimeStamp','ToTimeStamp'], axis=1)
train = df.query('test_split == "train" ')
test = df.query('test_split == "test" ')


# Define columns by type 
cat_cols = [
            'FromState',
            'ToState',
            'rackAssembly',
            'discreteSKUName',
            'Generation',
            'clusterType',
            'RepeatedFaultCodes',
                ]

# empty for now
binary_cols = []
cont_cols = []


# Dictionary to store the encoders for each column  
encoders_dict = {}  
EMBEDDING_TABLE_SHAPES = {}  
  
for column in cat_cols:  
    # Fit the encoder on the training data only  
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)  
    encoder.fit(train[[column]])  
  
    # Transform the training and testing data, incrementing by 1 to reserve 0 for padding  
    train[column] = encoder.transform(train[[column]]) + 1  
    test[column] = encoder.transform(test[[column]]) + 1  
  
    # Store the fitted encoder in the dictionary  
    encoders_dict[column] = encoder  
  
    # Determine the number of unique categories in the training data  
    num_unique_categories = len(encoder.categories_[0])  
  
    # For columns with NaN values, add 2 to account for padding and unknown  
    # For columns without NaN values, add 1 to account only for padding  
    if train[column].isnull().any():  
        num_unique_categories += 2  
    else:  
        num_unique_categories += 1  
  
    # Define embedding table shapes  
    EMBEDDING_TABLE_SHAPES[column] = (num_unique_categories, 100)  # embedding_dim is the size of the embedding vector  

# Save scalers to a file  
joblib.dump(encoders_dict, './encoders/encoders_dict.joblib')  

# set time steps
max_time_step = train['ResourceId'].value_counts().quantile(0.95).astype(int)
print(f'max time step: {max_time_step}')


# In[21]:


# pad fn
def pad_dataframe(df, cat_cols, binary_cols, cont_cols, max_time_step):    
    # Ensure the DataFrame is sorted by 'ResourceId' and 'TimeStep' in ascending order    
    df_sorted = df.sort_values(by=['ResourceId', 'TimeStep'], ascending=[True, True])    
        
    # Create a MultiIndex with all possible combinations of 'ResourceId' and 'TimeStep'    
    multi_index = pd.MultiIndex.from_product(    
        [df_sorted['ResourceId'].unique(), range(1, max_time_step + 1)],    
        names=['ResourceId', 'TimeStep']    
    )    
        
    # Reindex the sorted DataFrame to have a complete MultiIndex with padding where necessary    
    full_data = df_sorted.set_index(['ResourceId', 'TimeStep']).reindex(multi_index)    
        
    # Fill NaNs with padding values for the different types of columns
    padding_values = {**{col: 0 for col in binary_cols + cat_cols}, **{col: 0.0 for col in cont_cols}}    
    full_data.fillna(padding_values, inplace=True)    
        
    # Create the mask based on the presence of a valid 'test_split'    
    # If 'test_split' is NaN, it means the row is padding, so the mask is False    
    # If 'test_split' is not NaN, it means the row is valid, so the mask is True    
    full_data['Mask'] = full_data['test_split'].notna()    
        
    # Optionally, sort the index to have 'ResourceId' and 'TimeStep' in ascending order    
    full_data.sort_index(inplace=True)    
        
    # Reset the index if you want 'ResourceId' and 'TimeStep' back as columns    
    full_data.reset_index(inplace=True)    
        
    return full_data  


# pad dfs
train = pad_dataframe(train, cat_cols, binary_cols, cont_cols, max_time_step)
test = pad_dataframe(test, cat_cols, binary_cols, cont_cols, max_time_step)

# fill
train['test_split'] = train['test_split'].ffill()
test['test_split'] = test['test_split'].ffill()

# dtypes
for col in cat_cols + binary_cols:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)

train['Mask'] = train['Mask'].astype(bool)
test['Mask'] = test['Mask'].astype(bool)
print('dtypes finished!')

# checks
group_sizes = train.groupby('ResourceId').size()  
all_groups_of_size_n = all(group_sizes == max_time_step)  
  
if all_groups_of_size_n:  
    print(f"All groups are of size {max_time_step}.")  
else:  
    print(f"Not all groups are of size {max_time_step}.")  
    # Optional: print the ResourceIds that do not have the expected size  
    print(group_sizes[group_sizes != max_time_step])  


# Group the data by 'ResourceId' and check for groups with no 'True' in the 'Mask' column  
mask_check = train.groupby('ResourceId')['Mask'].any()  
# Get the list of ResourceId values where all mask entries are False  
resource_ids_with_all_padding = mask_check.index[mask_check == False].tolist()  
# Print or return the ResourceIds with all padding  
print("ResourceIds with all padding:", len(resource_ids_with_all_padding))  


train[cat_cols + ['ResourceId', 'test_split', 'Mask']].to_parquet('./data/autoenc_train.parquet')
test[cat_cols + ['ResourceId', 'test_split', 'Mask']].to_parquet('./data/autoenc_test.parquet')


# In[22]:


EMBEDDING_TABLE_SHAPES


# In[11]:


class GroupedDataFrameDataset(Dataset):  
    def __init__(self, dataframe, group_key, cat_cols, binary_cols, cont_cols, mask_col, label_col, embedding_col=None, embedding_dim_size=0):  
        # Store the original dataframe  
        self.dataframe = dataframe 
        # Store column names  
        self.cat_cols = cat_cols  
        self.binary_cols = binary_cols  
        self.cont_cols = cont_cols  
        self.mask_col = mask_col  
        self.label_col = label_col  
        self.embedding_col = embedding_col
        self.embedding_dim_size = embedding_dim_size
        # set key
        self.group_key = group_key
        # Store the indices for each group  
        self.group_indices = list(dataframe.groupby(self.group_key).indices.values())
        # Create a mapping from ResourceId to group index  
        self.resource_to_index = {resource: idx for idx, (resource, _) in enumerate(dataframe.groupby(group_key))}  

  
    def __len__(self):  
        # The length of the dataset is the number of unique groups  
        return len(self.group_indices)  

    def get_group_index(self, resource_id):  
        """  
        Returns the group index for the given ResourceId.  

        """  
        return self.resource_to_index.get(resource_id, None)  # Returns None if ResourceId is not found  
  
        # If the ResourceId is not associated with any group (this should not happen), return None  
        return None  

    def get_group_key(self, idx):
        '''
        Return ResourceId for a given idx group value
        '''
        # Assuming that each group has a unique identifier (ResourceId)  
        return self.dataframe.iloc[self.group_indices[idx]][self.group_key].unique()[0]  
  
    def get_original_data(self, group_key):  
        # Retrieve original data based on the group key  
        return self.dataframe[self.dataframe[self.group_key] == group_key]  

    def __getitem__(self, idx):  
        # Fetch the indices for the group  
        indices = self.group_indices[idx]  
        # Select data for the group  
        group_data = self.dataframe.iloc[indices]  

        # Use the get_group_key method to fetch the actual group key value  
        actual_group_key = self.get_group_key(idx)  

        # Process categorical and binary columns  
        x_cat = None  
        if self.cat_cols:  
            x_cat = torch.tensor(group_data[self.cat_cols].values, dtype=torch.long)  
            # Check that all categorical columns are within the valid range  
            for col in self.cat_cols:  
                if col in EMBEDDING_TABLE_SHAPES:  
                    num_classes, _ = EMBEDDING_TABLE_SHAPES[col]  
                    if not ((x_cat[:, self.cat_cols.index(col)] >= 0) &  
                            (x_cat[:, self.cat_cols.index(col)] < num_classes)).all():  
                        raise ValueError(f"Column '{col}' contains values out of range [0, {num_classes - 1}]")  

        x_bin = torch.tensor(group_data[self.binary_cols].values, dtype=torch.long) if self.binary_cols else None  
    
        # Initialize a list to collect all continuous features  
        continuous_features = []  
    
        # Process continuous columns  
        if self.cont_cols:  
            # Extract continuous features as a tensor  
            cont_features = torch.tensor(group_data[self.cont_cols].values, dtype=torch.float)  
            continuous_features.append(cont_features)  
    
        # Handle embeddings, filling in missing values with zero vectors  
        if self.embedding_col:  
            # Initialize a tensor to hold the embeddings, filled with zeros  
            embeddings = torch.zeros((len(indices), self.embedding_dim_size), dtype=torch.float)  
            # Iterate over the group data and replace zeros with actual embeddings where they exist  
            for i, emb in enumerate(group_data[self.embedding_col]):  
                if isinstance(emb, np.ndarray):  
                    embeddings[i] = torch.tensor(emb, dtype=torch.float)  
            continuous_features.append(embeddings)  
    
        # Concatenate all continuous features along the last dimension  
        x_cont = torch.cat(continuous_features, dim=-1) if continuous_features else None  
    
        # Process mask and labels  
        mask = torch.tensor(group_data[self.mask_col].values, dtype=torch.bool) if self.mask_col else None  
    
        return {  
            'x_cat': x_cat,  
            #'x_bin': x_bin,  
            #'x_cont': x_cont,  
            'mask': mask,
            'ResourceId': actual_group_key,
            'group_idx': torch.tensor([idx], dtype=torch.long)  # Wrap idx in a list to keep dimension  
           
        }  


# In[12]:


# load dfs
train = pd.read_parquet('./data/autoenc_train.parquet')
test = pd.read_parquet('./data/autoenc_test.parquet')

# emb shape
EMBEDDING_TABLE_SHAPES = {'FromState': (19, 100),
 'ToState': (19, 100),
 'rackAssembly': (1303, 100),
 'discreteSKUName': (517, 100),
 'Generation': (54, 100),
 'clusterType': (14, 100),
 'RepeatedFaultCodes': (805, 100)}


cat_cols = list(EMBEDDING_TABLE_SHAPES.keys())

# torch sets
train_set = GroupedDataFrameDataset(  
    dataframe=train,
    group_key='ResourceId',
    cat_cols=cat_cols,  
    binary_cols=None,  
    cont_cols=None,  
    mask_col='Mask',  
    embedding_col=None,
    label_col=None
)  

test_set = GroupedDataFrameDataset(  
    dataframe=test,
    group_key='ResourceId',
    cat_cols=cat_cols,  
    binary_cols=None,  
    cont_cols=None,  
    mask_col='Mask',  
    embedding_col=None,
    label_col=None
)  


batch_size = 1024
train_dataloader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=16,
    prefetch_factor=2,
    drop_last=True,
)

eval_dataloader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=16,
    prefetch_factor=2,
    drop_last=False,
)


# In[20]:


class EntityEmbeddingLayer(nn.Module):  
    def __init__(self, embedding_table_shapes):  
        super().__init__()  
        self.embeddings = nn.ModuleDict({  
            col: nn.Embedding(num_embeddings=cat_size, embedding_dim=emb_size, padding_idx=0)  
            for col, (cat_size, emb_size) in embedding_table_shapes.items()  
        })  
        self.embedding_table_shapes = embedding_table_shapes  
  
    def forward(self, x):  
        # x is expected to have dimensions [batch_size, sequence_length, num_categorical_features]  
        batch_size, sequence_length, num_cat_features = x.shape  
        assert num_cat_features == len(self.embedding_table_shapes), "Mismatch in number of categorical features and embeddings."  

        embeddings = [self.embeddings[col](x[:, :, i]) for i, col in enumerate(self.embedding_table_shapes)]  
        x = torch.cat(embeddings, dim=2)  # Concatenate along the last dimension (features)  
        return x  
    

class EntityOutputLayer(nn.Module):  
    def __init__(self, feature_cardinalities, input_dim):  
        super().__init__()  
        self.output_layers = nn.ModuleDict({  
            col: nn.Linear(input_dim, num_classes)  
            for col, num_classes in feature_cardinalities.items()  
        })  
        self.feature_cardinalities = feature_cardinalities  
  
    def forward(self, x):  
        # x is expected to have dimensions [batch_size, sequence_length, input_dim]  
        batch_size, sequence_length, input_dim = x.shape  
        assert input_dim == next(iter(self.output_layers.values())).in_features, "Input dimension must match output layer input features."  
  
        # Compute the logits for each categorical feature using the corresponding output layer  
        logits_dict = {col: output_layer(x)  
                       for col, output_layer in self.output_layers.items()}  
  
        # Concatenate the logits for all features along the last dimension  
        logits = torch.cat(list(logits_dict.values()), dim=2)  
  
        return logits  


# Multi-Layer Perceptron for feature projection  
class MLP(nn.Module):  
    def __init__(self, input_size, output_size, hidden_layers, dropout_rate):  
        super(MLP, self).__init__()  
        layers = []  
        for hidden_size in hidden_layers:  
            layers.append(nn.Linear(input_size, hidden_size))  
            layers.append(nn.ReLU())  
            layers.append(nn.Dropout(dropout_rate))  
            input_size = hidden_size  
        layers.append(nn.Linear(hidden_layers[-1], output_size))  
        self.mlp = nn.Sequential(*layers)  
      
    def forward(self, x):  
        return self.mlp(x)  


class WarmupCosineAnnealingLR(_LRScheduler):    
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr_fraction=0.01, last_epoch=-1):    
        self.warmup_epochs = warmup_epochs    
        self.max_epochs = max_epochs - warmup_epochs    
        self.min_lr_fraction = min_lr_fraction  # Minimum fraction of the base_lr  
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)    
        
    def get_lr(self):    
        if self.last_epoch < self.warmup_epochs:    
            # Warmup: Increase from a fraction of base_lr to base_lr over warmup_epochs  
            warmup_factor = ((1 - self.min_lr_fraction) * (self.last_epoch / self.warmup_epochs) + self.min_lr_fraction)  
            return [base_lr * warmup_factor for base_lr in self.base_lrs]    
        else:    
            # Cosine annealing  
            cos_epoch = self.last_epoch - self.warmup_epochs    
            return [base_lr * (1 + math.cos(math.pi * cos_epoch / self.max_epochs)) / 2    
                    for base_lr in self.base_lrs]  

  

class LearnedPositionalEncoding(nn.Module):  
    def __init__(self, dim_model, max_len=5000):  
        super(LearnedPositionalEncoding, self).__init__()  
        self.encoding = nn.Parameter(torch.zeros(max_len, dim_model))  
        nn.init.normal_(self.encoding)  
  
    def forward(self, token_embeddings):  
        batch_size, seq_length, _ = token_embeddings.size()  
        encoding = self.encoding[:seq_length, :].unsqueeze(0).expand(batch_size, -1, -1)  
        return token_embeddings + encoding  
  

class TabularTransformerAutoEncoder(nn.Module):    
    def __init__(self, config, embedding_table_shapes, num_binary=0, num_continuous=0):    
        super(TabularTransformerAutoEncoder, self).__init__()    
            
        # Configuration parameters    
        max_sequence_length = config.get("max_sequence_length", 512)    
        dim_model = config.get("dim_model", 1024)    
        nhead = config.get("num_heads", 8)    
        dim_feedforward = config.get("dim_feedforward", 2048)    
        dropout_rate = config.get("dropout", 0.1)  # Renamed to dropout_rate for clarity  
        n_layers = config.get("n_layers", 3)    
        activation_function = config.get("activation_function", "relu")    
        hidden_layers = config.get("mlp_hidden_layers", [512])  

        feature_cardinalities = {k: v[0] for k, v in embedding_table_shapes.items()}
  
        # Entity Embedding for categorical features    
        self.embedding_layer = EntityEmbeddingLayer(embedding_table_shapes)    
        self.embedding_size = sum(emb_size for _, emb_size in embedding_table_shapes.values())      

        # Dropout layer    
        self.dropout = nn.Dropout(dropout_rate)  # This line defines the dropout layer  

        # Include binary and continuous data sizes    
        self.num_binary = num_binary    
        self.num_continuous = num_continuous    
            
        # Normalization for continuous features    
        if num_continuous > 0:    
            self.continuous_ln = nn.LayerNorm(num_continuous)    
            
        # Total input size calculation    
        total_input_size = self.embedding_size + num_binary + num_continuous
        print(f"Total Input Size: {total_input_size}")   

        # MLP for feature projection    
        self.feature_projector = MLP(total_input_size, dim_model, hidden_layers, dropout_rate)    
           
        # Dynamic Positional Encoding (unchanged, assuming you use learned positional encoding)    
        self.pos_encoder = LearnedPositionalEncoding(dim_model=dim_model, max_len=max_sequence_length)    
                  
        # Custom Transformer Encoder Layer with LayerNorm    
        encoder_layer = nn.TransformerEncoderLayer(    
            d_model=dim_model,    
            nhead=nhead,    
            dim_feedforward=dim_feedforward,    
            dropout=dropout_rate,    
            activation=activation_function,    
            batch_first=True,    
            norm_first=True  # Add LayerNorm to the beginning of the sublayers  
        )    
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)    

        # Initialize the custom output layer  
        self.entity_output_layer = EntityOutputLayer(feature_cardinalities, input_dim=dim_model)  
  
  
    def forward(self, x_cat, x_bin=None, x_cont=None, mask=None, return_embeddings=False):    
        x_cat = self.embedding_layer(x_cat)    
        x_cat = self.dropout(x_cat)    
    
        x_additional = []    
        if x_bin is not None:    
            x_additional.append(x_bin)    
            
        if x_cont is not None:    
            x_cont = self.continuous_ln(x_cont)    
            x_additional.append(x_cont)    
            
        if x_additional:    
            x_additional = torch.cat(x_additional, dim=2)    
            x = torch.cat((x_cat, x_additional), dim=2)    
        else:    
            x = x_cat    
            
        # Feature projection through MLP    
        x = self.feature_projector(x)    
            
        # Apply Positional Encoding    
        x = self.pos_encoder(x)    
            
        # Masking for Transformer Encoder    
        # mask is [batch_size, seq_len, 1] -- where TRUE is valid data and FALSE is padding
        if mask is not None:    
            encoder_mask = ~mask.squeeze(-1)  
        else:    
            encoder_mask = None

        # now mask is [batch-size, seq_len]
            
        # Apply Transformer Encoder    
        x = self.transformer_encoder(x, src_key_padding_mask=encoder_mask)
    
        # Check if we want to return the pooled embeddings  
        if return_embeddings:  
            pooled_embeddings = x.mean(dim=1)  
            return pooled_embeddings  
  
        # Use the custom output layer to get logits for each categorical feature  
        logits_dict = self.entity_output_layer(x)  
  
        # Return the logits dictionary, where each key corresponds to a categorical feature  
        return logits_dict  
  

max_seq_len = train.groupby(['ResourceId']).size().max().astype(int)
#max_seq_len = 46



config = {  
    "max_sequence_length": max_seq_len,  # The maximum length of the input sequences  
    "dim_model": 512,           # The dimensionality of the input embeddings and model  
    "num_heads": 2,              # The number of attention heads in the Transformer encoder  
    "dim_feedforward": 768,     # The dimensionality of the feedforward network model in the Transformer encoder  
    "dropout": 0.057,              # The dropout rate used in the Transformer encoder and other dropout layers  
    "n_layers": 2,               # The number of Transformer encoder layers  
    "activation_function": "gelu", # The activation function to use: "relu", "gelu", "leaky_relu", etc.  
    "emb_dropout": 0.036,          # The dropout rate applied to the embedding layers  
    "mlp_hidden_layers": [512],
}  


# config, embedding_table_shapes, emb_dropout, num_binary=0, num_continuous=0
model = TabularTransformerAutoEncoder(config=config,
                            embedding_table_shapes=EMBEDDING_TABLE_SHAPES,
                            num_binary=0,
                            num_continuous=0,
                            )
  


# In[23]:


# Initialize accelerator
accelerator = Accelerator(
    deepspeed_plugin=None,
    gradient_accumulation_steps=1,
    mixed_precision="fp16",
)


# In[27]:


num_epochs = 12
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.001)  
scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=1, max_epochs=num_epochs, min_lr_fraction=0.1)  
criterion = nn.CrossEntropyLoss()  

# prepare
model, optimizer, train_dataloader, eval_dataloader, scheduler, criterion = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, scheduler, criterion)


def evaluate(model, eval_loader):
    model.eval()
    running_loss_sum = 0.0  # Variable to track the sum of losses over all steps  

  
    with torch.no_grad():  # Move the no_grad context to cover the entire loop  
        for step, batch in tqdm(enumerate(eval_loader), total=len(eval_loader)):  
            
            # Forward pass  
            logits = model(x_cat=batch["x_cat"],
                        mask=batch['mask'],
                            )
            
            target = batch["x_cat"]  # torch.Size([512, 46, 6])
            mask = batch['mask']  #  torch.Size([512, 46])

            # Calculate the loss for each feature and sum them up  
            loss = 0  
            start_index = 0  
            for i, cardinality in enumerate(feature_cardinalities.values()):  
                # Calculate the end index for the features in the flat representation  
                end_index = start_index + cardinality  

                # Select the appropriate slice from the reconstructed output  
                reconstructed_feature = logits[:, :, start_index:end_index]  

                # Apply the mask to filter out the padded values and flatten the output  
                valid_reconstructed = reconstructed_feature[mask].view(-1, cardinality)  
                valid_reconstructed.shape  # torch.Size([9, 10])
                valid_target = target[:, :, i][mask].view(-1)  
                valid_target.shape  # torch.Size([9])

                # Calculate loss for the current feature  
                feature_loss = criterion(valid_reconstructed, valid_target)  
                loss += feature_loss  

                # Move the start index to the next feature slice  
                start_index = end_index  

            # Average the loss over the number of features if necessary  
            loss /= len(feature_cardinalities)  
        
            loss_item = loss.item()  
            running_loss_sum += loss_item  

    # average across batches
    average_loss = running_loss_sum / len(eval_loader)  

    # Calculate mean loss and accuracy  
    return average_loss  



feature_cardinalities = {k: v[0] for k, v in EMBEDDING_TABLE_SHAPES.items()}
best_eval_loss = 999.9

# loop
for epoch in range(1, num_epochs + 1):
    model.train()

    running_loss_sum = 0.0
    last_hundred_loss_sum = 0.0

    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optimizer.zero_grad()       
        
        # Forward pass  
        logits = model(x_cat=batch["x_cat"],
                    mask=batch['mask'],
                        )
                
        target = batch["x_cat"]
        mask = batch['mask']

        # Calculate the loss for each feature and sum them up  
        loss = 0  
        start_index = 0  
        for i, cardinality in enumerate(feature_cardinalities.values()):  
            # Calculate the end index for the features in the flat representation  
            end_index = start_index + cardinality  

            # Select the appropriate slice from the reconstructed output  
            reconstructed_feature = logits[:, :, start_index:end_index]  

            # Apply the mask to filter out the padded values and flatten the output  
            valid_reconstructed = reconstructed_feature[mask].view(-1, cardinality)  
            valid_reconstructed.shape  # torch.Size([9, 10])
            valid_target = target[:, :, i][mask].view(-1)  
            valid_target.shape  # torch.Size([9])

            # Calculate loss for the current feature  
            feature_loss = criterion(valid_reconstructed, valid_target)  
            loss += feature_loss  

            # Move the start index to the next feature slice  
            start_index = end_index  

        # Average the loss over the number of features if necessary  
        loss /= len(feature_cardinalities)  

        loss_item = loss.item()  
        running_loss_sum += loss_item  
        last_hundred_loss_sum += loss_item  # Accumulate loss for the last 100 steps  
        
        # Backward pass  
        accelerator.backward(loss)
        optimizer.step()  

        # Report progress  
        if (step + 1) % 1000 == 0:
            average_loss_last_hundred = last_hundred_loss_sum / 100
    
            if accelerator.is_main_process:  
                accelerator.print(  
                    f"[epoch {epoch} step {step} "  
                    f"average loss: {average_loss_last_hundred:.4f} | " 
                    f"current lr: {scheduler.get_lr()[0]:.8f} ]" 

                )
            last_hundred_loss_sum = 0.0  # Reset the sum for the next 100 steps  

    # eval
    eval_loss = evaluate(model, eval_dataloader)

    # average across batches
    average_loss = running_loss_sum / len(train_dataloader)  

    # epoch step
    scheduler.step()

    # save
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss

        if accelerator.is_main_process:
            state = accelerator.get_state_dict(model)
            accelerator.save(state, f"./autoencoder_ckpt/epoch_{epoch}_autoencoder_state_dict.pt")      

    print(f"Epoch {epoch}/{num_epochs}, Loss: {average_loss} | Eval Loss: {eval_loss}")  


# In[15]:


# combine dfs
all_embeddings_df = pd.concat([train, test], axis=0)


# In[16]:


embedding_ds = GroupedDataFrameDataset(  
    dataframe=all_embeddings_df,
    group_key='ResourceId',
    cat_cols=cat_cols,  
    binary_cols=None,  
    cont_cols=None,  
    mask_col='Mask',  
    embedding_col=None,
    label_col=None
)

batch_size = 1024
embedding_dataloader = DataLoader(
    embedding_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=16,
    prefetch_factor=2,
    drop_last=False,
)


# In[18]:


import gc
del df;
del train;
del test;
gc.collect()


# In[21]:


# load fine tuned autoencoder
state_dict = torch.load('./autoencoder_ckpt/epoch_12_autoencoder_state_dict.pt', map_location='cuda:0')
model.load_state_dict(state_dict)
model.eval();


# In[24]:


# prepare
model, embedding_dataloader = accelerator.prepare(model, embedding_dataloader)


# In[27]:


# Store embeddings and their corresponding group indices in a list  
embedding_store = []  
group_indices_store = []  # To keep track of group indices  
group_resourceid_store = []

for step, batch in tqdm(enumerate(embedding_dataloader), total=len(embedding_dataloader)):  
    with torch.no_grad():  
        # Forward pass to get embeddings  
        embeddings = model(x_cat=batch["x_cat"],  
                           mask=batch['mask'],  
                           return_embeddings=True)  

        # Move embeddings to CPU and convert to numpy  
        embedding_store.append(embeddings.cpu().numpy()) 

        # Append the corresponding group index for each batch  
        group_indices_store.extend(batch['group_idx'].cpu().numpy().tolist())

        # and resource ids
        group_resourceid_store.extend(batch['ResourceId'])


  
# Concatenate all embeddings 
all_embeddings = np.concatenate(embedding_store, axis=0)  
print(f"All Embeddings Shape: {all_embeddings.shape}")

all_indices = np.concatenate(group_indices_store, axis=0)  
print(f"All Indices Shape: {all_indices.shape}")

group_resourceid_store = np.array(group_resourceid_store)
print(f"All ResourceIds Shape: {group_resourceid_store.shape}")


# In[31]:


# store
np.save('./embeddings/embeddings.npy', all_embeddings)
np.save('./embeddings/indices.npy', all_indices)
np.save('./embeddings/resource_ids.npy', group_resourceid_store)


# In[36]:


# load embeddings
import numpy as np
all_embeddings = np.load('./embeddings/embeddings.npy')
all_indices = np.load('./embeddings/indices.npy')
all_resource_ids = np.load('./embeddings/resource_ids.npy')


# In[8]:

from azure.core.credentials import AzureKeyCredential  
from azure.search.documents.aio import SearchClient as AsyncSearchClient  
import numpy as np  
import json  

# Define your Azure service details  
service_endpoint = "https://tmchatvdb.search.windows.net"  
api_key = "my_key"  
index_name = "fleetdebug-mtx"  

 
# Create a list of documents  
documents = []  
for j in range(len(all_resource_ids)):  
    documents.append({  
        'ResourceId': str(all_resource_ids[j]),  
        'Vector': all_embeddings[j].tolist(),  
        'GroupIndex': int(all_indices[j])  
    })  
  


# In[3]:


import asyncio  
import nest_asyncio
from azure.core.exceptions import HttpResponseError
nest_asyncio.apply()  


# Define a function to handle the upload in batches  
async def upload_documents_in_batches(documents, batch_size=1500):  
    credential = AzureKeyCredential(api_key)  
    async with AsyncSearchClient(service_endpoint, index_name, credential) as client:  
        try:  
            for i in range(0, len(documents), batch_size):  
                batch = documents[i:i+batch_size]  
                await client.upload_documents(documents=batch)  
                print(f"Uploaded batch {i // batch_size + 1}")  
        except HttpResponseError as e:  
            print(f"Upload of batch {i // batch_size + 1} failed: {e.message}")  
  
# batch size
batch_size = 1500  
  
# Run the upload in an asynchronous event loop  
loop = asyncio.get_event_loop()  
loop.run_until_complete(upload_documents_in_batches(documents, batch_size))  


# In[ ]:




