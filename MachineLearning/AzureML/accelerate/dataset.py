import torch  
from torch.utils.data import Dataset  
import numpy as np  
import pandas as pd  
import random  
import json





class GroupedDataFrameDataset(Dataset):  
    def __init__(self, sequences_path, data_df, cat_cols, binary_cols, cont_cols, text_col, split='train'):  
        self.data_df = data_df.set_index('NodeId')  # Set NodeId as the index  
        self.sequences_df = pd.read_parquet(sequences_path, filters=[('test_split', '=', split)]).reset_index(drop=True)  
  
        self.cat_cols = cat_cols  
        self.binary_cols = binary_cols  
        self.cont_cols = cont_cols  
        self.text_col = text_col  
        self.split = split  
        self.group_indices = self.sequences_df.index.tolist()  
        self._shuffle_indices()  
        self.curriculum_indices = {  
            'low': self.sequences_df[self.sequences_df['Curriculum'] == 'low'].index.tolist(),  
            'medium': self.sequences_df[self.sequences_df['Curriculum'] == 'medium'].index.tolist(),  
            'high': self.sequences_df[self.sequences_df['Curriculum'] == 'high'].index.tolist(),  
        }  
  
    def _shuffle_indices(self):  
        np.random.shuffle(self.group_indices)  
  
    def __len__(self):  
        return len(self.group_indices)  
            
    def __getitem__(self, idx):  
        actual_idx = self.group_indices[idx]  
        row = self.sequences_df.iloc[actual_idx]  
        
        node_id = row['NodeId']  
        spell_number = row['SpellNumber']  
        sequence_length = row['SequenceLength']  
        curriculum_label = row['Curriculum']  
        
        # Collect all NodeIds to be loaded at once  
        node_ids = [node_id]  
        negative_examples = json.loads(row['NegativeExamplesJson'])  
        node_ids.extend([neg['NodeId'] for neg in negative_examples])  
        
        # Load all relevant data once  
        node_data = self._batch_load_data(node_ids)  
        
        # Filter data for the query and positive example  
        query_data, positive_data = self._filter_data(node_data.loc[node_id], spell_number, sequence_length)  
        
        # Handle negative examples  
        negative_data_list = []  
        for neg in negative_examples:  
            neg_node_id = neg['NodeId']  
            neg_data = self._filter_data(node_data.loc[neg_node_id], neg['SpellNumber'], sequence_length)[1]  
            negative_data_list.append(neg_data)  
        
        # Prepare data  
        query = self._prepare_data(self._get_batch_data(query_data))  
        positive_pair = self._prepare_data(self._get_batch_data(positive_data))  
        negative_pairs = [self._prepare_data(self._get_batch_data(neg_data)) for neg_data in negative_data_list]  
        
        # Prepare labels  
        query_label = int(row['QueryLabel'])  
        positive_label = int(row['PositiveLabel'])  
        
        negative_labels_str = row['NegativeLabels']
        negative_labels = negative_labels_str.tolist()        
        # Ensure all elements are integers  
        negative_labels = [int(label) for label in negative_labels]  
        
        return {  
            'query': query,  
            'positive_pair': positive_pair,  
            'negative_pairs': negative_pairs,  
            'labels': {  
                'query_label': torch.tensor(query_label, dtype=torch.long),  
                'positive_label': torch.tensor(positive_label, dtype=torch.long),  
                'negative_labels': torch.tensor(negative_labels, dtype=torch.long)  
            },  
            'group_info': {  
                'query': {  
                    'NodeId': node_id,  
                    'SpellNumber': spell_number,  
                    'SequenceLength': sequence_length,  
                    'Curriculum': curriculum_label  
                },  
                'positive_pair': {  
                    'NodeId': node_id,  
                    'SpellNumber': spell_number,  
                    'SequenceLength': sequence_length + 1,  
                    'Curriculum': curriculum_label  
                },  
                'negative_pairs': [  
                    {  
                        'NodeId': neg['NodeId'],  
                        'SpellNumber': neg['SpellNumber'],  
                        'SequenceLength': sequence_length + 1,  
                        'Curriculum': curriculum_label  
                    }  
                    for neg in negative_examples  
                ],  
            }  
        }  


  
    def _get_batch_data(self, data):  
        x_cat = data[self.cat_cols].values if self.cat_cols else None  
        x_bin = data[self.binary_cols].values if self.binary_cols else None  
        x_cont = data[self.cont_cols].values if self.cont_cols else None  
        x_text = data[self.text_col].tolist() if self.text_col else None  
  
        batch_data = {  
            'x_cat': x_cat,  
            'x_bin': x_bin,  
            'x_cont': x_cont,  
            'x_text': x_text,  
        }  
  
        return batch_data  
  
    def _prepare_data(self, data):  
        prepared_data = {}  
  
        if data:  
            if data['x_cat'] is not None:  
                prepared_data['x_cat'] = torch.tensor(data['x_cat'], dtype=torch.long)  
            if data['x_bin'] is not None:  
                prepared_data['x_bin'] = torch.tensor(data['x_bin'], dtype=torch.float)  
            if data['x_cont'] is not None:  
                prepared_data['x_cont'] = torch.tensor(data['x_cont'], dtype=torch.float)  
            if data['x_text'] is not None:  
                prepared_data['x_text'] = data['x_text']  
  
        return prepared_data  
  
    def _batch_load_data(self, node_ids):  
        # Load all rows for the given node_ids  
        node_data = self.data_df.loc[node_ids]  
        return node_data  
  
    def _filter_data(self, node_data, spell_number, sequence_length):  
        # Filter data within the node data for the given spell_number and sequence_length  
        filtered_data = node_data[node_data['SpellNumber'] == spell_number].head(sequence_length + 1)  
        columns = self.cat_cols + self.binary_cols + self.cont_cols + [self.text_col]  
        filtered_data = filtered_data[columns]  
  
        query_data = filtered_data.iloc[:-1]  # All but the last row for the query  
        positive_data = filtered_data  # All rows for the positive example  
  
        return query_data, positive_data  




class RepeatOffenderInference(Dataset):  
    def __init__(self, data_df, cat_cols, binary_cols, cont_cols, text_col):  
        self.data_df = data_df.set_index('NodeId')  # Set NodeId as the index  
        self.cat_cols = cat_cols  
        self.binary_cols = binary_cols  
        self.cont_cols = cont_cols  
        self.text_col = text_col  
  
        # Precompute all possible NodeId and SpellNumber combinations we need to generate data for  
        self.node_spell_pairs = (  
            self.data_df.reset_index()  
            .groupby(['NodeId', 'SpellNumber'])  
            .filter(lambda group: set(group['RepeatOffender']) >= {1, 2} and len(group) <= 23)  
            [['NodeId', 'SpellNumber', 'RepeatOffender']]  
            .query("RepeatOffender != 0")  
            .drop_duplicates()  
            .values.tolist()  
        )  

    def __len__(self):  
        return len(self.node_spell_pairs)  # Each pair generates a single sequence based on RepeatOffender value  
  
    def __getitem__(self, idx):  
        node_id, spell_number, repeat_offender_value = self.node_spell_pairs[idx]  
        group = self.data_df.loc[node_id]  
        group = group[group['SpellNumber'] == spell_number].reset_index()  
  
        # Step 1: Find the earliest row where RepeatOffender == 2  
        start_index = group[group['RepeatOffender'] == 2].index[0]  
  
        # Step 2: Get all data up to this point -- now we have group 2  
        group2 = group.loc[:start_index]  
  
        # Step 3: Group1 is just group2 minus its tail  
        group1 = group2.iloc[:-1]  
  
        # Determine which group to return based on RepeatOffender value  
        if repeat_offender_value == 1:  
            data = group1  
            return_repeat_offender_value = 1  
        else:  
            data = group2  
            return_repeat_offender_value = 2  
  
        # Ensure data is not empty  
        if data.empty:  
            raise IndexError("Generated empty sequence for given index.")  
  
        # extract last row of the group sequence  
        repeat_offender_label = data['RepeatOffender'].iloc[-1]  
        fault_code_label = data['FaultCode'].iloc[-1] if 'FaultCode' in data.columns else None  
  
        # Prepare data  
        prepared_data = self._prepare_data(self._get_batch_data(data))  
  
        return {  
            'query': prepared_data,  
            'group_info': {  
                'NodeId': node_id,  
                'SpellNumber': spell_number,  
                'SequenceLength': data.shape[0],  
                'RepeatOffender': return_repeat_offender_value,  
                'FaultCode': fault_code_label,  
            }  
        }  
  
    def _get_batch_data(self, data):  
        x_cat = data[self.cat_cols].values if self.cat_cols else None  
        x_bin = data[self.binary_cols].values if self.binary_cols else None  
        x_cont = data[self.cont_cols].values if self.cont_cols else None  
        x_text = data[self.text_col].tolist() if self.text_col else None  
  
        batch_data = {  
            'x_cat': x_cat,  
            'x_bin': x_bin,  
            'x_cont': x_cont,  
            'x_text': x_text,  
        }  
  
        return batch_data  
  
    def _prepare_data(self, data):  
        prepared_data = {}  
  
        if data:  
            if data['x_cat'] is not None:  
                prepared_data['x_cat'] = torch.tensor(data['x_cat'], dtype=torch.long)  
            if data['x_bin'] is not None:  
                prepared_data['x_bin'] = torch.tensor(data['x_bin'], dtype=torch.float)  
            if data['x_cont'] is not None:  
                prepared_data['x_cont'] = torch.tensor(data['x_cont'], dtype=torch.float)  
            if data['x_text'] is not None:  
                prepared_data['x_text'] = data['x_text']  
  
        return prepared_data  











class GroupedInferenceDataFrameDataset(Dataset):  
    def __init__(self, data_df, cat_cols, binary_cols, cont_cols, text_col):  
        self.data_df = data_df.set_index('NodeId')  # Set NodeId as the index  
        self.cat_cols = cat_cols  
        self.binary_cols = binary_cols  
        self.cont_cols = cont_cols  
        self.text_col = text_col  
  
        # Create a list of (NodeId, SpellNumber, RepeatOffender_SI) pairs using vectorized operations  
        self.node_spell_pairs = self.data_df.reset_index()[['NodeId', 'SpellNumber', 'RepeatOffender']].drop_duplicates().values.tolist()  
  
    def __len__(self):  
        return len(self.node_spell_pairs)  
  
    def __getitem__(self, idx):  
        node_id, spell_number, repeat_offender_si = self.node_spell_pairs[idx]  
        node_data = self.data_df.loc[node_id]  
  
        # Filter data for the given SpellNumber and RepeatOffender_SI  
        sequence_length = node_data[(node_data['SpellNumber'] == spell_number) & (node_data['RepeatOffender_SI'] == repeat_offender_si)].shape[0]  
        query_data = self._filter_data(node_data, spell_number, repeat_offender_si, sequence_length)  
  
        # Prepare data  
        prepared_data = self._prepare_data(self._get_batch_data(query_data))  
  
        return {  
            'query': prepared_data,  
            'group_info': {  
                'NodeId': node_id,  
                'SpellNumber': spell_number,  
                'RepeatOffender_SI': repeat_offender_si,  
                'SequenceLength': sequence_length  
            }  
        }  
  
    def _get_batch_data(self, data):  
        x_cat = data[self.cat_cols].values if self.cat_cols else None  
        x_bin = data[self.binary_cols].values if self.binary_cols else None  
        x_cont = data[self.cont_cols].values if self.cont_cols else None  
        x_text = data[self.text_col].tolist() if self.text_col else None  
  
        batch_data = {  
            'x_cat': x_cat,  
            'x_bin': x_bin,  
            'x_cont': x_cont,  
            'x_text': x_text,  
        }  
  
        return batch_data  
  
    def _prepare_data(self, data):  
        prepared_data = {}  
  
        if data:  
            if data['x_cat'] is not None:  
                prepared_data['x_cat'] = torch.tensor(data['x_cat'], dtype=torch.long)  
            if data['x_bin'] is not None:  
                prepared_data['x_bin'] = torch.tensor(data['x_bin'], dtype=torch.float)  
            if data['x_cont'] is not None:  
                prepared_data['x_cont'] = torch.tensor(data['x_cont'], dtype=torch.float)  
            if data['x_text'] is not None:  
                prepared_data['x_text'] = data['x_text']  
                  
        return prepared_data  
  
    def _filter_data(self, node_data, spell_number, repeat_offender_si, sequence_length):  
        # Filter data within the node data for the given spell_number, repeat_offender_si, and sequence_length  
        filtered_data = node_data[(node_data['SpellNumber'] == spell_number) & (node_data['RepeatOffender_SI'] == repeat_offender_si)].head(sequence_length)  
        columns = self.cat_cols + self.binary_cols + self.cont_cols + [self.text_col]  
        filtered_data = filtered_data[columns]  
  
        return filtered_data  


