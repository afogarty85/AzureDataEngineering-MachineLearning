import torch  
from torch.utils.data import Dataset  
import numpy as np  
import pandas as pd  
import random  



class GroupedDataFrameDataset(Dataset):  
    def __init__(self, pairs_path, data_df, cat_cols, binary_cols, cont_cols, label_cols, text_col, split='train'):  
        self.pairs_path = pairs_path  
        self.data_df = data_df.set_index('NodeId')  # Set NodeId as the index  
        self.pairs_df = pd.read_parquet(pairs_path, filters=[('test_split', '=', split)]).reset_index(drop=True)  
          
        self.cat_cols = cat_cols  
        self.binary_cols = binary_cols  
        self.cont_cols = cont_cols  
        self.label_cols = label_cols  
        self.text_col = text_col  
        self.split = split  
        self.group_indices = self.pairs_df.index.tolist()  
        self._shuffle_indices()  
  
    def _shuffle_indices(self):  
        random.shuffle(self.group_indices)  
  
    def __len__(self):  
        return len(self.group_indices)  
  
    def __getitem__(self, idx):  
        actual_idx = self.group_indices[idx]  
        row = self.pairs_df.iloc[actual_idx]  
        node_id = row['NodeId']  
        inter_node_id = row['InterNodeId']  
        query_oos_span = row['QueryOOSSpan']  
        positive_oos_span = row['PositiveOOSSpan']  
        negative_oos_spans = row['NegativeOOSSpan']  
        inter_positive_oos_span = row['InterPositiveOOSSpan']  
        inter_negative_oos_spans = row['InterNegativeOOSSpan']  
  
        # Ensure negative_oos_spans and inter_negative_oos_spans are lists  
        if isinstance(negative_oos_spans, str):  
            negative_oos_spans = eval(negative_oos_spans)  
        elif isinstance(negative_oos_spans, np.ndarray):  
            negative_oos_spans = negative_oos_spans.tolist()  
        elif not isinstance(negative_oos_spans, list):  
            negative_oos_spans = [negative_oos_spans]  
  
        if isinstance(inter_negative_oos_spans, str):  
            inter_negative_oos_spans = eval(inter_negative_oos_spans)  
        elif isinstance(inter_negative_oos_spans, np.ndarray):  
            inter_negative_oos_spans = inter_negative_oos_spans.tolist()  
        elif not isinstance(inter_negative_oos_spans, list):  
            inter_negative_oos_spans = [inter_negative_oos_spans]  
  
        # Load all data for the node_id and inter_node_id once  
        node_data = self.data_df.loc[[node_id]]  # Using a list to ensure it returns a DataFrame  
        inter_node_data = self.data_df.loc[[inter_node_id]]  
  
        query_data = self._load_data(node_data, query_oos_span)  
        positive_data = self._load_data(node_data, positive_oos_span)  
        negative_data_list = [self._load_data(node_data, neg_span) for neg_span in negative_oos_spans]  
          
        inter_positive_data = self._load_data(inter_node_data, inter_positive_oos_span)  
        inter_negative_data_list = [self._load_data(inter_node_data, neg_span) for neg_span in inter_negative_oos_spans]  
  
        query = self._prepare_data(self._get_batch_data(query_data))  
        positive_pair = self._prepare_data(self._get_batch_data(positive_data))  
        negative_pairs = [self._prepare_data(self._get_batch_data(neg_data)) for neg_data in negative_data_list]  
          
        inter_positive_pair = self._prepare_data(self._get_batch_data(inter_positive_data))  
        inter_negative_pairs = [self._prepare_data(self._get_batch_data(neg_data)) for neg_data in inter_negative_data_list]  
  
        # Return data without 'group_info' to reduce memory usage  
        return {  
            'query': query,  
            'positive_pair': positive_pair,  
            'negative_pairs': negative_pairs,  
            'inter_positive_pair': inter_positive_pair,  
            'inter_negative_pairs': inter_negative_pairs,  
            'group_info': {  
                'query': (node_id, query_oos_span),  
                'positive_pair': (node_id, positive_oos_span),  
                'negative_pairs': [(node_id, neg_span) for neg_span in negative_oos_spans],  
                'inter_positive_pair': (inter_node_id, inter_positive_oos_span),  
                'inter_negative_pairs': [(inter_node_id, neg_span) for neg_span in inter_negative_oos_spans]  
            }  
        }  
  
    def _get_batch_data(self, data):  
        x_cat = data[self.cat_cols].values if self.cat_cols else None  
        x_bin = data[self.binary_cols].values if self.binary_cols else None  
        x_cont = data[self.cont_cols].values if self.cont_cols else None  
        labels = data[self.label_cols].values if self.label_cols else None  
        x_text = data[self.text_col].tolist() if self.text_col else None  
  
        batch_data = {  
            'x_cat': x_cat,  
            'x_bin': x_bin,  
            'x_cont': x_cont,  
            'labels': labels,  
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
            if data['labels'] is not None:  
                prepared_data['labels'] = torch.tensor(data['labels'], dtype=torch.float)  
            if data['x_text'] is not None:  
                # Optionally tokenize/encode text data here  
                prepared_data['x_text'] = data['x_text']  
  
        return prepared_data  
  
    def _load_data(self, node_data, oos_span):  
        filters = node_data[node_data['OOS_Span'] == oos_span]  
        columns = self.cat_cols + self.binary_cols + self.cont_cols + self.label_cols + [self.text_col]  
        return filters[columns]  




  
class GroupedInferenceDataset(Dataset):  
    def __init__(self, data_df, cat_cols, binary_cols, cont_cols, label_cols, text_col):  
        self.data_df = data_df.set_index('NodeId')  # Set NodeId as the index  
        self.cat_cols = cat_cols  
        self.binary_cols = binary_cols  
        self.cont_cols = cont_cols  
        self.label_cols = label_cols  
        self.text_col = text_col  
  
        # Create a mapping of NodeId to set of OOS_Span values  
        self.mapping = self.data_df.groupby('NodeId')['OOS_Span'].agg(lambda x: set(x)).to_dict()  
  
        # Create a list of (NodeId, OOS_Span) pairs  
        self.group_indices = [(node_id, oos_span) for node_id, oos_spans in self.mapping.items() for oos_span in oos_spans]  
  
    def __len__(self):  
        return len(self.group_indices)  
  
    def __getitem__(self, idx):  
        node_id, oos_span = self.group_indices[idx]  
  
        # Filter data for the selected NodeId and OOS_Span  
        data = self._load_data(node_id, oos_span)  
        if data.empty:  
            print(f"No data found for NodeId: {node_id} and OOS_Span: {oos_span}")  
  
        prepared_data = self._prepare_data(self._get_batch_data(data))  
        return {  
            'query_data': prepared_data,  
            'group_info': {  
                'node_id': node_id,  
                'oos_span': oos_span  
            }  
        }  
  
    def _get_batch_data(self, data):  
        x_cat = data[self.cat_cols].values if self.cat_cols else None  
        x_bin = data[self.binary_cols].values if self.binary_cols else None  
        x_cont = data[self.cont_cols].values if self.cont_cols else None  
        labels = data[self.label_cols].values if self.label_cols else None  
        x_text = data[self.text_col].tolist() if self.text_col else None  
  
        batch_data = {  
            'x_cat': x_cat,  
            'x_bin': x_bin,  
            'x_cont': x_cont,  
            'labels': labels,  
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
            if data['labels'] is not None:  
                prepared_data['labels'] = torch.tensor(data['labels'], dtype=torch.float)  
            if data['x_text'] is not None:  
                # Optionally tokenize/encode text data here  
                prepared_data['x_text'] = data['x_text']  
  
        return prepared_data  
  
    def _load_data(self, node_id, oos_span):  
        try:  
            node_data = self.data_df.loc[[node_id]]  # Using a list to ensure it returns a DataFrame  
            filters = node_data[node_data['OOS_Span'] == oos_span]  
            columns = self.cat_cols + self.binary_cols + self.cont_cols + self.label_cols + [self.text_col]  
            return filters[columns]  
        except KeyError as e:  
            print(f"KeyError in _load_data: {e}")  
            return pd.DataFrame()  # Return an empty DataFrame to handle the error  

