# main.py  
import os
from azureml.core import Run
import argparse
from dataset import GroupedInferenceDataset
import pandas as pd
from config import *
import pyarrow.dataset as ds  
from huggingface_hub import snapshot_download
import torch 
import torch.nn.functional as F  
from accelerate import Accelerator  
import tqdm  
import numpy as np  
from model import TabularTransformerEncoder  
from torch.nn.utils.rnn import pad_sequence  
from sentence_transformers import SentenceTransformer  
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader  






def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--batch_size_per_device", type=int, default=32, help="Batch size to use per device.")
    parser.add_argument("--num_devices", type=int, default=1, help="Number of devices to use.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use.")    
    parser.add_argument("--seed", type=int, default=1, help="Seed.") 
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train for.")
    parser.add_argument("--output_dir", type=str, default='/mnt/c/Users/afogarty/Desktop/InfoNCE/checkpoints', help="Output dir.")
    parser.add_argument("--lr", type=float, default=1e-05, help="Learning rate to use.")
    parser.add_argument("--dropout", type=float, default=0.134, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--noise_std", type=float, default=0.05, help="Embedding noise")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--d_model", type=int, default=512, help="Embedding dimension size.")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=768, help="FFNN Size")
    args = parser.parse_args()

    return args


  
def set_seed(seed):  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  

def collate_fn(batch):  
    def stack_and_pad(batch_data, key):  
        sequences = [item[key] for item in batch_data if item[key] is not None]  
        if not sequences:  
            return None, None  
  
        # Pad the sequences  
        padded_sequences = pad_sequence(sequences, batch_first=True)  
  
        # Create mask  
        mask = torch.zeros(padded_sequences.shape[:-1], dtype=torch.bool)  
        for i, seq in enumerate(sequences):  
            mask[i, :seq.size(0)] = True  
  
        return padded_sequences, mask  
  
    def process_data(data_list):  
        processed_data = {}  
        masks = {}  
        for key in ['x_cat', 'x_bin', 'x_cont', 'labels']:  
            stacked_data, mask = stack_and_pad(data_list, key)  
            if stacked_data is not None and mask is not None:  
                processed_data[key] = stacked_data  
                masks[key] = mask.unsqueeze(-1)  
        return processed_data, masks  
  
    def transform_x_cat(x_cat, embedding_table_shapes):  
        # Split x_cat into individual feature tensors  
        x_cat_split = torch.split(x_cat, 1, dim=-1)  
        x_cat_split = [tensor.squeeze(-1) for tensor in x_cat_split]  
  
        # Ensure the feature names match the embedding table shapes keys  
        if len(x_cat_split) != len(embedding_table_shapes):  
            raise ValueError("Mismatch between x_cat features and embedding table shapes")  
  
        # Create a dictionary to map feature names to tensors  
        x_cat_dict = {feature: tensor for feature, tensor in zip(embedding_table_shapes.keys(), x_cat_split)}  
        return x_cat_dict  
  
    def create_text_mask(text_data):  
        max_length = max(len(sequence) for sequence in text_data)  
        padded_sequences = []  
        for sequence in text_data:  
            padded_sequence = sequence + ['Sentinel Value'] * (max_length - len(sequence))  
            padded_sequences.append(padded_sequence)  
        return padded_sequences  
  
    # Extract query pairs  
    query_pairs = [item['query_data'] for item in batch]  
  
    # Process the query pairs data  
    query_data, query_masks = process_data(query_pairs)  
  
    # Include x_text as is (strings) and create masks  
    query_data['x_text'] = create_text_mask([item['x_text'] for item in query_pairs])  
    if 'x_cat' in query_data and query_data['x_cat'] is not None:  
        query_data['x_cat'] = transform_x_cat(query_data['x_cat'], embedding_table_shapes)  
  
    # Extract group info  
    query_group_info = [item['group_info'] for item in batch]  
  
    # Include sequence length information  
    seq_lengths = {  
        'query': query_data['x_cont'].size(1) if 'x_cont' in query_data else 0  
    }  
  
    return {  
        'query_data': query_data,  
        'query_masks': query_masks,  
        'seq_lengths': seq_lengths,  
        'query_group_info': query_group_info  
    }  




def encode_strings_as_tensors(strings):  
    """  
    Encodes a list of strings into a tensor of byte values.  
    """  
    max_length = max(len(s) for s in strings)  
    tensor = torch.zeros((len(strings), max_length), dtype=torch.uint8)  
    for i, s in enumerate(strings):  
        tensor[i, :len(s)] = torch.tensor(list(s.encode('utf-8')), dtype=torch.uint8)  
    return tensor, max_length  
  
def decode_tensors_to_strings(tensor, max_length):  
    """  
    Decodes a tensor of byte values back into a list of strings.  
    """  
    strings = []  
    for i in range(tensor.size(0)):  
        byte_array = tensor[i, :max_length].cpu().numpy().tobytes()  
        string = byte_array.split(b'\x00', 1)[0].decode('utf-8')  
        strings.append(string)  
    return strings  


def train_loop_per_worker(accelerator, config, inference_dataloader):  
    print("training_function called")  
    set_seed(config['seed'])  

    # embedding model
    st_model = SentenceTransformer(config['sentence_transformer_model_path']).to(accelerator.device).eval()

    # get sentinel value
    sentence = ['Sentinel Value']
    sentinel_embedding_value = torch.as_tensor(st_model.encode(sentence).squeeze(0)).to(accelerator.device)

    def encode_text_in_batches(batched_text_list, seq_len, model, sentinel_value="Sentinel Value", sentinel_embedding_value=sentinel_embedding_value):  
        flat_text = []  
        index_mapping = []  
    
        for i, sublist in enumerate(batched_text_list):  
            for text in sublist:  
                if text == sentinel_value and sentinel_embedding_value is not None:  
                    flat_text.append(None)  # Placeholder for sentinel value  
                else:  
                    flat_text.append(text)  
                index_mapping.append(i)  
    
        # Obtain embeddings  
        with torch.no_grad():  
            embeddings = []  
            for text in flat_text:  
                if text is None:  # Use precomputed sentinel embedding  
                    embeddings.append(sentinel_embedding_value)  
                else:  
                    embedding = accelerator.unwrap_model(model).encode([text], convert_to_tensor=True, show_progress_bar=False)  
                    embeddings.append(embedding.squeeze(0))  
    
        # Clone embeddings to avoid shared memory issues  
        embeddings = [emb.detach().clone() for emb in embeddings]  
    
        encoded_texts = [[] for _ in range(len(batched_text_list))]  
        for idx, emb in zip(index_mapping, embeddings):  
            encoded_texts[idx].append(emb)  
    
        # Pad sequences to the provided sequence length  
        padded_encoded_texts = []  
        for seq in encoded_texts:  
            if len(seq) < seq_len:  
                padding = [torch.zeros(embeddings[0].size(0), device=embeddings[0].device)] * (seq_len - len(seq))  
                seq.extend(padding)  
            else:  
                seq = seq[:seq_len]  # Truncate sequences longer than seq_len  
            
            # Clone each sequence tensor  
            padded_seq = torch.stack([s.detach().clone() for s in seq])  
            padded_encoded_texts.append(padded_seq)  
    
        # Stack the sequences to create a tensor of shape [batch_size, seq_len, n_features]  
        return torch.stack(padded_encoded_texts)  
      
    def encode_text_data_separately(batch, model):  
        encoded_texts = {}  
        encoded_texts['query'] = encode_text_in_batches(batch['query_data']['x_text'], batch['seq_lengths']['query'], model)  
        return encoded_texts  

    
    # train steps
    num_train_steps_per_epoch = config['inference_ds_len'] // ((accelerator.num_processes * config['batch_size_per_device']))
    print(f"Inference Len: {config['inference_ds_len']} | Num Workers; {accelerator.num_processes} | Batch Size: {config['batch_size_per_device']} ")

    # model
    model = TabularTransformerEncoder(  
        embedding_table_shapes=embedding_table_shapes,  
        num_cont_features=len(numerical_cols),   
        num_bin_features=len(binary_cols),   
        num_text_features=768,   
        d_model=config['d_model'],   
        nhead=config['nhead'],   
        num_layers=config['num_layers'],   
        dim_feedforward=config['dim_feedforward'],  
        noise_std=config['noise_std']
    ).to(accelerator.device)

    # Define the path to the fine-tuned model weights  
    fine_tuned_model_path = config['fine_tuned_model_path']
    
    # Ensure compatibility with distributed training  
    if accelerator.is_main_process:  
        # Load the state dict  
        state_dict = torch.load(fine_tuned_model_path, map_location=accelerator.device)  
        
        # Load the weights into the model  
        model.load_state_dict(state_dict)  
    
    # prepare
    model, inference_dataloader, st_model = accelerator.prepare(model, inference_dataloader, st_model)  
  
    if accelerator.is_main_process:  
        print("Starting training ...")  
        print("Number of batches on main process", num_train_steps_per_epoch)  

    # eval loop
    model.eval()  
    embeddings = []
    node_ids = []
    oos_span_ids = []
            
    for step, batch in tqdm.tqdm(enumerate(inference_dataloader), total=num_train_steps_per_epoch):

        with torch.no_grad():

            batch_node_ids = [x.get('node_id') for x in batch['query_group_info']]
            batch_oos_span_ids = [x.get('SpellNumber') for x in batch['query_group_info']]

            # Encode text data separately  
            encoded_texts = encode_text_data_separately(batch, st_model)  

            # Manually extract data and insert into the model for query data  
            query_pooled_output = model(  
                batch['query_data']['x_cat'],  
                batch['query_data']['x_bin'],  
                batch['query_data']['x_cont'],  
                encoded_texts['query'],  
                batch['query_masks']['x_cat'],  
                batch['query_masks']['x_bin'],  
                batch['query_masks']['x_cont']  
            )  
    
            # Move the output to the appropriate device before gathering  
            query_pooled_output = query_pooled_output.to(accelerator.device)  
            gathered_embeddings = accelerator.gather(query_pooled_output)  
            
            # Detach and move to CPU to avoid GPU memory overflow  
            gathered_embeddings = gathered_embeddings.detach().cpu()  
            embeddings.append(gathered_embeddings)  
    
            # Convert strings to tensors for gathering  
            batch_node_ids_tensor, max_node_id_length = encode_strings_as_tensors(batch_node_ids)  
            batch_oos_span_ids_tensor = torch.tensor(batch_oos_span_ids, dtype=torch.long, device=accelerator.device)  
    
            # Move the tensors to the appropriate device before gathering  
            batch_node_ids_tensor = batch_node_ids_tensor.to(accelerator.device)  
    
            # Gather tensors across processes  
            gathered_node_ids_tensor = accelerator.gather(batch_node_ids_tensor)  
            gathered_oos_span_ids_tensor = accelerator.gather(batch_oos_span_ids_tensor)  
    
            # Detach and move to CPU to avoid GPU memory overflow  
            gathered_node_ids_tensor = gathered_node_ids_tensor.detach().cpu()  
            gathered_oos_span_ids_tensor = gathered_oos_span_ids_tensor.detach().cpu()  
    
            # Convert tensors back to strings  
            gathered_node_ids = decode_tensors_to_strings(gathered_node_ids_tensor, max_node_id_length)  
            gathered_oos_span_ids = gathered_oos_span_ids_tensor.tolist()  
    
            # Flatten the list of lists  
            node_ids.extend(gathered_node_ids)  
            oos_span_ids.extend(gathered_oos_span_ids)  
    
    # Concatenate all gathered embeddings  
    all_embeddings = torch.cat(embeddings).numpy()  
    accelerator.wait_for_everyone()  
    
    if accelerator.is_main_process:  
        if accelerator.process_index == 0:  
            # Save embeddings to disk  
            df = pd.DataFrame({  
                'node_id': node_ids,  
                'oos_span_id': oos_span_ids,  
                'embedding': list(all_embeddings)  
            })  
            df.to_parquet(output_path + '/embeddings.parquet')  

            

if __name__ == '__main__':  

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    

    # init accelerator
    accelerator = Accelerator(mixed_precision="fp16")  

    aml_context = Run.get_context()
    data_path = './data' if aml_context._run_id.startswith('OfflineRun') else aml_context.input_datasets['train_files']
    output_path = '/mnt/c/Users/afogarty/Desktop/InfoNCE/checkpoints' if aml_context._run_id.startswith('OfflineRun') else aml_context.output_datasets['output_dir']  
    support_path = '/mnt/c/Users/afogarty/Desktop/InfoNCE/data' if aml_context._run_id.startswith('OfflineRun') else aml_context.input_datasets['support_files']  

    # get args
    args = parse_args()
    args.output_dir = output_path

    # update the config with args so that we have access to them.
    config = vars(args)

    # load data
    print("Loading train data")
    inference_df = ds.dataset(data_path, format="parquet")  
    inference_df = inference_df.to_table()  
    inference_df = inference_df.to_pandas()
    inference_df = inference_df.sort_values(by=['NodeId', 'LatestRecord'], ascending=False).reset_index(drop=True)

    # get sentence transformer
    shared_data_basepath = '/mnt/modelpath'
    sentence_transformer_model = 'avsolatorio/GIST-Embedding-v0'
    sentence_transformer_model_path_name = 'models--avsolatorio--GIST-Embedding-v0'
    sentence_transformer_model_snapshot = 'bf6b2e55e92f510a570ad4d7d2da2ec8cd22590c'
    sentence_transformer_model_path = shared_data_basepath + '/' + sentence_transformer_model_path_name + '/snapshots/' + sentence_transformer_model_snapshot

    # prepare the persistent shared directory to store artifacts needed for the ray workers
    os.makedirs(shared_data_basepath, exist_ok=True)

    # One time download of the sentence transformer model to a shared persistent storage available to the ray workers
    snapshot_download(repo_id=sentence_transformer_model, revision=sentence_transformer_model_snapshot, cache_dir=shared_data_basepath)

    # add to config
    config['sentence_transformer_model_path'] = sentence_transformer_model_path
    config['fine_tuned_model_path'] = support_path + '/epoch_4_state_dict.pt'

    # build train and test data
    inference_set = OptimizedInferenceDataset(
        data_df=inference_df,
        cat_cols=cat_cols,
        binary_cols=binary_cols,
        cont_cols=numerical_cols,
        label_cols=label_col,
        text_col=text_col,
    )
    print(f"Len Inference Set: {len(inference_set)} ")


    # dataloaders
    print("Generating data loaders")  
    inference_dataloader = DataLoader(  
        inference_set,  
        batch_size=config['batch_size_per_device'],  
        shuffle=False,  
        collate_fn=collate_fn,  
        pin_memory=True,  
        num_workers=4,  
        prefetch_factor=2,
        drop_last=False,
    )

    # add count
    config['inference_ds_len'] = len(inference_set)

    # loop
    train_loop_per_worker(accelerator, config, inference_dataloader)  

