# main.py  
import os
from azureml.core import Run
import argparse
from dataset import GroupedDataFrameDataset  
import pandas as pd
from config import *
import pyarrow.dataset as ds  
from huggingface_hub import snapshot_download
import torch 
import torch.nn.functional as F  
from accelerate import Accelerator  
from transformers import get_cosine_schedule_with_warmup  
import tqdm  
from tempfile import TemporaryDirectory  
import mlflow  
import numpy as np  
from model import TabularTransformerEncoder  
from loss import InfoNCELoss  
from torch.nn.utils.rnn import pad_sequence  
from sentence_transformers import SentenceTransformer  
from torch.utils.data import DataLoader
from multiprocessing import Manager  
from accelerate.utils import broadcast
import pickle
import time
from torch.utils.data import DataLoader, Dataset  
from accelerate import InitProcessGroupKwargs
from datetime import timedelta






def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--batch_size_per_device", type=int, default=32, help="Batch size to use per device.")
    parser.add_argument("--num_devices", type=int, default=1, help="Number of devices to use.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use.")    
    parser.add_argument("--seed", type=int, default=1, help="Seed.") 
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train for.")
    parser.add_argument("--output_dir", type=str, default='/mnt/c/Users/afogarty/Desktop/InfoNCE/checkpoints', help="Output dir.")
    parser.add_argument("--lr", type=float, default=2.7079965316792954e-05, help="Learning rate to use.")
    parser.add_argument("--dropout", type=float, default=0.134, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--noise_std", type=float, default=0.05, help="Embedding noise")
    parser.add_argument("--nhead", type=int, default=1, help="Number of attention heads.")
    parser.add_argument("--d_model", type=int, default=512, help="Embedding dimension size.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=768, help="FFNN Size")
    args = parser.parse_args()

    return args


def cosine_similarity(a, b):  
    return F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)  
  
def mean_reciprocal_rank(query_embeddings, positive_embeddings, negative_embeddings):  
    batch_size = query_embeddings.size(0)  
      
    # Calculate cosine similarity scores  
    pos_scores = cosine_similarity(query_embeddings, positive_embeddings)  # [batch_size, batch_size]  
    neg_scores = torch.matmul(query_embeddings.unsqueeze(1), negative_embeddings.permute(0, 2, 1))  # [batch_size, 1, num_negative]  
      
    # Only consider the diagonal elements for positive scores  
    pos_scores = pos_scores.diag().unsqueeze(1)  # [batch_size, 1]  
      
    # Combine positive and negative scores  
    all_scores = torch.cat([pos_scores, neg_scores.squeeze(1)], dim=1)  # [batch_size, 1 + num_negative]  
      
    # Rank the scores (descending order, so higher is better)  
    ranks = (all_scores >= pos_scores).sum(dim=1).float()  
      
    # Calculate MRR  
    mrr = (1.0 / ranks).mean().item()  
    return mrr  
  
def set_seed(seed):  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  


def collate_fn(batch):  
    def stack_and_pad(batch_data, key):  
        sequences = [item[key] for item in batch_data if item[key] is not None]  
        if not sequences:  
            return None, None  
        padded_sequences = pad_sequence(sequences, batch_first=True)  
        mask = torch.ones(padded_sequences.shape[:-1], dtype=torch.bool)  
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
        x_cat_split = torch.split(x_cat, 1, dim=-1)  
        x_cat_split = [tensor.squeeze(-1) for tensor in x_cat_split]  
        x_cat_dict = {feature: tensor for feature, tensor in zip(embedding_table_shapes.keys(), x_cat_split)}  
        return x_cat_dict  

    def create_text_mask(text_data):  
        # Create masks based on the Sentinel Value and pad sequences to the same length  
        max_length = max(len(sequence) for sequence in text_data)  
        padded_sequences = []  
        
        for sequence in text_data:  
            padded_sequence = sequence + ['Sentinel Value'] * (max_length - len(sequence))  
            padded_sequences.append(padded_sequence)  
        return padded_sequences

    positive_pairs = [item['positive_pair'] for item in batch]  
    query_pairs = [item['query'] for item in batch]  
    max_negative_pairs = 4  
    negative_pairs = []  
    for item in batch:  
        negs = item['negative_pairs']  
        if len(negs) < max_negative_pairs:  
            pad_value = negs[-1] if negs else {'x_cat': torch.zeros(1), 'x_bin': torch.zeros(1), 'x_cont': torch.zeros(1), 'labels': torch.zeros(1), 'x_text': "Sentinel Value"}  
            negs.extend([pad_value] * (max_negative_pairs - len(negs)))  
        negative_pairs.extend(negs[:max_negative_pairs])  

    positive_data, positive_masks = process_data(positive_pairs)  
    negative_data, negative_masks = process_data(negative_pairs)  
    query_data, query_masks = process_data(query_pairs)  

    # Include x_text as is (strings) and create masks  
    query_data['x_text'] = create_text_mask([item['x_text'] for item in query_pairs])  
    positive_data['x_text'] = create_text_mask([item['x_text'] for item in positive_pairs])  
    negative_data['x_text'] = create_text_mask([item['x_text'] for item in negative_pairs])  

    if 'x_cat' in query_data:  
        query_data['x_cat'] = transform_x_cat(query_data['x_cat'], embedding_table_shapes)  
    if 'x_cat' in positive_data:  
        positive_data['x_cat'] = transform_x_cat(positive_data['x_cat'], embedding_table_shapes)  
    if 'x_cat' in negative_data:  
        negative_data['x_cat'] = transform_x_cat(negative_data['x_cat'], embedding_table_shapes)  

    query_group_info = [item['group_info']['query'] for item in batch]  
    positive_group_info = [item['group_info']['positive_pair'] for item in batch]  
    negative_group_info = [item['group_info']['negative_pairs'] for item in batch]  

    # Include sequence length information  
    seq_lengths = {  
        'query': query_data['x_cont'].size(1) if 'x_cont' in query_data else 0,  
        'positive': positive_data['x_cont'].size(1) if 'x_cont' in positive_data else 0,  
        'negative': negative_data['x_cont'].size(1) if 'x_cont' in negative_data else 0  
    }  

    return {  
        'query_data': query_data,  
        'query_masks': query_masks,  
        'positive_data': positive_data,  
        'positive_masks': positive_masks,  
        'negative_data': negative_data,  
        'negative_masks': negative_masks,  
        'query_group_info': query_group_info,  
        'positive_group_info': positive_group_info,  
        'negative_group_info': negative_group_info,  
        'seq_lengths': seq_lengths  
    }  




def train_loop_per_worker(accelerator, config, train_dataloader, eval_dataloader):  
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
        encoded_texts['positive'] = encode_text_in_batches(batch['positive_data']['x_text'], batch['seq_lengths']['positive'], model)  
        encoded_texts['negative'] = encode_text_in_batches(batch['negative_data']['x_text'], batch['seq_lengths']['negative'], model)  
        return encoded_texts  
    
    # train steps
    num_train_steps_per_epoch = config['train_ds_len'] // ((accelerator.num_processes * config['batch_size_per_device']))
    print(f"Train Len: {config['train_ds_len']} | Num Workers; {accelerator.num_processes} | Batch Size: {config['batch_size_per_device']} ")
    num_eval_steps_per_epoch = config['eval_ds_len'] // ((accelerator.num_processes * config['batch_size_per_device']))

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

    def evaluate(model, eval_loader, eval_steps, loss_fn, accelerator, ds_kwargs):  
        model.eval()  
        losses = []  
        mrrs = []  
        print(f"Found this config: {config}")  
  
        for step, batch in tqdm.tqdm(enumerate(eval_loader), total=eval_steps):
            with torch.no_grad():  

                # get batch size
                batch_size = batch['query_data']['x_bin'].size(0)

                # Encode text data separately  
                encoded_texts = encode_text_data_separately(batch, st_model)  
                
                # Manually extract data and insert into the model  
                query_pooled_output = model(  
                    batch['query_data']['x_cat'],  
                    batch['query_data']['x_bin'],  
                    batch['query_data']['x_cont'],  
                    encoded_texts['query'],  
                    batch['query_masks']['x_cat'],  
                    batch['query_masks']['x_bin'],  
                    batch['query_masks']['x_cont']  
                )  
                
                positive_pooled_output = model(  
                    batch['positive_data']['x_cat'],  
                    batch['positive_data']['x_bin'],  
                    batch['positive_data']['x_cont'],  
                    encoded_texts['positive'],  
                    batch['positive_masks']['x_cat'],  
                    batch['positive_masks']['x_bin'],  
                    batch['positive_masks']['x_cont']  
                )  
                
                negative_pooled_output = model(  
                    batch['negative_data']['x_cat'],  
                    batch['negative_data']['x_bin'],  
                    batch['negative_data']['x_cont'],  
                    encoded_texts['negative'],  
                    batch['negative_masks']['x_cat'],  
                    batch['negative_masks']['x_bin'],  
                    batch['negative_masks']['x_cont']  
                ).view(batch_size, 4, 512)  
                
                loss = loss_fn(query_pooled_output, positive_pooled_output, negative_pooled_output)  
                losses.append(accelerator.gather(loss[None]))
  
                mrr = mean_reciprocal_rank(query_pooled_output, positive_pooled_output, negative_pooled_output)  
                mrrs.append(accelerator.gather(torch.tensor([mrr])))  
  
        all_losses = torch.cat(losses)  
        all_mrrs = torch.cat(mrrs)  
        avg_eval_loss = all_losses.mean().item()  
        avg_eval_mrr = all_mrrs.mean().item()  
  
        accelerator.print(f"Average Loss: {avg_eval_loss:.4f}")  
        accelerator.print(f"Average MRR: {avg_eval_mrr:.4f}")  
  
        return avg_eval_loss, avg_eval_mrr  
  
    # optimizer / scheduler
    warmup_ratio = 0.1  
    num_training_steps = num_train_steps_per_epoch * config['num_epochs']  
    num_warmup_steps = int(num_training_steps * warmup_ratio)  
  
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])  
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  

    # loss
    loss_fn = InfoNCELoss(temperature=0.07)  
  
    # prepare
    model, optimizer, train_dataloader, eval_dataloader, loss_fn, scheduler, st_model = accelerator.prepare(  
        model, optimizer, train_dataloader, eval_dataloader, loss_fn, scheduler, st_model)  
  
    if accelerator.is_main_process:  
        print("Starting training ...")  
        print("Number of batches on main process", num_train_steps_per_epoch)  

    # train loop
    for epoch in range(1, config['num_epochs'] + 1):  
        model.train()  
        losses = []  
        interval_loss = 0.0  
        interval_mrr = 0.0  
        
        for step, batch in tqdm.tqdm(enumerate(train_dataloader), total=num_train_steps_per_epoch):

            optimizer.zero_grad()

            # get batch size
            batch_size = batch['query_data']['x_bin'].size(0)         
            
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

            # Manually extract data and insert into the model for positive data  
            positive_pooled_output = model(  
                batch['positive_data']['x_cat'],  
                batch['positive_data']['x_bin'],  
                batch['positive_data']['x_cont'],  
                encoded_texts['positive'],  
                batch['positive_masks']['x_cat'],  
                batch['positive_masks']['x_bin'],  
                batch['positive_masks']['x_cont']  
            )  
            
            # Manually extract data and insert into the model for negative data  
            negative_pooled_output = model(  
                batch['negative_data']['x_cat'],  
                batch['negative_data']['x_bin'],  
                batch['negative_data']['x_cont'],  
                encoded_texts['negative'],  
                batch['negative_masks']['x_cat'],  
                batch['negative_masks']['x_bin'],  
                batch['negative_masks']['x_cont']  
            ).view(batch_size, 4, 512)  

            # loss fn
            loss = loss_fn(query_pooled_output, positive_pooled_output, negative_pooled_output)  
            losses.append(accelerator.gather(loss[None]))
            
            # metric
            mrr = mean_reciprocal_rank(query_pooled_output, positive_pooled_output, negative_pooled_output)

            # backward
            accelerator.backward(loss)  
            accelerator.clip_grad_norm_(model.parameters(), 1.0)  
            optimizer.step()  
            scheduler.step()  
  
            interval_loss += loss.item()  
            interval_mrr += mrr  
  
            if (step + 1) % 20 == 0:  
                if accelerator.is_main_process:  
                    mlflow.log_metric('interval_loss', interval_loss / 20)  
                    mlflow.log_metric('interval_contrastive_acc', interval_mrr / 20)  
                    accelerator.print(  
                        f"Epoch {epoch}/{config['num_epochs']}, Batch {step+1}, Interval Loss: {interval_loss / 20:.4f}, Interval MRR: {interval_mrr / 20:.4f}")  
                    interval_loss = 0.0  
                    interval_mrr = 0.0  

        all_losses = torch.cat(losses)  
        train_epoch_loss = all_losses.mean().item()  
        print(f"Epoch {epoch} / {config['num_epochs']}, Total Train Loss: {train_epoch_loss}")  
        print("Running evaluation ...")  
        accelerator.wait_for_everyone()  
  
        avg_eval_loss, avg_eval_mrr = evaluate(  
            model=model,  
            eval_loader=eval_dataloader,  
            eval_steps=num_eval_steps_per_epoch,
            loss_fn=loss_fn,  
            accelerator=accelerator,
            ds_kwargs={"collate_fn": collate_fn},
        )  
  
        if accelerator.is_main_process:  
            mlflow.log_metric('eval_loss', avg_eval_loss)  
            mlflow.log_metric('eval_mrr', avg_eval_mrr)  

        metrics = {  
            "epoch": epoch,  
            "iteration": step,  
            "train_loss": train_epoch_loss,  
            "eval_mrr": avg_eval_mrr,  
            "eval_loss": avg_eval_loss,  
            "num_iterations": step + 1,  
        }  

        accelerator.print(f"Saving the model locally at {config['output_dir']}")  
        with TemporaryDirectory() as tmpdir:  
            accelerator.wait_for_everyone()  
            if accelerator.is_main_process:  
                if accelerator.process_index == 0:  
                    state = accelerator.get_state_dict(model)  
                    accelerator.save(state, f"{tmpdir}/epoch_{epoch}_state_dict.pt")  

            print(metrics)

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
    train_df = ds.dataset(data_path, format="parquet")  
    train_df = train_df.to_table(filter=(ds.field('test_split') == 'train'))  
    train_df = train_df.to_pandas()
    train_df['NegativeOOSSpanTup'] = train_df['NegativeOOSSpan'].apply(tuple)
    train_df = train_df.drop_duplicates(subset=['NodeId', 'QueryOOSSpan', 'NegativeOOSSpanTup'])    
    train_df = train_df.sort_values(by=['NodeId', 'LatestRecord'], ascending=False).reset_index(drop=True)

    test_df = ds.dataset(data_path, format="parquet")  
    test_df = test_df.to_table(filter=(ds.field('test_split') == 'test'))  
    test_df = test_df.to_pandas()
    test_df['NegativeOOSSpanTup'] = test_df['NegativeOOSSpan'].apply(tuple)
    test_df = test_df.drop_duplicates(subset=['NodeId', 'QueryOOSSpan', 'NegativeOOSSpanTup'])    
    test_df = test_df.sort_values(by=['NodeId', 'LatestRecord'], ascending=False).reset_index(drop=True)

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

    # build train and test data  
    train_set = GroupedDataFrameDataset(  
        pairs_path=support_path + '/pairs.parquet',  
        data_df=train_df,  
        cat_cols=cat_cols,  
        binary_cols=binary_cols,  
        cont_cols=numerical_cols,  
        label_cols=label_col,  
        text_col=text_col,  
        split='train',  
    )  
    print(f"Len Train Set: {len(train_set)} ")

    test_set = GroupedDataFrameDataset(  
        pairs_path=support_path + '/pairs.parquet',  
        data_df=test_df,  
        cat_cols=cat_cols,  
        binary_cols=binary_cols,  
        cont_cols=numerical_cols,  
        label_cols=label_col,  
        text_col=text_col,  
        split='test',  
    )

    # dataloaders
    print("Generating data loaders")  
    train_dataloader = DataLoader(  
        train_set,  
        batch_size=config['batch_size_per_device'],  
        shuffle=True,  
        collate_fn=collate_fn,  
        pin_memory=True,  
        num_workers=5,  
        prefetch_factor=2,
        drop_last=True,
    )

    eval_dataloader = DataLoader(  
        test_set,  
        batch_size=config['batch_size_per_device'],  
        shuffle=False,  
        collate_fn=collate_fn,  
        pin_memory=True,  
        num_workers=5,  
        prefetch_factor=2  
    )  


    # add count
    config['train_ds_len'] = len(train_set)
    config['eval_ds_len'] = len(test_set)

    # loop
    train_loop_per_worker(accelerator, config, train_dataloader, eval_dataloader)  

