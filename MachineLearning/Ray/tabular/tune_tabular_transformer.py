from tempfile import TemporaryDirectory

import evaluate
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR  
from azureml.core import Run

from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import os

import ray.train
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, CheckpointConfig, ScalingConfig, Checkpoint
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Tuner
from ray import tune
from ray.tune.search import ConcurrencyLimiter

import torch.nn as nn  
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset  
from torch.nn.functional import sigmoid  




class FocalLoss(nn.Module):  
    def __init__(self, alpha=None, gamma=2):  
        super().__init__()  
        self.gamma = gamma  
        self.alpha = alpha  # The alpha parameter should be either a float or a tensor.  
  
    def forward(self, logits, labels, mask):  
        # logits is shape: [batch_size]  
        # labels is shape: [batch_size, seq_len, 1]  
        # mask is shape: [batch_size, seq_len, 1]  
          
        # Broadcast logits to match the shape of labels and mask  
        logits = logits.unsqueeze(-1).unsqueeze(-1).expand_as(labels)  
          
        # Use .reshape() to handle non-contiguous tensors  
        logits = logits.reshape(-1)  
        labels = labels.reshape(-1).float()  
        mask = mask.reshape(-1).float()  
  
        # Calculate BCE loss with logits without reduction  
        BCE_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')  
  
        # Calculate the probability term (pt) directly from the BCE loss  
        pt = torch.exp(-BCE_loss)  
  
        # Apply alpha balancing factor  
        if self.alpha is not None:  
            alpha_t = torch.where(labels == 1, self.alpha[1], self.alpha[0])  
            F_loss = alpha_t * ((1 - pt) ** self.gamma) * BCE_loss  
        else:  
            F_loss = ((1 - pt) ** self.gamma) * BCE_loss  
  
        # Apply the mask to the Focal loss  
        masked_F_loss = F_loss * mask  

       # Reduce the loss: sum and then divide by the number of unmasked elements  
        # Add a small epsilon to avoid division by zero  
        loss = masked_F_loss.sum() / (mask.sum() + 1e-8)  
  
        return loss  



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


class DynamicPositionalEncoding(nn.Module):  
    def __init__(self, dim_model, max_len=5000):  
        super(DynamicPositionalEncoding, self).__init__()  
        self.dim_model = dim_model  
        self.encoding = nn.Parameter(torch.randn(max_len, dim_model), requires_grad=False)  # Encoding for max_len positions  
  
    def forward(self, token_embeddings):  
        batch_size, seq_length, _ = token_embeddings.size()  
        encoding = self.encoding[:seq_length, :].unsqueeze(0).expand(batch_size, -1, -1)  # Match the input shape  
        return token_embeddings + encoding  

  
class SimplifiedAttentionPooling(nn.Module):    
    def __init__(self, dim_model):    
        super(SimplifiedAttentionPooling, self).__init__()    
        self.attention_weights = nn.Linear(dim_model, 1)    
    
    def forward(self, x, mask):    
        # Generate attention scores using a linear layer  
        attention_scores = self.attention_weights(x).squeeze(-1)  # [batch_size, seq_len]  
  
        # Squeeze the mask to match the dimensions of attention_scores  
        mask_squeezed = mask.squeeze(-1)  # [batch_size, seq_len]  
  
        # Apply the mask to set attention scores to '-inf' for padding tokens  
        attention_scores = attention_scores.masked_fill(~mask_squeezed, float('-inf'))  
  
        # Apply softmax to get the attention weights  
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]  
  
        # Weighted sum of the inputs using attention weights  
        attended = torch.sum(attention_weights * x, dim=1)  # [batch_size, dim_model]  
  
        return attended  



def masked_preds_labels(labels, logits, mask):  
    # logits should be broadcasted to match the shape of labels and mask  
    logits = logits.unsqueeze(-1).unsqueeze(-1).expand_as(labels)  
    mask_bool = mask.bool()  
      
    # Apply sigmoid to logits to get probabilities  
    probs = sigmoid(logits)  
      
    # Threshold probabilities to get predictions  
    predictions = (probs > 0.5).float()  
      
    # Use the mask to select the non-masked (True) elements from tensors  
    masked_predictions = predictions[mask_bool]  
    masked_labels = labels[mask_bool]  
      
    return masked_predictions, masked_labels



def evaluate(model, eval_loader, accelerator, loss_fn):
    model.eval()
    running_loss_sum = 0.0  # Variable to track the sum of losses over all steps  
    masked_pred_store = []
    masked_label_store = []
  
    with torch.no_grad():  # Move the no_grad context to cover the entire loop  
        for step, batch in tqdm(enumerate(eval_loader), total=len(eval_loader)):  
            logits = model(x_cat=batch["x_cat"],  
                           x_bin=batch["x_bin"],  
                           x_cont=batch["x_cont"],  
                           mask=batch['mask'],  
                           )
  
            labels = batch['labels']    
            mask = batch['mask']

            # loss
            loss = loss_fn(logits, labels, mask)
            loss_item = loss.item()  
            running_loss_sum += loss_item

            # get masked predictions
            masked_predictions, masked_labels = masked_preds_labels(labels, logits, mask)

            # store
            masked_pred_store.append(masked_predictions)
            masked_label_store.append(masked_labels)


    # average across batches
    average_loss = running_loss_sum / len(eval_loader)  

    # agg
    aggregated_loss = torch.mean(accelerator.gather(torch.tensor(average_loss).to(accelerator.device)[None])).item()    

    masked_label_cat = torch.cat(masked_label_store).cpu().numpy()
    masked_pred_cat = torch.cat(masked_pred_store).cpu().numpy()

    eval_precision_score = precision_score(masked_label_cat, masked_pred_cat, average='binary')
    eval_recall_score = recall_score(masked_label_cat, masked_pred_cat, average='binary')

    # Calculate mean loss and accuracy  
    return aggregated_loss, eval_precision_score, eval_recall_score  


class TabularTransformer(nn.Module):  
    def __init__(self, config, embedding_table_shapes, num_binary=0, num_continuous=0):  
        super(TabularTransformer, self).__init__()  
          
        # Configuration parameters  
        max_sequence_length = config.get("max_sequence_length", 512)  
        dim_model = config.get("dim_model", 1024)  
        nhead = config.get("num_heads", 8)  
        dim_feedforward = config.get("dim_feedforward", 2048)  
        dropout = config.get("dropout", 0.1)  
        n_layers = config.get("n_layers", 3)  
        activation_function = config.get("activation_function", "relu")  

        # Regularization: Dropout  
        self.dropout = nn.Dropout(dropout)  
          
        # Activation function (can be swapped out based on the 'activation_function' argument)  
        self.activation = getattr(F, activation_function)  

        # Entity Embedding for categorical features  
        self.embedding_layer = EntityEmbeddingLayer(embedding_table_shapes)  
        self.embedding_size = sum(emb_size for _, emb_size in embedding_table_shapes.values())  
          
        # Include binary and continuous data sizes  
        self.num_binary = num_binary  
        self.num_continuous = num_continuous  
          
        # Normalization for continuous features  
        if num_continuous > 0:  
            self.continuous_ln = nn.LayerNorm(num_continuous)  
          
        # Total input size calculation  
        total_input_size = self.embedding_size + num_binary + num_continuous  
          
        # Feature projection layer  
        self.feature_projector = nn.Linear(total_input_size, dim_model)  
          
        # Dynamic Positional Encoding  
        self.pos_encoder = DynamicPositionalEncoding(dim_model=config['dim_model'], max_len=config['max_sequence_length'])  
                
        # Custom Transformer Encoder Layer  
        encoder_layer = nn.TransformerEncoderLayer(  
            d_model=dim_model,  
            nhead=nhead,  
            dim_feedforward=dim_feedforward,  
            dropout=dropout,  
            activation=activation_function,  
            batch_first=True  # Set batch_first to True  
        )  
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)  


        # Simplified Attention Pooling  
        self.attention_pooling = SimplifiedAttentionPooling(dim_model)  

        # Output  
        self.regressor = nn.Linear(dim_model, 1)  

  
    def forward(self, x_cat, x_bin=None, x_cont=None, mask=None):  
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
          
        # Feature projection  
        x = self.feature_projector(x)  
        x = self.activation(x)  
          
        # Apply Dynamic Positional Encoding  
        x = self.pos_encoder(x)  
          
        # Simplified Masking for Transformer Encoder  
        if mask is not None:  
            encoder_mask = ~mask.squeeze(-1)
        else:  
            encoder_mask = None  
          
        # Apply Transformer Encoder  
        x = self.transformer_encoder(x, src_key_padding_mask=encoder_mask)  
          
        # Apply Simplified Attention Pooling  
        x = self.attention_pooling(x, mask.squeeze(-1))  
          
        # Output layer  
        output = self.regressor(x)  
        return output.squeeze(-1)  
  


embedding_size_ada = 0
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
        # Store the indices for each group  
        self.group_indices = list(dataframe.groupby(group_key).indices.values())  
  
    def __len__(self):  
        # The length of the dataset is the number of unique groups  
        return len(self.group_indices)  
    
    def __getitem__(self, idx):  
        # Fetch the indices for the group  
        indices = self.group_indices[idx]  
        # Select data for the group  
        group_data = self.dataframe.iloc[indices]  
    
        # Process categorical and binary columns  
        x_cat = torch.tensor(group_data[self.cat_cols].values, dtype=torch.long) if self.cat_cols else None  
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
            embeddings = torch.zeros((len(indices), embedding_size_ada), dtype=torch.float)  
            # Iterate over the group data and replace zeros with actual embeddings where they exist  
            for i, emb in enumerate(group_data[self.embedding_col]):  
                if isinstance(emb, np.ndarray):  
                    embeddings[i] = torch.tensor(emb, dtype=torch.float)  
            continuous_features.append(embeddings)  
    
        # Concatenate all continuous features along the last dimension  
        x_cont = torch.cat(continuous_features, dim=-1) if continuous_features else None  
    
        # Process mask and labels  
        mask = torch.tensor(group_data[self.mask_col].values, dtype=torch.bool) if self.mask_col else None  
        labels = torch.tensor(group_data[self.label_col].values, dtype=torch.float) if self.label_col else None  
    
        return {  
            'x_cat': x_cat,  
            'x_bin': x_bin,  
            'x_cont': x_cont,  
            'mask': mask,  
            'labels': labels  
        }  


def train_func(config):
    """Your training function that will be launched on each worker."""
    print(f"Found this config: {config}")


    aml_context = Run.get_context()
    train_files = aml_context.input_datasets['train_files']

    # embed table       
    EMBEDDING_TABLE_SHAPES = {'FaultCode': (294, 100),
    'RobertsRules': (254, 100),
    'rackType': (5, 100),
    'Role': (44, 100),
    'serverCount': (38, 100),
    'targetWorkload': (10, 100)}

    # update
    EMBEDDING_TABLE_SHAPES = {k: (v[0], config['emb_size']) for k, v in EMBEDDING_TABLE_SHAPES.items()}  

    # feature set
    cat_cols = list(EMBEDDING_TABLE_SHAPES.keys())

    binary_cols = [
                    'SP6RootCaused',
                    'wasCatastrophic',
                    'KickedFlag', 
                    'ReturnedToProduction',
                    ]  
    
    cont_cols = ['HoursLastRepair', 'DeviceAge',
                'RepairsPerDevice', 'SuccessRate', 'MTTR', 'SpecializationScore',
                "AvgRepairTimeByErrorCode", "EfficiencyScore", "ExperienceScore",
                'CumulativeErrors', 'CumulativeFCFreq',
                ]

    label_col = [
        'Survived90Days',
    ]

    mask_col = ['Mask']

    set_seed(config['seed'])

    # Initialize accelerator
    accelerator = Accelerator(
        deepspeed_plugin=None,
        gradient_accumulation_steps=1,
        mixed_precision=config['mx'],
    )

    # Load datasets
    # train_path = Path('/mnt/c/Users/afogarty/Desktop/AZRepairRL/data/seq_azrepair_train.parquet')  
    # test_path = Path('/mnt/c/Users/afogarty/Desktop/AZRepairRL/data/seq_azrepair_test.parquet')  
    train_set_df = pd.read_parquet(train_files + '/seq_azrepair_train.parquet')
    test_set_df = pd.read_parquet(train_files + '/seq_azrepair_test.parquet')

    # Prepare PyTorch Datasets
    # ====================================================
    train_set = GroupedDataFrameDataset(  
        dataframe=train_set_df,
        group_key='ResourceId',
        cat_cols=cat_cols,  
        binary_cols=binary_cols,  
        cont_cols=cont_cols,  
        mask_col=mask_col,  
        label_col=label_col,
        embedding_col=None,
        embedding_dim_size=embedding_size_ada
    )  

    test_set = GroupedDataFrameDataset(  
        dataframe=test_set_df,
        group_key='ResourceId',
        cat_cols=cat_cols,  
        binary_cols=binary_cols,  
        cont_cols=cont_cols,  
        mask_col=mask_col,  
        label_col=label_col,
        embedding_col=None,
        embedding_dim_size=embedding_size_ada
    )  

    # Prepare PyTorch DataLoaders
    # ====================================================
    train_dataloader = DataLoader(
        train_set,
        batch_size=config['batch_size_per_device'],
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        drop_last=True,
    )

    eval_dataloader = DataLoader(
        test_set,
        batch_size=config['batch_size_per_device'],
        shuffle=False,
        num_workers=8,
        prefetch_factor=2,
        drop_last=False,
    )

    # ====================================================

    # Instantiate the model, optimizer, lr_scheduler
    transformer_config = config['transformer_config']

    model = TabularTransformer(config=transformer_config,
                                embedding_table_shapes=EMBEDDING_TABLE_SHAPES,
                                num_binary=len(binary_cols),
                                num_continuous=len(cont_cols),
                                )



    # Assuming you have these values defined:  
    num_epochs = config['num_epochs']
    
    # optim  
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])  
    
    # Total number of training steps  
    total_steps = len(train_dataloader) * num_epochs  
    
    # CosineAnnealingLR scheduler  
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)
    
    # Prepare everything with accelerator
    (
        model,
        optimizer,
        scheduler,
        train_dataloader,
        eval_dataloader,
    ) = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, eval_dataloader
    )


    alpha_tensor = torch.tensor([1 - config['alpha'], config['alpha']]).float().to(accelerator.device)
    loss_fn = FocalLoss(alpha=alpha_tensor, gamma=config['gamma'])  

    for epoch in range(1, num_epochs + 1):
        model.train()

        running_loss_sum = 0.0  # Variable to track the sum of losses over all steps  
        last_hundred_loss_sum = 0.0  # Variable to track the sum of losses for the last 100 steps  

        masked_pred_store = []
        masked_label_store = []

        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            logits = model(x_cat=batch["x_cat"],
                        x_bin=batch["x_bin"],
                        x_cont=batch["x_cont"],
                        mask=batch['mask'],
                            )
            
            assert not torch.isnan(logits).any(), "NaNs in outputs"  

            # loss
            labels = batch['labels']    
            assert not torch.isnan(labels).any(), "NaNs detected in labels"  

            mask = batch['mask']
            loss = loss_fn(logits, labels, mask)
            loss_item = loss.item()  
            running_loss_sum += loss_item  
            last_hundred_loss_sum += loss_item  # Accumulate loss for the last 100 steps  
            assert not torch.isnan(loss).any() and not torch.isinf(loss).any(), "Invalid loss detected"  

            # get masked predictions
            masked_predictions, masked_labels = masked_preds_labels(labels, logits, mask)

            # store
            masked_pred_store.append(masked_predictions)
            masked_label_store.append(masked_labels)

            accelerator.backward(loss)
            clip_grad_norm_(accelerator.unwrap_model(model).parameters(), 1.0)  

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Report progress  
            if (step + 1) % 100 == 0:
                average_loss_last_hundred = last_hundred_loss_sum / 100
        
                if accelerator.is_main_process:  
                    accelerator.print(  
                        f"[epoch {epoch} step {step} "  
                        f"average loss: {average_loss_last_hundred:.4f} | "  
                    )
                last_hundred_loss_sum = 0.0  # Reset the sum for the next 100 steps  


        # wait
        accelerator.wait_for_everyone()
        
        # average across batches
        average_loss = running_loss_sum / len(train_dataloader)  

        # agg
        aggregated_loss = torch.mean(accelerator.gather(torch.tensor(average_loss).to(accelerator.device)[None])).item()    

        # reset
        running_loss_sum = 0.0  

        # eval
        eval_loss, eval_precision, eval_recall = evaluate(model, eval_dataloader, accelerator, loss_fn)
        if accelerator.is_main_process:  
            accelerator.print(  
            f"[ epoch {epoch}"  
            f" train loss: {aggregated_loss:.4f} |"

            f" eval loss: {eval_loss:.4f} |"
            f" eval prec: {eval_precision :.4f} |"
            f" eval recall: {eval_recall :.4f} ]"

            )

        eval_metrics = {"loss": eval_loss,
                        "precision": eval_precision,
                        "recall": eval_recall,
                        }

        ray.train.report(metrics=eval_metrics)





def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mx",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--batch_size_per_device",
        type=int,
        default=1024,
        help="Batch size to use per device.",
    )
    parser.add_argument(
        "--eval_batch_size_per_device",
        type=int,
        default=4096,
        help="Batch size to use per device (For evaluation).",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Focal loss alpha."
    )
    parser.add_argument(
        "--emb_dropout", type=float, default=0.5, help="Embedding dropout."
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=1, help="Number of concurrent trials."
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of trials."
    )             
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="Focal loss gamma."
    )    
    parser.add_argument(
        "--warmup", type=int, default=100, help="Warmup steps."
    )    
    parser.add_argument(
        "--emb_size", type=int, default=20, help="Number of embedding features."
    )    
    parser.add_argument(
        "--num_devices", type=int, default=1, help="Number of devices to use."
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers to use."
    )    
    parser.add_argument(
        "--seed", type=int, default=1, help="Seed."
    ) 
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--output_dir", type=str, default='/mnt/c/Users/afogarty/Desktop/RandomizedTest/checkpoints', help="Output dir."
    )
    parser.add_argument("--lr", type=float, default=0.00028, help="Learning rate to use.")
    parser.add_argument("--weight_decay", type=float, default=0.071, help="Weight decay.")

    args = parser.parse_args()

    return args


# get args
args = parse_args()

# update out
args.output_dir = Run.get_context().output_datasets['output_dir']

# update the config with args
config = vars(args)


os.environ['RAY_AIR_NEW_OUTPUT'] = '1'
#ray.init(num_gpus=1, ignore_reinit_error=True)
ray.init(
    runtime_env={
        "env_vars": {
            "RAY_AIR_LOCAL_CACHE_DIR": args.output_dir,
        },
        "working_dir": ".",
    }
)


# Parameter space
param_space = {
    "train_loop_config": {


        "lr": tune.loguniform(1e-8, 1e-3),  
        "emb_size": tune.choice([20, 40, 60, 80, 100,]),
        "transformer_config": {  
            "max_sequence_length": tune.choice([8]),
            "dim_model": tune.choice([128, 256, 384, 512, 640, 768]),
            "num_heads": tune.choice([1, 2, 4]),  
            "dim_feedforward": tune.choice([128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]),  
            "dropout": tune.uniform(0.0, 0.5),  
            "n_layers": tune.choice([1, 2, 3, 4, 5]),  
            "activation_function": tune.choice(["gelu", "relu"]),  
            "emb_dropout": tune.uniform(0.0, 0.5),
        },

        "weight_decay": tune.loguniform(1e-4, 0.2),

        "batch_size_per_device": tune.choice([32, 64, 128,
                                              256,
                                              288,
                                              320,
                                              352,
                                              384,
                                              416,
                                              448,
                                              480,
                                              512,
                                              544,
                                              576,
                                              608,
                                              640,
                                              672,
                                              704,
                                              736,
                                              768,
                                              800,
                                              832,
                                              864,
                                              896,
                                              928,
                                              960,
                                              992,
                                              1024,
                                              2056,
                                              1024*3,
                                              1024*4
                                              ]),

    }
}

# Scheduler
scheduler = ASHAScheduler(
    time_attr='epoch',
    max_t=config['num_epochs'],
    grace_period=5,
    reduction_factor=2,
    metric='precision',
    mode='max',
)

search_alg = HyperOptSearch(metric='precision', mode='max', points_to_evaluate=None)
search_alg = ConcurrencyLimiter(search_alg, max_concurrent=args.max_concurrent)

# Tune config
tune_config = tune.TuneConfig(
    search_alg=search_alg,
    scheduler=scheduler,
    num_samples=args.num_samples,
)

strategy = 'STRICT_PACK'
scaling_config = ScalingConfig(num_workers=args.num_workers, use_gpu=True, placement_strategy=strategy, resources_per_worker={"GPU": args.num_devices,})
run_config = RunConfig(storage_path=args.output_dir)


trainer = TorchTrainer(
    train_func,
    train_loop_config=config,
    scaling_config=scaling_config,
    run_config=run_config,
)

# Tuner
tuner = Tuner(
    trainable=trainer,
    param_space=param_space,
    tune_config=tune_config,
)

# Tune
results = tuner.fit()
best_trial = results.get_best_result(metric="precision", mode="max")
print("Best Result:", best_trial)

# get checkpoint
checkpoint = best_trial.checkpoint
print(f"Best Checkpoint: {checkpoint}")
print("Best Params", best_trial.config["train_loop_config"])
print("Training complete!")
print(best_trial.metrics_dataframe.to_dict())



