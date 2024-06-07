from azureml.core import Run
import ray
import ray.train as train
from ray.train import (
    Checkpoint,
    CheckpointConfig,
    DataConfig,
    RunConfig,
    ScalingConfig,
)
from ray.train.torch import TorchTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import pyarrow.compute as pc
from accelerate.utils import set_seed
from accelerate import Accelerator
import tqdm
import argparse
from transformers import get_cosine_schedule_with_warmup
import mlflow
import numpy as np
import pandas as pd
import os
from tempfile import TemporaryDirectory
np.set_printoptions(suppress=True)


# Best Params {'lr': 0.0009765578446116087, 'transformer_config': {'dim_model': 640, 'num_heads': 4, 'dim_feedforward': 1792, 'dropout': 0.032955656443587056, 'n_layers': 6, 'activation_function': 'relu', 'mlp_hidden_layers': (256, 512, 768)}, 'weight_decay': 0.00010025881134348113, 'batch_size_per_device': 960}


# init Azure ML run context data
aml_context = Run.get_context()
# args.output_dir = aml_context.output_datasets['output_dir']
# data_path = aml_context.input_datasets['train_files']


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--batch_size_per_device", type=int, default=256,
                        help="Batch size to use per device.")
    parser.add_argument("--num_devices", type=int, default=1, help="Number of devices to use.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use.")
    parser.add_argument("--seed", type=int, default=1, help="Seed.")
    parser.add_argument("--dim_model", type=int, default=512,
                        help="The dimensionality of the input embeddings and model")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--dim_feedforward", type=int, default=768,
                        help="The dimensionality of the feedforward network model in the Transformer encoder  ")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--activation_function", type=str, default='relu',
                        help="Activation fn for transformer block.")
    parser.add_argument("--mlp_hidden_layers", nargs='+', type=int,
                        default=[256, 512, 768], help="MLP feature transformer")
    parser.add_argument("--num_classes", type=int, default=1, help="Num labels.")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs to train for.")
    parser.add_argument("--output_dir", type=str,
                        default='/mnt/c/Users/afogarty/Desktop/AZRepairRL/checkpoints', help="Output dir.")
    parser.add_argument("--label_col", type=str, default='Label', help="label col")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate to use.")
    parser.add_argument("--dropout", type=float, default=0.032, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay.")
    args = parser.parse_args()

    return args


class EntityEmbeddingLayer(nn.Module):
    def __init__(self, embedding_table_shapes):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cat_size, embedding_dim=emb_size, padding_idx=0)
            for cat_size, emb_size in embedding_table_shapes.values()
        ])

    def forward(self, x):
        embeddings = [emb(x[:, :, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embeddings, dim=2)
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000):
        super().__init__()
        self.positional_embeddings = nn.Embedding(max_len, dim_model)
        self.max_len = max_len

    def forward(self, token_embeddings):
        batch_size, seq_length, _ = token_embeddings.size()
        positions = torch.arange(seq_length, dtype=torch.long,
                                 device=token_embeddings.device).unsqueeze(0)
        positional_encoding = self.positional_embeddings(positions)
        # Normalize token embeddings before adding positional encodings
        token_embeddings = F.layer_norm(token_embeddings, (token_embeddings.size(-1),))
        return token_embeddings + positional_encoding


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        for i, hidden_size in enumerate(hidden_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_layers[i-1], hidden_size))
            # Using layer normalization instead of batch normalization
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class AttentivePooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention_weight = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        attention_scores = self.attention_weight(x).squeeze(-1)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_sum = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
        return weighted_sum


class TabularTransformer(nn.Module):
    def __init__(self, config, embedding_table_shapes, num_binary=0, num_continuous=0, num_targets=1):
        super().__init__()
        # Configuration parameters
        self.config = config
        dim_model = config["dim_model"]
        self.embedding_layer = EntityEmbeddingLayer(embedding_table_shapes)
        self.embedding_size = sum(emb_size for _, emb_size in embedding_table_shapes.values())
        self.num_binary = num_binary
        self.num_continuous = num_continuous
        total_input_size = self.embedding_size + num_binary + num_continuous
        self.feature_projector = MLP(total_input_size, dim_model,
                                     config["mlp_hidden_layers"], config["dropout"])
        self.pos_encoder = LearnedPositionalEncoding(
            dim_model=dim_model, max_len=config["max_sequence_length"])
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=config["num_heads"], dim_feedforward=config[
                                                   "dim_feedforward"], dropout=config["dropout"], activation=config["activation_function"], batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config["n_layers"])
        self.attention_pooling = AttentivePooling(dim_model)
        self.regressor = nn.Linear(dim_model, num_targets)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x_cat, x_bin=None, x_cont=None, mask=None):
        x_cat = self.embedding_layer(x_cat)
        if x_bin is not None and x_cont is not None:
            # Layer normalization for continuous features
            x_cont = F.layer_norm(x_cont, (self.num_continuous,))
            x = torch.cat((x_cat, x_bin, x_cont), dim=2)
        else:
            x = x_cat
        x = self.feature_projector(x)
        x = self.pos_encoder(x)
        encoder_mask = ~mask if mask is not None else None
        x = self.transformer_encoder(x, src_key_padding_mask=encoder_mask)
        x = self.layer_norm(x)  # Apply layer normalization after the encoder
        x = self.attention_pooling(x, mask=encoder_mask)
        output = self.regressor(x)
        output = F.relu(output)
        return output


class GroupedDataFrameDataset(Dataset):
    def __init__(self, dataframe, group_key, cat_cols, binary_cols, cont_cols, label_cols, embeddings_col=None):
        self.dataframe = dataframe.sort_values(by=group_key)
        self.cat_cols = cat_cols
        self.binary_cols = binary_cols
        self.cont_cols = cont_cols
        self.label_cols = label_cols
        self.embeddings_col = embeddings_col
        self.group_key = group_key

        # Precompute the start and end indices for each group
        self.group_indices = {
            key: (min(indices), max(indices) + 1)
            for key, indices in self.dataframe.groupby(group_key).indices.items()
        }
        self.group_keys = list(self.group_indices.keys())

    def __len__(self):
        return len(self.group_keys)

    def __getitem__(self, idx):
        group_key = self.group_keys[idx]
        start_idx, end_idx = self.group_indices[group_key]
        group_data = self.dataframe.iloc[start_idx:end_idx]

        x_cat = torch.tensor(group_data[self.cat_cols].to_numpy(),
                             dtype=torch.long) if self.cat_cols else None
        x_bin = torch.tensor(group_data[self.binary_cols].to_numpy(),
                             dtype=torch.long) if self.binary_cols else None
        x_cont = torch.tensor(group_data[self.cont_cols].to_numpy(),
                              dtype=torch.float) if self.cont_cols else None
        labels = torch.tensor(group_data[self.label_cols].to_numpy(),
                              dtype=torch.float) if self.label_cols else None

        # Efficiently handle embeddings
        if self.embeddings_col is not None:
            # Convert list of ndarrays to a single ndarray
            embeddings_np = np.vstack(group_data[self.embeddings_col].to_numpy())
            # Convert the single ndarray to a PyTorch tensor
            embeddings = torch.tensor(embeddings_np, dtype=torch.float)
            # Concatenate continuous features and embeddings
            x_cont = torch.cat((x_cont, embeddings), dim=1) if x_cont is not None else embeddings

        return {
            'x_cat': x_cat,
            'x_bin': x_bin,
            'x_cont': x_cont,
            'labels': labels,
            'ResourceId': group_key,
            'sequence_length': len(group_data)
        }



  
def kendalls_tau(logits, labels):
    '''
    Tau-B -- matches scipy implementation --  accounts for ties in ranks
    '''

    n = logits.size(0)  
    pred_sign_matrix = torch.sign(logits.unsqueeze(1) - logits.unsqueeze(0))  
    actual_sign_matrix = torch.sign(labels.unsqueeze(1) - labels.unsqueeze(0))  
  
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()  
  
    pred_sign_matrix = pred_sign_matrix[mask]  
    actual_sign_matrix = actual_sign_matrix[mask]  
  
    concordant = (pred_sign_matrix == actual_sign_matrix) & (pred_sign_matrix != 0)  
    discordant = (pred_sign_matrix != actual_sign_matrix) & (pred_sign_matrix != 0) & (actual_sign_matrix != 0)  
  
    n_concordant_pairs = concordant.sum().item()  
    n_discordant_pairs = discordant.sum().item()  
  
    ties_pred = (pred_sign_matrix == 0) & (actual_sign_matrix != 0)  
    ties_actual = (actual_sign_matrix == 0) & (pred_sign_matrix != 0)  
    ties_both = (pred_sign_matrix == 0) & (actual_sign_matrix == 0)  
  
    n1 = ties_pred.sum().item() + ties_both.sum().item() / 2  
    n2 = ties_actual.sum().item() + ties_both.sum().item() / 2  
    n0 = mask.sum().item()  
  
    denominator = torch.sqrt(torch.tensor((n0 - n1) * (n0 - n2), dtype=torch.float))  
  
    if denominator == 0:  
        return float('nan')  
  
    tau_b = (n_concordant_pairs - n_discordant_pairs) / denominator  
    return tau_b.item()  




def mean_absolute_error(y_pred, y_true):
    """  
    Calculate the Mean Absolute Error (MAE) between predictions and true values.  

    Args:  
        y_pred (Tensor): Predicted values.  
        y_true (Tensor): Ground truth values.  

    Returns:  
        Tensor: The MAE value.  
    """
    mae = torch.mean(torch.abs(y_pred - y_true))
    return mae.item()


def calculate_embedding_sizes(cardinalities, min_size=10, max_size=50):
    """  
    Calculate embedding sizes based on the cardinalities of the categorical features.  

    Returns:  
    - A dictionary where the keys are feature names and the values are tuples (cardinality, embedding size).  
    """
    embedding_sizes = {}
    for feature, (cardinality, _) in cardinalities.items():
        # Apply the heuristic to the cardinality value
        emb_size = int(min(max_size, max(min_size, (cardinality // 2 + 1) ** 0.56)))
        embedding_sizes[feature] = (cardinality, emb_size)
    return embedding_sizes


def evaluate(model, eval_loader, loss_fn, accelerator):
    model.eval()
    losses = []
    kendal_taus = []
    maes = []

    with torch.no_grad():  # Move the no_grad context to cover the entire loop
        for step, batch in tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader)):

            # unpack
            inputs = {
                'x_cont': batch['x_cont'],
                'x_bin': batch['x_bin'],
                'x_cat': batch['x_cat'],
                'mask': batch['mask'],
            }
            labels = batch['labels'].squeeze()
            mask = batch['mask']

            # forward
            logits = model(**inputs).squeeze()

            # Calculate final timestep indices based on the mask and select the corresponding labels
            final_timestep_indices = (mask.sum(dim=1) - 1).long()
            final_labels = labels[final_timestep_indices]
            final_labels = labels[torch.arange(logits.size(0)), final_timestep_indices]

            # Compute avg batch loss
            loss = loss_fn(logits, final_labels)
            losses.append(accelerator.gather(loss))  # Gather loss for each batch

            # Calculate nDCG@k and MAE
            batch_mae = mean_absolute_error(logits, final_labels)
            batch_tau = kendalls_tau(logits, final_labels)

            maes.append(accelerator.gather(torch.tensor([batch_mae], device=accelerator.device)))
            kendal_taus.append(accelerator.gather(
                torch.tensor([batch_tau], device=accelerator.device)))

    # Concatenate the gathered results
    all_losses = torch.cat(losses)
    all_kendal_taus = torch.cat(kendal_taus)
    all_maes = torch.cat(maes)

    # Compute average metrics over all evaluation steps
    avg_eval_loss = all_losses.mean().item()
    avg_eval_tau = all_kendal_taus.mean().item()
    avg_eval_mae = all_maes.mean().item()

    # Print the evaluation metrics
    accelerator.print(f"Average Loss: {avg_eval_loss:.4f}")
    accelerator.print(f"Average Tau: {avg_eval_tau:.4f}")
    accelerator.print(f"Average MAE: {avg_eval_mae:.4f}")

    return avg_eval_loss, avg_eval_tau, avg_eval_mae


EMBEDDING_TABLE_SHAPES = {'FaultCode': (284, 25),
                          'DiscreteSkuMSFId': (1181, 25),
                          'RobertsRules': (4496, 25),
                          'Cluster': (9624, 25),
                          'TopFCHops': (178, 25),
                          'ClusterLabel': (126, 25),
                          'FCMSFRepair': (1679, 25),
                          'BusinessGroup': (11, 25),
                          'CFMAirFlow': (86, 25),
                          'Depth': (24, 25),
                          'Weight': (74, 25),
                          'TargetWorkload': (11, 25),
                          'Width': (17, 25)}

# emb shape
EMBEDDING_TABLE_SHAPES = calculate_embedding_sizes(EMBEDDING_TABLE_SHAPES)


cat_cols = list(EMBEDDING_TABLE_SHAPES.keys())


binary_cols = [
    'SP6RootCaused',
    'SevRepair',
    'KickedFlag',
    'InFleet',
    'Censored',
]

cont_cols = [
    'DeviceAge', 'FCMSFEntropy',
    'NumBOMParts', 'NumBOMSubstitutes', 'MaxBOMDepth', 'NumBOMRows', 'NumBlades',
    'NumberPrevRepairs',
]


label_cols = ['MinutesInProduction']


# train loop
def train_loop_per_worker(config):
    '''Train FN for every worker'''

    print("training_function called")

    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="fp16")

    def collate_fn(batch):
        batch_dict = {
            'x_cat': pad_sequence([item['x_cat'] for item in batch if item['x_cat'] is not None], batch_first=True),
            'x_bin': pad_sequence([item['x_bin'] for item in batch if item['x_bin'] is not None], batch_first=True),
            'x_cont': pad_sequence([item['x_cont'] for item in batch if item['x_cont'] is not None], batch_first=True),
            'labels': pad_sequence([item['labels'] for item in batch if item['labels'] is not None], batch_first=True, padding_value=-1),
            'ResourceId': [item['ResourceId'] for item in batch],
            'sequence_length': torch.tensor([item['sequence_length'] for item in batch], dtype=torch.long),
        }

        max_seq_len = max(batch_dict['sequence_length']).item()
        mask_batch = torch.arange(max_seq_len).expand(
            len(batch), max_seq_len) < batch_dict['sequence_length'].unsqueeze(1)

        batch_dict['mask'] = mask_batch
        return batch_dict

    # set seed
    set_seed(config['seed'])

    # Load Datasets
    # ====================================================
    train_set_df = pd.read_parquet(
        '/mnt/c/Users/afogarty/Desktop/AZRepairRL/data/seq_azrepair_train.parquet')
    test_set_df = pd.read_parquet(
        '/mnt/c/Users/afogarty/Desktop/AZRepairRL/data/seq_azrepair_test.parquet')

    # Prepare PyTorch Datasets
    # ====================================================

    train_set = GroupedDataFrameDataset(
        dataframe=train_set_df,
        group_key='ResourceId',
        cat_cols=cat_cols,
        binary_cols=binary_cols,
        cont_cols=cont_cols,
        label_cols=label_cols,
        embeddings_col='RobertsRulesEmbeddings',
    )
    

    test_set = GroupedDataFrameDataset(
        dataframe=test_set_df,
        group_key='ResourceId',
        cat_cols=cat_cols,
        binary_cols=binary_cols,
        cont_cols=cont_cols,
        label_cols=label_cols,
        embeddings_col='RobertsRulesEmbeddings',
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
        collate_fn=collate_fn,
    )

    eval_dataloader = DataLoader(
        test_set,
        batch_size=config['batch_size_per_device'],
        shuffle=False,
        num_workers=8,
        prefetch_factor=2,
        drop_last=False,
        collate_fn=collate_fn,
    )


    # Model
    # ====================================================

    max_seq_len = int(train_set_df.groupby(['ResourceId']).size().max())

    transformer_config = {
        "max_sequence_length": max_seq_len,
        "dim_model": config['dim_model'],
        "num_heads": config['num_heads'],
        "dim_feedforward": config['dim_feedforward'],
        "dropout": config['dropout'],
        "n_layers": config['n_layers'],
        "activation_function": config['activation_function'],
        "mlp_hidden_layers": config['mlp_hidden_layers'],
    }

    embeddings_dim = 256
    model = TabularTransformer(config=transformer_config,
                               embedding_table_shapes=EMBEDDING_TABLE_SHAPES,
                               num_binary=len(binary_cols),
                               num_continuous=len(cont_cols) + embeddings_dim,
                               num_targets=len(label_cols)
                               )

    # proportion of warmup steps (e.g., 10% of the total training steps)
    warmup_ratio = 0.1

    # total number of batches per epoch
    num_train_steps_per_epoch = len(train_dataloader)

    # total number of training steps
    num_training_steps = num_train_steps_per_epoch * config['num_epochs']

    # number of warmup steps
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    loss_fn = torch.nn.MSELoss()

    # prepare
    model, optimizer, train_dataloader, eval_dataloader, loss_fn, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, loss_fn, scheduler)

    # train the model
    if accelerator.is_main_process:
        print("Starting training ...")
        print("Number of batches on main process", num_train_steps_per_epoch)

    for epoch in range(1, config['num_epochs'] + 1):

        model.train()
        losses = []
        interval_loss = 0.0
        running_tau = 0.0
        running_mae = 0.0

        for step, batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # zero
            optimizer.zero_grad()

            # unpack
            inputs = {
                'x_cont': batch['x_cont'],
                'x_bin': batch['x_bin'],
                'x_cat': batch['x_cat'],
                'mask': batch['mask'],
            }
            labels = batch['labels'].squeeze()
            mask = batch['mask']

            # forward
            logits = model(**inputs).squeeze()

            # Calculate final timestep indices based on the mask and select the corresponding labels
            final_timestep_indices = (mask.sum(dim=1) - 1).long()
            final_labels = labels[torch.arange(logits.size(0)), final_timestep_indices]

            # Compute the avg batch loss
            loss = loss_fn(logits, final_labels)
            losses.append(accelerator.gather(loss))  # Gather loss for each batch

            # backward pass and optimize
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # update
            scheduler.step()

            # Calculate MAE and TAU
            batch_mae = mean_absolute_error(logits, final_labels)
            batch_tau = kendalls_tau(logits, final_labels)

            # update running stats
            interval_loss += loss.item()
            running_tau += batch_tau
            running_mae += batch_mae

            # Check if it's time to print the loss (every 100 batches)
            if (step + 1) % 100 == 0:
                if accelerator.is_main_process:

                    mlflow.log_metric('interval_loss', interval_loss / 100)
                    mlflow.log_metric('interval_tau', running_tau / 100)
                    mlflow.log_metric('interval_mae', running_mae / 100)

                    accelerator.print(
                        f"Epoch {epoch}/{config['num_epochs']}, Batch {step+1}, Interval Loss: {interval_loss / 100:.4f}, Tau: {running_tau / 100:.4f},  MAE: {running_mae / 100:.4f}")
                    interval_loss = 0.0  # reset after printing
                    running_tau = 0.0
                    running_mae = 0.0

        # cat the gathered results
        all_losses = torch.cat(losses)

        # average
        train_epoch_loss = all_losses.mean().item()
        print(f"Epoch {epoch} / {config['num_epochs']}, Total Train Loss: {train_epoch_loss}")

        print("Running evaluation ...")
        accelerator.wait_for_everyone()

        # eval
        avg_eval_loss, avg_eval_tau, avg_eval_mae = evaluate(
            model=model,
            eval_loader=eval_dataloader,
            loss_fn=loss_fn,
            accelerator=accelerator,
        )
        if accelerator.is_main_process:
            mlflow.log_metric('eval_loss', avg_eval_loss)
            mlflow.log_metric('eval_tau', avg_eval_tau)
            mlflow.log_metric('eval_mae', avg_eval_mae)

        metrics = {
            "epoch": epoch,
            "iteration": step,
            "train_loss": train_epoch_loss,
            "eval_tau": avg_eval_tau,
            "eval_loss": avg_eval_loss,
            "eval_mae": avg_eval_mae,
            "num_iterations": step + 1,
        }

        accelerator.print(f"Saving the model locally at {args.output_dir}")

        with TemporaryDirectory() as tmpdir:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if accelerator.process_index == 0:
                    # This will call the unwrap model as well
                    state = accelerator.get_state_dict(model)
                    accelerator.save(state, f"{tmpdir}/epoch_{epoch}_state_dict.pt")
                    checkpoint = Checkpoint.from_directory(tmpdir)
            else:
                checkpoint = None
            train.report(metrics=metrics, checkpoint=checkpoint)


if __name__ == '__main__':

    # get args
    args = parse_args()

    if not ray.is_initialized():
        # init a driver
        ray.init(runtime_env={
            "env_vars": {
                "RAY_AIR_LOCAL_CACHE_DIR": args.output_dir,
                "RAY_memory_usage_threshold": ".95"
            },
        }
        )
        print('ray initialized')

    # cluster available resources which can change dynamically
    print(f"Ray cluster resources_ {ray.cluster_resources()}")

    # update the config with args so that we have access to them.
    config = vars(args)

    # placement strategy
    strategy = "SPREAD"

    # keep the 1 checkpoint
    checkpoint_config = CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="eval_tau",
        checkpoint_score_order="max"
    )

    # configs
    run_config = RunConfig(callbacks=None, checkpoint_config=checkpoint_config,
                           storage_path=args.output_dir)
    scaling_config = ScalingConfig(num_workers=args.num_workers, use_gpu=True,
                                   placement_strategy=strategy, resources_per_worker={"GPU": args.num_devices, })
    options = ray.data.ExecutionOptions(preserve_order=True)
    dataset_config = DataConfig(datasets_to_split=["train"], execution_options=options)

    # trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=run_config,
        dataset_config=dataset_config,
    )

    # fit
    result = trainer.fit()

    print("Training complete!")
