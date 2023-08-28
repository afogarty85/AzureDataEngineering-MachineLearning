from datasets import load_dataset
from torch.utils.data import DataLoader
import glob
import torch
import apex
import time
import torch.nn as nn
import pickle
import numpy as np
from sklearn.metrics import f1_score
import os
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(44)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


# load data in streaming mode
data_files = {"train": glob.glob('./data/train/*')}
my_iterable_dataset = load_dataset("parquet",
                                   split='train',
                                   data_files=data_files,
                                   streaming=True) \
                                   .with_format('numpy')

# shuffle
my_iterable_dataset = my_iterable_dataset.shuffle(seed=44, buffer_size=5000)


# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# cat cols
cat_cols = ['INPUT__REG__m_cmd_merge_mode_0_',
            'INPUT__REG__m_cmd_merge_mode_1_' ,
            'INPUT__REG__m_hazard_type',
            'INPUT__YML__testname',
            'INPUT__REG__m_ddr_speed_variant']


# collate
def my_collate(batch):

    # combine list of dicts
    bar = {
        k: np.stack([d.get(k) for d in batch])
        for k in set().union(*batch)
    }

    # x_cont
    x_cont = torch.stack([torch.tensor(v, dtype=torch.float) for k, v in bar.items()
                          if k not in cat_cols + ['SYNDROME', 'test_split']], dim=1)
    
    # force order of cat cols
    x_cat = torch.stack([torch.tensor(bar[k], dtype=torch.long) for k in cat_cols], dim=1)

    # y
    labels = torch.tensor(bar['SYNDROME'], dtype=torch.long)

    # package
    sample = {'x_cont': x_cont,
              'x_cat': x_cat,
              'labels': labels,
              }
    return sample


# model
class ConcatenatedEmbeddings(torch.nn.Module):
    """Map multiple categorical variables to concatenated embeddings.

    Args:
        embedding_table_shapes: A dictionary mapping column names to
            (cardinality, embedding_size) tuples.
        dropout: A float.
        sparse_columns: A list of sparse columns

    Inputs:
        x: An int64 Tensor with shape [batch_size, num_variables].

    Outputs:
        A Float Tensor with shape [batch_size, embedding_size_after_concat].
    """

    def __init__(self, embedding_table_shapes, dropout=0.0):
        super().__init__()

        self.embedding_layers = torch.nn.ModuleList(
            [
                torch.nn.Embedding(cat_size, emb_size,)
                for col, (cat_size, emb_size) in embedding_table_shapes.items()
            ]
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        if len(x.shape) <= 1:
            x = x.unsqueeze(0)
        x = [layer(x[:, i]) for i, layer in enumerate(self.embedding_layers)]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return x


class MLP(torch.nn.Module):
    """
    Generic Base Pytorch Model, that contains support for Categorical and Continuous values.

    Parameters
    ----------
    embedding_tables_shapes: dict
        A dictionary representing the <column>: <max cardinality of column> for all
        categorical columns.
    num_continuous: int
        Number of continuous columns in data.
    emb_dropout: float, 0 - 1
        Sets the embedding dropout rate.
    layer_hidden_dims: list
        Hidden layer dimensions.
    layer_dropout_rates: list
        A list of the layer dropout rates expressed as floats, 0-1, for each layer
    """

    def __init__(
        self,
        embedding_table_shapes,
        num_continuous,
        emb_dropout,
        layer_hidden_dims,
        layer_dropout_rates,
        num_classes,
    ):
        super().__init__()
        mh_shapes = None
        if isinstance(embedding_table_shapes, tuple):
            embedding_table_shapes, mh_shapes = embedding_table_shapes
        if embedding_table_shapes:
            self.initial_cat_layer = ConcatenatedEmbeddings(embedding_table_shapes, dropout=emb_dropout)
        self.initial_cont_layer = torch.nn.BatchNorm1d(num_continuous)

        embedding_size = sum(emb_size for _, emb_size in embedding_table_shapes.values())
        if mh_shapes is not None:
            embedding_size = embedding_size + sum(emb_size for _, emb_size in mh_shapes.values())
        layer_input_sizes = [embedding_size + num_continuous] + layer_hidden_dims[:-1]
        layer_output_sizes = layer_hidden_dims
        self.layers = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm1d(output_size),
                torch.nn.Dropout(dropout_rate),
            )
            for input_size, output_size, dropout_rate in zip(
                layer_input_sizes, layer_output_sizes, layer_dropout_rates
            )
        )

        self.output_layer = torch.nn.Linear(layer_output_sizes[-1], num_classes)

    def forward(self, x_cat, x_cont):
        concat_list = []
        if x_cat.dim() == 1:
            x_cat = x_cat.unsqueeze(1)
        # must use is not None for tensor, and len logic for empty list
        if x_cat is not None and len(x_cat) > 0:
            x_cat = self.initial_cat_layer(x_cat)
            concat_list.append(x_cat)
        if x_cont is not None and len(x_cont) > 0:
            x_cont = self.initial_cont_layer(x_cont)
            concat_list.append(x_cont)
        # if no layers in concat_list this breaks by design
        if len(concat_list) > 1:
            x = torch.cat(concat_list, 1)
        else:
            x = concat_list[0]
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


# set model params
# (max_val, n_features) -- max_val needs to be count(distinct(val)) +1 we can encounter
EMBEDDING_TABLE_SHAPES = {
                          'INPUT__REG__m_cmd_merge_mode_0_': (5, 25),
                          'INPUT__REG__m_cmd_merge_mode_1_': (5, 25),
                          'INPUT__REG__m_hazard_type': (6, 25),
                          'INPUT__YML__testname': (523, 100),
                          'INPUT__REG__m_ddr_speed_variant': (6, 25),
                          }
NUM_CLASSES = 67
EMBEDDING_DROPOUT_RATE = 0.01
DROPOUT_RATES = [0.01, 0.01]
HIDDEN_DIMS = [2000, 1000]
NUM_CONTINUOUS = 18
num_epochs = 1
batch_size = 1024

# init model
model = MLP(embedding_table_shapes=EMBEDDING_TABLE_SHAPES,
            num_continuous=NUM_CONTINUOUS,
            emb_dropout=EMBEDDING_DROPOUT_RATE,
            layer_hidden_dims=HIDDEN_DIMS,
            layer_dropout_rates=DROPOUT_RATES,
            num_classes=NUM_CLASSES
            ).to(device)


# use torch loaders
dataloader = DataLoader(my_iterable_dataset,
                        batch_size=batch_size,
                        num_workers=8,
                        prefetch_factor=4,
                        pin_memory=True,
                        collate_fn=my_collate,
                        drop_last=True,
                        )

# # debug
# for i, batch in enumerate(dataloader):
#     if i == 1: break

# optim
optimizer = apex.optimizers.FusedAdam(model.parameters(),
                                      adam_w_mode=True,
                                      lr=0.001,
                                      weight_decay=0.01)


# create weights
class_weights = torch.tensor([  0.01646575,   0.53236018,   1.15584174,   1.54778856,
                                    2.26079437,   3.46198553,   3.67448811,   4.2577765 ,
                                    5.08383489,   7.06892775,   7.47755662,  13.582093  ,
                                    14.3027387 ,  15.5879883 ,  16.5108291 ,  18.0089051 ,
                                    19.5361366 ,  22.3280369 ,  22.5298771 ,  22.7677123 ,
                                    23.300391  ,  24.9419628 ,  25.3140529 ,  25.4673478 ,
                                    36.9054405 ,  37.1935    ,  37.2162386 ,  37.6821687 ,
                                    39.0215551 ,  41.2122412 ,  44.8266579 ,  46.3294657 ,
                                    47.8696219 ,  48.9050825 ,  49.5832548 ,  63.2266322 ,
                                    65.2006657 ,  69.0364339 ,  71.0322071 ,  71.2441939 ,
                                    85.2316868 ,  97.177797  , 101.903497  , 102.448     ,
                                103.943655  , 113.050005  , 124.341728  , 135.714193  ,
                                148.983361  , 150.140835  , 158.383248  , 169.131351  ,
                                169.697126  , 184.53839   , 184.874653  , 187.133727  ,
                                193.76017   , 198.417867  , 200.047984  , 205.449212  ,
                                342.473145  , 343.390785  , 349.251873  , 374.382543  ,
                                402.21078   , 405.763049  , 437.239012  ],
                                                device=device, dtype=torch.float) \
                            / batch_size


# loss fn, weighted
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# # compile
model = torch.compile(model, mode='default')

scaler = torch.cuda.amp.GradScaler()

# train loop
for epoch in range(1, num_epochs + 1):

    # set shuffle
    my_iterable_dataset.set_epoch(epoch)

    # set train
    model.train()

    total_loss = 0.0
    completed_steps = 0
    n_correct = 0
    nb_examples = 0
    t0 = time.time()

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        #with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):

        # forward
        logits = model(x_cat=batch["x_cat"].to(device), x_cont=batch["x_cont"].to(device))

        # unpack labels
        labels = batch["labels"].to(device)

        # loss
        loss = loss_fn(logits, labels)

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # no-scale
        loss.backward()
        optimizer.step()

        # metrics
        total_loss += loss.detach().float().item()
        completed_steps += 1
        nb_examples += labels.size(0)

        # report progress
        if i % 100 == 0 and i != 0:
            print(f"Epoch: {epoch} | Step {i} | Loss: {(total_loss / completed_steps)} | Time: {time.time() - t0:.2f}")
            # reset
            t0 = time.time()

        # checkpoint
        if i % 10000 == 0 and i != 0:
            checkpoint = {"model": model.state_dict()}
            torch.save(checkpoint, f'./model_checkpoint/step_{completed_steps}_rte.pt')

    # save epoch state
    checkpoint = {"model": model.state_dict()}
    torch.save(checkpoint, f'./model_checkpoint/epoch_{epoch}_rte.pt')
    print(f"Epoch {epoch} Complete!")