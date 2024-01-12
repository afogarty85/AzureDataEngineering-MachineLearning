import datasets
from transformers import (AutoTokenizer, default_data_collator, T5ForConditionalGeneration)
from transformers.optimization import Adafactor, get_scheduler
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from accelerate import Accelerator
torch.backends.cuda.matmul.allow_tf32 = True


# load sample data
dataset = datasets.load_dataset('imdb', num_proc=6) \
    .rename_column('label', 'labels') \
    .shuffle(seed=0)

# get num labels
num_classes = np.unique(dataset['test']['labels']).shape[0]

# model choice
model_choice = 'google/flan-t5-small'

# init tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_choice, use_fast=True)

# transform text to tokens
tokenized_datasets = dataset.map(function=lambda x: tokenizer(x['text'],
                                                              max_length=512,
                                                              truncation=True,
                                                              padding='max_length',
                                                              ),
                                 batched=True,
                                 num_proc=6,
                                 remove_columns='text',
                                 keep_in_memory=True,
                                 )

# add a categorical and numerical column
tokenized_datasets = tokenized_datasets.map(function=lambda x: {"x_cat": np.random.randint(55),
                                                                "x_cont": np.random.normal(1)},
                                            batched=False,
                                            num_proc=6,
                                            keep_in_memory=True,
                                            )

# init model
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small",
                                                   device_map='auto')


class T5ForSequenceClassification(torch.nn.Module):
    '''
    Custom T5 Model setup for sequence classification;
    Logic derived from HuggingFace BART / BART paper
    '''
    def __init__(self, model, num_classes, num_continuous, embedding_shapes, layer_dropout_rates, layer_hidden_dims):
        super(T5ForSequenceClassification, self).__init__()
        
        # load model
        self.model = model

        # embedding mod
        self.embedding_layers = torch.nn.ModuleList(
            [
                torch.nn.Embedding(cat_size, emb_size,)
                for col, (cat_size, emb_size) in embedding_shapes.items()
            ]
        )

        # get embedding size
        self.embedding_size = sum(emb_size for _, emb_size in embedding_shapes.values())
        self.layer_input_sizes = [self.embedding_size + num_continuous] + layer_hidden_dims[:-1]
        self.layer_output_sizes = layer_hidden_dims
        
        # initial x_cont layer
        self.initial_cont_layer = torch.nn.BatchNorm1d(num_continuous)

        # small FFNN
        self.layers = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm1d(output_size),
                torch.nn.Dropout(dropout_rate),
            )
            for input_size, output_size, dropout_rate in zip(
                self.layer_input_sizes, self.layer_output_sizes, layer_dropout_rates
            )
        )

        # final classification layer
        self.classifier = torch.nn.Linear(self.model.config.d_model + layer_hidden_dims[-1], num_classes)

    def forward(self, input_ids, attention_mask, decoder_input_ids, x_cat, x_cont):
        # decoder gets same input and encoder
        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            output_hidden_states=True)

        # last hidden decoder layer
        hidden_states = outputs.decoder_hidden_states[-1]

        # just the eos token
        eos_mask = input_ids.eq(self.model.config.eos_token_id)

        # final hidden state of final eos token embedding
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]

        # handle categorical
        if len(x_cat.shape) <= 1:
            # add another dim (n_cats, n_features)
            x_cat = x_cat.unsqueeze(1)

        # get continuous representation of x_cat (n_cats, n_features)
        x_cat = [layer(x_cat[:, i]) for i, layer in enumerate(self.embedding_layers)]

        # combine
        x_cat = torch.cat(x_cat, dim=1)

        # handle continuous
        if len(x_cont.shape) <= 1:
            # add another dim (bz, n_features)
            x_cont = x_cont.unsqueeze(1)

        # init cont layer
        x_cont = self.initial_cont_layer(x_cont)

        # combine (bz, n_features)
        x_cat_cont = torch.cat((x_cat, x_cont), dim=1)

        for layer in self.layers:
            # small FFNN
            x_cat_cont = layer(x_cat_cont)

        # combine with transformer (bz, n_features)
        joint_representation = torch.cat((x_cat_cont, sentence_representation), dim=1)

        # final classifier
        out = self.classifier(joint_representation)
        return out


# start accelerator
accelerator = Accelerator(mixed_precision='bf16',
                            gradient_accumulation_steps=1,
                              device_placement=True,
                               )

# init model
my_custom_t5 = T5ForSequenceClassification(model=model,
                                           num_classes=num_classes,
                                           num_continuous=1,
                                           embedding_shapes={'important_cat_1': (55, 18)},  # (max categories + 1, n_features)
                                           layer_dropout_rates=[0.01],
                                           layer_hidden_dims=[50]
                                           ).to(accelerator.device)

# adafactor
optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)

# scheduler
lr_scheduler = get_scheduler(name='constant',
                                optimizer=optimizer,
                            )

# loader
train_dataloader = torch.utils.data.DataLoader(tokenized_datasets['train'],
                                          num_workers=6,
                                          shuffle=True,
                                          drop_last=True,
                                          pin_memory=True,
                                          collate_fn=default_data_collator,
                                          batch_size=32)

# prepare items
my_custom_t5, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    my_custom_t5, optimizer, train_dataloader, lr_scheduler)

# loss fct
loss_fct = CrossEntropyLoss()

# basic stats
num_epochs = 2
total_loss = 0
completed_steps = 0
n_correct = 0
nb_examples = 0

# basic train loop
for epoch in range(1, num_epochs + 1):
    print(f" Starting epoch {epoch}")
    model.train()

    for step, batch in enumerate(train_dataloader):

        # forward
        with accelerator.accumulate(model):
            logits = my_custom_t5(input_ids=batch['input_ids'],
                                  attention_mask=batch['attention_mask'],
                                  decoder_input_ids=batch['input_ids'],
                                  x_cat=batch['x_cat'],
                                  x_cont=batch['x_cont'],
                                  )

            # loss
            labels = batch['labels']
            loss = loss_fct(logits.view(-1, num_classes), labels.view(-1))

            # backward
            accelerator.backward(loss)

            # update
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # metrics
            total_loss += loss.detach().float().item()
            completed_steps += 1
            nb_examples += labels.size(0)
            n_correct += (torch.max(logits, dim=1)[1].eq(labels)).cpu().sum().item()

            # track some progress
            if step % 20 == 0:
                print(f"Epoch: {epoch} | Step {step} | Loss: {(total_loss / completed_steps)} | Acc: {n_correct / nb_examples}")
