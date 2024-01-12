from transformers import DataCollatorWithPadding
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor, get_scheduler

# init tokenizer / model / n_classes
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", use_fast=False, use_legacy=False)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
num_classes = 7

class T5ForSequenceClassification(torch.nn.Module):
    '''
    Custom T5 Model setup for sequence classification;
    Logic derived from HuggingFace BART
    '''
    def __init__(self, model):
        super(T5ForSequenceClassification, self).__init__()
        
        # load model
        self.l1 = model

        # final classification layer
        self.classifier = torch.nn.Linear(self.l1.config.d_model, num_classes)

    def forward(self, input_ids, attention_mask, decoder_input_ids, output_hidden_states):
        # generate outputs
        outputs = self.l1(input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=input_ids,
                            output_hidden_states=True)

        # last hidden decoder layer
        hidden_states = outputs.decoder_hidden_states[-1]

        # just the eos token
        eos_mask = input_ids.eq(self.l1.config.eos_token_id)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")

        # final hidden state of final eos token sent into classifier
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
        logits = self.classifier(sentence_representation)
        return logits

# init model
my_custom_t5 = T5ForSequenceClassification(model=model)

# adafactor
optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)

# collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# scheduler
lr_scheduler = get_scheduler(name='constant',
                                optimizer=optimizer,
                            )

# example forward -- decoder also gets input
logits = my_custom_t5(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            decoder_input_ids=batch['input_ids'],
                            output_hidden_states=True)

# cont
with torch.inference_mode():
    logits = my_custom_t5(input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    decoder_input_ids=batch['input_ids'],
                    output_hidden_states=True)

    labels = batch['labels']

nb_val_examples += labels.size(0)

# stats
probs = F.softmax(logits, dim=1)
vals, idx = torch.max(probs, dim=1)
acc_fn(idx, labels)