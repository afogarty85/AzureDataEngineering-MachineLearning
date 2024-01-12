
# embedding generator
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
model = AutoModel.from_pretrained("thenlper/gte-large")
device = 'cuda'


class EmbeddingGenerator():
    '''
    Embed Chunker

    # init
    my_cls_chunker = EmbedChunks(model=model, pooling_mode='cls')
    cls_out = my_cls_chunker.forward(**batch_dict)
    '''

    def __init__(self, model, pooling_mode):
        self.model = model.to(device)
        self.model.eval()
        self.pooling_mode = pooling_mode

    def __call__(self, batch):

        x_input_ids = torch.stack([torch.as_tensor(x, device=device) for x in batch['x_input_ids']], dim=0)
        x_attention_mask = torch.stack([torch.as_tensor(x, device=device) for x in batch['x_attention_mask']], dim=0)

        y_input_ids = torch.stack([torch.as_tensor(x, device=device) for x in batch['y_input_ids']], dim=0)
        y_attention_mask = torch.stack([torch.as_tensor(x, device=device) for x in batch['y_attention_mask']], dim=0)

        with torch.no_grad():
            x_hidden_states = self.model(input_ids=x_input_ids,
                                        attention_mask=x_attention_mask).last_hidden_state
            
            y_hidden_states = self.model(input_ids=y_input_ids,
                                        attention_mask=y_attention_mask).last_hidden_state

        if self.pooling_mode == 'cls':
            x_cls_mask = x_input_ids.eq(tokenizer.cls_token_id)
            if len(torch.unique_consecutive(x_cls_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of CLS tokens.")

            y_cls_mask = y_input_ids.eq(tokenizer.cls_token_id)
            if len(torch.unique_consecutive(y_cls_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of CLS tokens.")
            
            # final hidden state of cls token
            x_sentence_representation = x_hidden_states[x_cls_mask, :].view(
                x_hidden_states.size(0), -1, x_hidden_states.size(-1))[:, -1, :]
            
            y_sentence_representation = y_hidden_states[y_cls_mask, :].view(
                y_hidden_states.size(0), -1, y_hidden_states.size(-1))[:, -1, :]
            
            # (Optionally) normalize embeddings
            x_embeddings = F.normalize(x_sentence_representation, p=2, dim=1)
            y_embeddings = F.normalize(y_sentence_representation, p=2, dim=1)

            return {'cos_sim': cos(x_sentence_representation, y_sentence_representation).cpu().numpy()}
            

        elif self.pooling_mode == 'avg':
            x_sentence_representation = x_hidden_states.masked_fill(
                ~x_attention_mask[..., None].bool(), 0.0)
            x_sentence_representation = x_sentence_representation.sum(dim=1) / x_attention_mask.sum(dim=1)[..., None]

            y_sentence_representation = y_hidden_states.masked_fill(
                ~y_attention_mask[..., None].bool(), 0.0)
            y_sentence_representation = y_sentence_representation.sum(dim=1) / y_attention_mask.sum(dim=1)[..., None]

            # (Optionally) normalize embeddings
            x_embeddings = F.normalize(x_sentence_representation, p=2, dim=1)
            y_embeddings = F.normalize(y_sentence_representation, p=2, dim=1)

            return {'cos_sim': cos(x_embeddings, y_embeddings).cpu().numpy()}


# tokenize custom
def preprocess_function(batch):

    x_inputs = tokenizer(list(batch['FaultComponent']), max_length=55, truncation=True, padding=True, return_tensors='np')
    y_inputs = tokenizer(list(batch['TechnicianFocus']), max_length=55, truncation=True, padding=True, return_tensors='np')

    return {
            'x_input_ids': x_inputs['input_ids'],
            'x_attention_mask': x_inputs['attention_mask'].astype(np.int64),
            'y_input_ids': y_inputs['input_ids'].astype(np.int64),
            'y_attention_mask': y_inputs['attention_mask'].astype(np.int64),
            }

ctx = ray.data.context.DatasetContext.get_current()
ctx.use_streaming_executor = True
ctx.execution_options.preserve_order = True  # keep idx order the same

tokenize_df = ray.data.from_pandas(df[['FaultComponent', 'TechnicianFocus']]).repartition(12)
tokenize_df = tokenize_df.map_batches(preprocess_function, batch_size=4096)


# generate emeddings in batches
embeddings = tokenize_df.map_batches(
    EmbeddingGenerator,
    fn_constructor_kwargs={"model": model, "pooling_mode": "avg"},
    batch_size=512,
    num_gpus=1,
    compute=ray.data.ActorPoolStrategy(size=1)
    )

# get embedds
x = embeddings.take_all()

# combine list of dicts
x2 = {
        k: np.stack([d.get(k) for d in x])
        for k in set().union(*x)
    }


# save
with open('./data/cosine_embeddings.pickle', 'wb') as handle:
    pickle.dump(x2, handle, protocol=pickle.HIGHEST_PROTOCOL)