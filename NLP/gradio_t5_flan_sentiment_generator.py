import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
import gradio as gr
import pandas as pd
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import ClientSecretCredential
import os

# choose model
model_choice = "google/flan-t5-large"

# load tokenizer/model
tokenizer = AutoTokenizer.from_pretrained(model_choice, use_fast=False, use_legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_choice)

# generator kwargs
gen_sample_kwargs = {
    "max_length": 24,
    "num_beams": 6,
    "remove_invalid_values": True,
    "num_return_sequences": 1,
    "do_sample": True,
    "temperature": 1,
    "top_p": 1,
    "early_stopping": True,
}

# init generator
generator = pipeline(task='text2text-generation', model=model, tokenizer=tokenizer, **gen_sample_kwargs)

# load env vars
CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']
TENANT = os.environ['TENANT']

# specify location to store files
account_url = 'https://lakeloc.dfs.core.windows.net'
file_system = 'filesystem'
cloud_path = '/SupportFiles/PredictionSaver'


def save_results():
    '''
    Save submissions of interest to AZ Datalake
    '''
    # date the file
    dating = str(pd.Timestamp("today", tz="UTC"))[:19].replace('-', '_').replace(':', '_').replace(' ', '_')

    # gen json
    json_out = tdf.to_json(orient="records", lines=True)

    # get access to AAD -- not using AIO here
    credential = ClientSecretCredential(os.environ['TENANT'], os.environ['CLIENT_ID'], os.environ['CLIENT_SECRET'])

    # get into lake
    datalake_service_client = DataLakeServiceClient(account_url=account_url, credential=credential)

    # get into fs container
    file_system_client = datalake_service_client.get_file_system_client(file_system)

    # create file client with a path to the exact file
    file_client = file_system_client.get_file_client(f"{cloud_path}/{dating}.json")

    # create / upload
    file_client.create_file()
    file_client.append_data(data=json_out, offset=0)
    file_client.flush_data(len(json_out))
    return gr.Textbox.update(value=f'{json_out}', visible=True)


def compute_fn(*args):
    '''
    Main computation fn with help from https://huggingface.co/spaces/joaogante/color-coded-text-generation/blob/main/app.py#L96
    '''
    # args is a tuple of vals
    user_input = args[0].rstrip('.')
    sentiment_options = args[1].split(',')

    # generate options
    options_list = '\n'.join([f'-{s.strip()}' for s in sentiment_options])

    # prompt
    prompt = f"{user_input}.\nWhat is the sentiment of this review?\nOPTIONS:\n{options_list}"

    # generate prediction
    out = generator(prompt)[0]['generated_text']

    # set prob buckets
    probs_to_label = [
        (0.7, "p >= 70%"),
        (0.4, "p >= 40%"),
        (0.1, "p < 10%"),
    ]

    # get scores
    inputs = tokenizer([prompt], return_tensors="pt")
    outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, **gen_sample_kwargs)

    # Important: don't forget to set `normalize_logits=True` to obtain normalized probabilities (i.e. sum(p) = 1)
    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
    transition_proba = np.exp(transition_scores)

    # We only have scores for the generated tokens, so pop out the prompt tokens
    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    generated_ids = outputs.sequences[:, input_length:]
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0], skip_special_tokens=True)

    # Important: you might need to find a tokenization character to replace (e.g. "Ġ" for BPE) and get the correct
    # spacing into the final output
    if model.config.is_encoder_decoder:
        highlighted_out = []
    else:
        input_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids)
        highlighted_out = [(token.replace("▁", " "), None) for token in input_tokens]
    # Get the (decoded_token, label) pairs for the generated tokens
    for token, proba in zip(generated_tokens, transition_proba[0]):
        this_label = None
        assert 0. <= proba <= 1.0
        for min_proba, label in probs_to_label:
            if proba >= min_proba:
                this_label = label
                break
        highlighted_out.append((token.replace("▁", " "), this_label))

    # turn to df / make available elsewhere
    global tdf

    # extract just the proba bucket
    prob_bucket = highlighted_out[0][1]
    tdf = pd.DataFrame({"user_input": [user_input], "prediction": [out], "prob": [prob_bucket]})

    # tuple for multiple gradio outputs
    return ([[user_input, out]], highlighted_out)


# set prob colors
label_to_color = {
    "p >= 70%": "green",
    "p >= 40%": "yellow",
    "p < 10%": "red"
}

# gradio app
with gr.Blocks() as demo:
    gr.Markdown(fr"""# Sentiment Generator

    ### Instructions:
    1. Input text to classify.
    2. Choose your own sentiments, for example: Bad, Ok, Pretty Good, Not Great.
    3. Click `Inference Button™` to get predicted sentiment.
    4. Click `Show Predicted Probabilities!` to show likelihood of its result.
    5. Write the prediction to Azure Datalake
    """)
    with gr.Tab("Generative Model"):
        with gr.Row():
            text_to_classify = gr.Textbox(
                label="Enter text to classify",
                value='The wine has straw yellow colour with elegant scent, hints of banana, pineapple, wild flowers and bread.')
            classify_options = gr.Textbox(
                label="Sentiment options", value='Positive, Negative, Neutral',
                info="Choose how to classify your text! Enter comma separated sentiments")
            inference_button = gr.Button("Inference Button™!", scale=1)

        with gr.Row():
            # main fn
            inference_button.click(compute_fn,
                                   inputs=[text_to_classify, classify_options],
                                   outputs=[gr.Dataframe(headers=['Text', 'Predicted'],
                                                         overflow_row_behaviour='paginate',
                                                         wrap=True,
                                                         max_cols=2,
                                                         col_count=(2, "fixed")),
                                            gr.HighlightedText(label="Predicted Probabilities",
                                                               combine_adjacent=True,
                                                               show_legend=True)
                                            .style(color_map=label_to_color)
                                            ]
                                   )
        with gr.Row():
            # store results
            deploy_button = gr.Button("Send to Lake!", scale=1)
            payload_result = gr.Textbox(label="Payload sent:", value=None, visible=False)
            deploy_button.click(save_results, outputs=payload_result)

# localhost
demo.queue(concurrency_count=1) \
    .launch(share=True,
            server_name="0.0.0.0",
            server_port=8181,
            )

# local dev
#demo.close()
