# applied_nlp_demos in PyTorch


# Modern NLP

1. Pretrain T5 v1.1: Pre-Train T5 on C4 Dataset [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/accelerate_pretrain_t5_base_mlm.py). This code is exceedingly less complicated, more readable, and truer to Google's implementation, than other available options thanks to HuggingFace. In comparison to the [T5 1.1 paper](https://arxiv.org/pdf/2002.05202.pdf) which reports 1.942 loss at 65,536 steps, a single RTX 4090 produces comparable results on the test set (2.08) in roughly 18.5 hours of training using this code (see image below). Pretraining on your own data set is as simple as swapping out the existing `Dataset` with your own.

![T5 Pretraining Loss](./images/pretrain_loss.png)

2. Seq2Seq (ChatBot): Fine Tune Flan-T5 on Alpaca [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/accelerate_deepspeed_alpaca_t5_flan_finetune.py)

3. Seq2Seq: Fine Tune Flan-T5 on Data Using HuggingFace Dataset Framework [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/accelerate_deepspeed_t5_flan_hfdata.py)


### google/flan-t5-large

```
input sentence: Given a set of numbers, find the maximum value.
{10, 3, 25, 6, 16}
response: 25

input sentence: Convert from celsius to fahrenheit.
Temperature in Celsius: 15
response: Fahrenheit

input sentence: Arrange the given numbers in ascending order.
2, 4, 0, 8, 3
response: 0, 3, 4, 8

input sentence: What is the capital of France?
response: paris

input sentence: Name two types of desert biomes.
response: sahara
```

### google/flan-t5-large: Fine-tuned on Alpaca

```
input sentence: Given a set of numbers, find the maximum value.
{10, 3, 25, 6, 16}
response: 25

input sentence: Convert from celsius to fahrenheit.
Temperature in Celsius: 15
response: 77

input sentence: Arrange the given numbers in ascending order.
2, 4, 0, 8, 3
response: 0, 2, 3, 4, 8

input sentence: What is the capital of France?
response: Paris

input sentence: Name two types of desert biomes.
response: Desert biomes can be divided into two main types: arid and semi-arid. Arid deserts are characterized by high levels of deforestation, sparse vegetation, and limited water availability. Semi-desert deserts, on the other hand, are relatively dry deserts with little to no vegetation.
```



# Legacy NLP

This repository contains start-to-finish data processing and NLP algorithms using PyTorch and often HuggingFace (Transformers) for the following models:

1. [Paper: Hierarchical Attention Networks](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)  PyTorch Implementation: [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/HAN.py)

2. [Paper: BERT](https://arxiv.org/pdf/1810.04805.pdf?source=post_elevate_sequence_page)  PyTorch Implementation: [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/bert.py)

3. BERT-CNN Ensemble. PyTorch Implementation: [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/bert_cnn.py)

4. [Paper: Character-level CNN](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)  PyTorch Implementation: [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/char_cnn.py)

5. [Paper: DistilBERT](https://arxiv.org/pdf/1910.01108.pdf)  PyTorch Implementation: [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/distilbert.py)

6. DistilGPT-2.  PyTorch Implementation: [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/distilgpt2_generation.py)

7. [Paper: Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf?source=post_page)  PyTorch Implementation: [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/kim_cnn.py)

8. [Paper: T5-Classification](https://arxiv.org/pdf/1910.10683.pdf)  PyTorch Implementation: [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/t5_classification.py)

9. [Paper: T5-Summarization](https://arxiv.org/pdf/1910.10683.pdf)  PyTorch Implementation: [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/t5_conditional_generation.py)

10. Building a Corpus: Search Text Files. Code: [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/search_text_files.py)

11. [Paper: Heinsein Routing](https://arxiv.org/abs/1911.00792)  TorchText Implementation: [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/BERT_capsule.py)

12. Entity Embeddings and Lazy Loading. Code: [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/torch_dataset.py)

13. Semantic Similarity. Code: [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/semantic_sim.py)

14. SQuAD 2.0 BERT Embeddings Emissions in PyTorch. Code: [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/squad_embeds.py)

15. SST-5 BERT Embeddings Emissions in PyTorch. Code: [Code](https://github.com/afogarty85/applied_nlp_demos/blob/master/sst_embeds.py)

Credits: The [Hedwig](https://github.com/castorini/hedwig) group has been instrumental in helping me learn many of these models.

Nicer-looking R Markdown outputs can be found here: http://seekinginference.com/NLP/
