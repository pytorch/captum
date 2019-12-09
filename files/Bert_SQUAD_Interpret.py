#!/usr/bin/env python
# coding: utf-8

# # Interpreting BERT Models (Part 1)

# In this notebook we demonstrate how to interpret Bert models using  `Captum` library. In this particular case study we focus on a fine-tuned Question Answering model on SQUAD dataset using transformers library from Hugging Face: https://huggingface.co/transformers/
# 
# We show how to use interpretation hooks to examine and better understand embeddings, sub-embeddings, bert, and attention layers. 
# 
# Note: Before running this tutorial, please install `seaborn`, `pandas` and `matplotlib` python packages.

# In[1]:


import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig

from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer


# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# The first step is to fine-tune BERT model on SQUAD dataset. This can be easiy accomplished by following the steps described in hugging face's official web site: https://github.com/huggingface/transformers#run_squadpy-fine-tuning-on-squad-for-question-answering 

# After we pretrain the model, we can load the tokenizer and pre-trained BERT model using the commands described below. 

# In[3]:


# replace <PATH-TO-SAVED-MODEL> with the real path of the saved model

model_path = '<PATH-TO-SAVED-MODEL>'

# load model
model = BertForQuestionAnswering.from_pretrained(model_path)
model.to(device)
model.eval()
model.zero_grad()

# load tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)


# A helper function to perform forward pass of the model and make predictions.

# In[4]:


def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    return model(inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, )


# Defining a custom forward function that will allow us to access the start and end postitions of our prediction using `position` input argument.

# In[5]:


def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
    pred = predict(inputs,
                   token_type_ids=token_type_ids,
                   position_ids=position_ids,
                   attention_mask=attention_mask)
    pred = pred[position]
    return pred.max(1).values


# Let's interpret parent `BertEmbeddings` layer. For that reason we separate embedding layer and precompute the embeddings. `configure_interpretable_embedding_layer` function helps us to do that. It returns `InterpretableEmbeddingBase` layer that wraps `BertEmbeddings` and can be used to access the embedding vectors. Note that we need to remove `InterpretableEmbeddingBase` wrapper from our model using `remove_interpretable_embedding_layer` function after we finish interpretation.
# 
# Note that there is a way to interpret BertEmbeddings without configuring interpretable embedding layers that we will make available soon, However it is important to note that configuring interpretable embedding layers gives us more flexibility.

# In[ ]:


interpretable_embedding = configure_interpretable_embedding_layer(model, 'bert.embeddings')


# Defining and numericalizing / encoding special tokens

# In[7]:


ref = "[PAD]" # A token used for generating token reference
sep = "[SEP]" # A token used as a separator between question and text and it is also added to the end of the text.
cls = "[CLS]" # This token is prepended to the concatenated question-text word sequence

ref_token_id = tokenizer.encode(ref)
sep_token_id = tokenizer.encode(sep)
cls_token_id = tokenizer.encode(cls)


# Below we define a set of helper function for constructing references / baselines for word tokens, token types and position ids. We also provide separate helper functions that allow to construct the sub-embeddings and corresponding references for all sub-embeddings of `BertEmbeddings` layer.

# In[8]:


def construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id):
    question_ids = tokenizer.encode(question)
    text_ids = tokenizer.encode(text)

    # construct input token ids
    input_ids = cls_token_id + question_ids + sep_token_id + text_ids + sep_token_id

    # construct reference token ids 
    ref_input_ids = cls_token_id + ref_token_id * len(question_ids) + sep_token_id +         ref_token_id * len(text_ids) + sep_token_id

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(question_ids)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def construct_bert_sub_embedding(input_ids, ref_input_ids,
                                   token_type_ids, ref_token_type_ids,
                                   position_ids, ref_position_ids):
    input_embeddings = interpretable_embedding1.indices_to_embeddings(input_ids)
    ref_input_embeddings = interpretable_embedding1.indices_to_embeddings(ref_input_ids)

    input_embeddings_token_type = interpretable_embedding2.indices_to_embeddings(token_type_ids)
    ref_input_embeddings_token_type = interpretable_embedding2.indices_to_embeddings(ref_token_type_ids)

    input_embeddings_position_ids = interpretable_embedding3.indices_to_embeddings(position_ids)
    ref_input_embeddings_position_ids = interpretable_embedding3.indices_to_embeddings(ref_position_ids)
    
    return (input_embeddings, ref_input_embeddings),            (input_embeddings_token_type, ref_input_embeddings_token_type),            (input_embeddings_position_ids, ref_input_embeddings_position_ids)
    
def construct_whole_bert_embeddings(input_ids, ref_input_ids,                                     token_type_ids=None, ref_token_type_ids=None,                                     position_ids=None, ref_position_ids=None):
    input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    
    return input_embeddings, ref_input_embeddings


# Let's define the `question - text` pair that we'd like to use as an input for our Bert model and interpret what the model was forcusing on when predicting an answer to the question from given input text 

# In[9]:


question, text = "What is important to us?", "It is important to us to include, empower and support humans of all kinds."

ground_truth = 'to include, empower and support humans of all kinds'


# Let's numericalize the question, the input text and generate corresponding references for all three sub-embedding (word, token type and position embeddings) types using our helper functions defined above.

# In[10]:


input_ids, ref_input_ids, sep_id = construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id)
token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
attention_mask = construct_attention_mask(input_ids)


# In[11]:


indices = input_ids[0].detach().tolist()
all_tokens = tokenizer.convert_ids_to_tokens(indices)

ground_truth_tokens = tokenizer.encode(ground_truth)
ground_truth_end_ind = indices.index(ground_truth_tokens[-1])
ground_truth_start_ind = ground_truth_end_ind - len(ground_truth_tokens) + 1


# Now let's compute the embedding vectors for `BertEmbedding` and make predictions using pre-computed embeddings both for the input and the reference

# In[12]:


input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(input_ids, ref_input_ids,                                          token_type_ids=token_type_ids, ref_token_type_ids=ref_token_type_ids,                                          position_ids=position_ids, ref_position_ids=ref_position_ids)

start_scores, end_scores = predict(input_embeddings,                                    token_type_ids=token_type_ids,                                    position_ids=position_ids,                                    attention_mask=attention_mask)


print('Question: ', question)
print('Predicted Answer: ', ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))


# In[13]:


ig = IntegratedGradients(squad_pos_forward_func)

attributions_start, delta_start = ig.attribute(inputs=input_embeddings,
                                  baselines=ref_input_embeddings,
                                  additional_forward_args=(token_type_ids, position_ids,attention_mask, 0),
                                  return_convergence_delta=True, n_steps=200)
attributions_end, delta_end = ig.attribute(inputs=input_embeddings, baselines=ref_input_embeddings,
                                additional_forward_args=(token_type_ids, position_ids, attention_mask, 1),
                                return_convergence_delta=True, n_steps=200)


# Let's remove interpretation hook from embeddings after we finished interpretation

# In[14]:


remove_interpretable_embedding_layer(model, interpretable_embedding)


# A helper function to summarize attributions for each word token in the sequence.

# In[15]:


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


# In[16]:


attributions_start_sum = summarize_attributions(attributions_start)
attributions_end_sum = summarize_attributions(attributions_end)


# In[17]:


# storing couple samples in an array for visualization purposes
start_position_vis = viz.VisualizationDataRecord(
                        attributions_start_sum,
                        torch.max(torch.softmax(start_scores[0], dim=0)),
                        torch.argmax(start_scores),
                        torch.argmax(start_scores),
                        str(ground_truth_start_ind),
                        attributions_start_sum.sum(),       
                        all_tokens,
                        delta_start)

end_position_vis = viz.VisualizationDataRecord(
                        attributions_end_sum,
                        torch.max(torch.softmax(end_scores[0], dim=0)),
                        torch.argmax(end_scores),
                        torch.argmax(end_scores),
                        str(ground_truth_end_ind),
                        attributions_end_sum.sum(),       
                        all_tokens,
                        delta_end)

print('\033[1m', 'Visualizations For Start Position', '\033[0m')
viz.visualize_text([start_position_vis])

print('\033[1m', 'Visualizations For End Position', '\033[0m')
viz.visualize_text([end_position_vis])


# In[1]:


from IPython.display import Image
Image(filename='img/bert/visuals_of_start_end_predictions.png')


# From the results above we can tell that for predicting start position our model is focusing more on the question side. More specifically on the tokens `what` and `important`. It has also slight focus on the token sequence `to us to` in the text side.
# 
# In contrary to that, for predicting end position, our model focuses more on the text side and has relative high attribution on the last end position token `kinds`.

# # Multi-Embedding attribution

# Now let's look into the sub-embeddings of `BerEmbeddings` and try to understand the contributions and roles of each of them for both start and end predicted positions.
# 
# To do so we'd need to place interpretation hooks in each three of them.

# In[18]:


interpretable_embedding1 = configure_interpretable_embedding_layer(model, 'bert.embeddings.word_embeddings')
interpretable_embedding2 = configure_interpretable_embedding_layer(model, 'bert.embeddings.token_type_embeddings')
interpretable_embedding3 = configure_interpretable_embedding_layer(model, 'bert.embeddings.position_embeddings')


# `BertEmbeddings` has three sub-embeddings, namely, `word_embeddings`, `token_type_embeddings` and `position_embeddings` and this time we would like to attribute to each of them independently.
# `construct_bert_sub_embedding` helper function helps us to construct input embeddings and corresponding references in a separation.

# In[19]:


(input_embed, ref_input_embed), (token_type_ids_embed, ref_token_type_ids_embed), (position_ids_embed, ref_position_ids_embed) = construct_bert_sub_embedding(input_ids, ref_input_ids,                                          token_type_ids=token_type_ids, ref_token_type_ids=ref_token_type_ids,                                          position_ids=position_ids, ref_position_ids=ref_position_ids)


# Now let's compute the attributions with respect to all those embeddings both for the start and end positions and summarize them for each word token.

# In[20]:


attributions_start = ig.attribute(inputs=(input_embed, token_type_ids_embed, position_ids_embed),
                                  baselines=(ref_input_embed, ref_token_type_ids_embed, ref_position_ids_embed),
                                  additional_forward_args=(attention_mask, 0))
attributions_end = ig.attribute(inputs=(input_embed, token_type_ids_embed, position_ids_embed),
                                  baselines=(ref_input_embed, ref_token_type_ids_embed, ref_position_ids_embed),
                                  additional_forward_args=(attention_mask, 1))

attributions_start_word = summarize_attributions(attributions_start[0])
attributions_end_word = summarize_attributions(attributions_end[0])

attributions_start_token_type = summarize_attributions(attributions_start[1])
attributions_end_token_type = summarize_attributions(attributions_end[1])

attributions_start_position = summarize_attributions(attributions_start[2])
attributions_end_position = summarize_attributions(attributions_end[2])


# An auxilary function that will help us to compute topk attributions and corresponding indices

# In[21]:


def get_topk_attributed_tokens(attrs, k=5):
    values, indices = torch.topk(attrs, k)
    top_tokens = [all_tokens[idx] for idx in indices]
    return top_tokens, values, indices


# Removing interpretation hooks from all layers after finishing attribution.

# In[22]:


remove_interpretable_embedding_layer(model, interpretable_embedding1)
remove_interpretable_embedding_layer(model, interpretable_embedding2)
remove_interpretable_embedding_layer(model, interpretable_embedding3)


# Computing topk attributions for all sub-embeddings and placing them in pandas dataframes for better visualization.

# In[23]:


top_words_start, top_words_val_start, top_word_ind_start = get_topk_attributed_tokens(attributions_start_word)
top_words_end, top_words_val_end, top_words_ind_end = get_topk_attributed_tokens(attributions_end_word)

top_token_type_start, top_token_type_val_start, top_token_type_ind_start = get_topk_attributed_tokens(attributions_start_token_type)
top_token_type_end, top_token_type_val_end, top_token_type_ind_end = get_topk_attributed_tokens(attributions_end_token_type)

top_pos_start, top_pos_val_start, pos_ind_start = get_topk_attributed_tokens(attributions_start_position)
top_pos_end, top_pos_val_end, pos_ind_end = get_topk_attributed_tokens(attributions_end_position)

df_start = pd.DataFrame({'Word(Index), Attribution': ["{} ({}), {}".format(word, pos, round(val.item(),2)) for word, pos, val in zip(top_words_start, top_word_ind_start, top_words_val_start)],
                   'Token Type(Index), Attribution': ["{} ({}), {}".format(ttype, pos, round(val.item(),2)) for ttype, pos, val in zip(top_token_type_start, top_token_type_ind_start, top_words_val_start)],
                   'Position(Index), Attribution': ["{} ({}), {}".format(position, pos, round(val.item(),2)) for position, pos, val in zip(top_pos_start, pos_ind_start, top_pos_val_start)]}).style.set_caption("Top 5 attributed embeddings for start position")

df_end = pd.DataFrame({'Word(Index), Attribution': ["{} ({}), {}".format(word, pos, round(val.item(),2)) for word, pos, val in zip(top_words_end, top_words_ind_end, top_words_val_end)],
                   'Token Type(Index), Attribution': ["{} ({}), {}".format(ttype, pos, round(val.item(),2)) for ttype, pos, val in zip(top_token_type_end, top_token_type_ind_end, top_words_val_end)],
                   'Position(Index), Attribution': ["{} ({}), {}".format(position, pos, round(val.item(),2)) for position, pos, val in zip(top_pos_end, pos_ind_end, top_pos_val_end)]}).style.set_caption("Top 5 attributed embeddings for end position")

['{}({})'.format(token, str(i)) for i, token in enumerate(all_tokens)]


# Below we can see top 5 attribution results from all three embedding types in predicting start positions.
# 
# Word embeddings help to focus more on the the start position answer `to` and its surrounding tokens. It also has relatively high attribution for the tokens in the question such as `it` and `important`.
# 
# In contrary to word embedding, token embedding type focuses more on the tokens in the text part such as `include`,`em` and `and`.
# 
# Position embedding also has high attribution for correctly predicted start token `to`, including some other tokens in the answer such as `humans` and `##power`.
# 

# In[24]:


df_start


# We can perform similar analysis, and visualize top 5 attributed tokens for all three embedding types, also for the end position prediction.
# 
# It is interesting to observe high concentration of highly attributed tokens such as `.`, `of`, `kinds`, `humans` and `[SEP]` for end position prediction.
# 
# The token `kinds`, which is the correct predicted token appears to have high attribution score both according word and position embeddings.
# 

# In[25]:


df_end


# # Interpreting Bert Layers

# Now let's look into the layers of our network. More specifically we would like to look into the distribution of attribution scores for each token across all layers in Bert model and dive deeper into specific tokens.  
# We do that using one of layer attribution algorithms, namely, layer conductance. However, we encourage you to try out and compare the results with other algorithms as well.
# 
# 
# Let's configure `InterpretableEmbeddingsBase` again, in this case in order to interpret the layers of our model.

# In[26]:


interpretable_embedding = configure_interpretable_embedding_layer(model, 'bert.embeddings')


# Let's iterate over all layers and compute the attributions for all tokens. In addition to that let's also choose a specific token that we would like to examine in detail, specified by an id `token_to_explain` and store related information in a separate array.
# 
# 
# Note: Since below code is iterating over all layers it can take over 10 seconds. Please be patient!

# In[27]:


layer_attrs_start = []
layer_attrs_end = []

# The token that we would like to examine separately.
token_to_explain = 23 # the index od the token
layer_attrs_start_dist = []
layer_attrs_end_dist = []

for i in range(model.config.num_hidden_layers):
    lc = LayerConductance(squad_pos_forward_func, model.bert.encoder.layer[i])
    layer_attributions_start = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, additional_forward_args=(token_type_ids, position_ids,attention_mask, 0))
    layer_attributions_end = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, additional_forward_args=(token_type_ids, position_ids,attention_mask, 1))

    layer_attrs_start.append(summarize_attributions(layer_attributions_start).cpu().detach().tolist())
    layer_attrs_end.append(summarize_attributions(layer_attributions_end).cpu().detach().tolist())

    # storing attributions of the token id that we would like to examine in more detail in token_to_explain
    layer_attrs_start_dist.append(layer_attributions_start[0,token_to_explain,:].cpu().detach().tolist())
    layer_attrs_end_dist.append(layer_attributions_end[0,token_to_explain,:].cpu().detach().tolist())


# The plot below represents a heat map of attributions across all layers and tokens for the start position prediction. 
# It is interesting to observe that the question word `what` gains increasingly high attribution from layer one to ten. In the last two layers that importance is slowly diminishing.  
# In contrary to `what` token `it` has negative and zero attribution in the first 10 layers. However, surprisingly, last two layers assign higher attribution score to it.
# 
# And lastly, our correctly predicted token `to` for the start position gains increasingly positive attribution starting from layer seven.
# 

# In[28]:


fig, ax = plt.subplots(figsize=(15,5))
xticklabels=all_tokens
yticklabels=list(range(1,13))
ax = sns.heatmap(np.array(layer_attrs_start), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
plt.xlabel('Tokens')
plt.ylabel('Layers')
plt.show()


# Now let's examine the heat map of the attributions for the end position prediction. In the case of end position prediction we again observe high attribution scores for the token `what` in the last 11 layers.
# The token `it` seems to be harming the prediction the most. It's even more noticeable for the end position prediction than for the start. The correctly predicted end token `kinds` has positive attribution across all layers and it is especially prominent in the last two layers.

# In[29]:


fig, ax = plt.subplots(figsize=(15,5))

xticklabels=all_tokens
yticklabels=list(range(1,13))
ax = sns.heatmap(np.array(layer_attrs_end), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2) #, annot=True
plt.xlabel('Tokens')
plt.ylabel('Layers')

plt.show()


# It is interesting to note that when we compare the heat maps of start and end position, overall the colors for start position prediction on the map have darker intensities. This implies that there are less tokens that attribute positively to the start position prediction and there are more tokens which are negative indicators or signals of start position prediction.

# Now let's dig deeper into specific tokens and look into the distribution of attributions per layer for the token `kinds` in the start and end positions. The box plot diagram below shows the presence of outliers especially in the first four layers and in layer 8. We also observe that for start position prediction interquartile range slowly decreases as we go deeper into the layers and finally it is dimishing.
# 
# 

# In[30]:


fig, ax = plt.subplots(figsize=(20,10))
ax = sns.boxplot(data=layer_attrs_start_dist)
plt.xlabel('Layers')
plt.ylabel('Attribution')
plt.show()


# Now let's plot same distribution but for the prediction of the end position. Here attribution has larger positive values across all layers and the interquartile range doesn't change much when moving deeper into the layers.

# In[31]:


fig, ax = plt.subplots(figsize=(20,10))
ax = sns.boxplot(data=layer_attrs_end_dist)
plt.xlabel('Layers')
plt.ylabel('Attribution')
plt.show()


# Now, let's remove interpretation hooks, since we finished interpretation at this point

# In[32]:


remove_interpretable_embedding_layer(model, interpretable_embedding)


# An addition to this we can also represent attributions as a probability density function and compute the entropy of it in order to estimate the uncertainties of attributions.
# 
# TODO add entropy diagram here
# 

# In the Part 2 of this tutorial we will to go deeper into attention layers, heads and compare the attributions with the attention weight matrices and study and discuss related statistics.
