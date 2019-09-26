#!/usr/bin/env python
# coding: utf-8

# # Predict and Interpret samples

# This notebook loads pretrained CNN model for sentiment analysis on IMDB dataset. It makes predictions on test samples and interprets those predictions using integrated gradients method.
# 
# The model was trained using an open source sentiment analysis tutorials described in: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb

# In[1]:


import sys
import spacy
import time

import torch
import torchtext
import torchtext.data
import torch.nn as nn
import torch.nn.functional as F

from torchtext.vocab import Vocab

from captum.attr import IntegratedGradients
from captum.attr import InterpretableEmbeddingBase, TokenReferenceBase
from captum.attr import visualization
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

nlp = spacy.load('en')


# In[2]:


device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


# The dataset used for training this model can be found in: https://ai.stanford.edu/~amaas/data/sentiment/
# 
# Redefining the model in order to be able to load it
# 

# In[3]:


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        #text = text.permute(1, 0)
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
                
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)


# Load pretrained model

# In[21]:


model = torch.load('models/imdb-model-cnn.pt')
model


# In[22]:


model.eval()
model


# Load test data

# In[23]:


TEXT = torchtext.data.Field(lower=True, tokenize='spacy')
Label = torchtext.data.LabelField(dtype = torch.float)


# In[24]:


train, test = torchtext.datasets.IMDB.splits(text_field=TEXT,
                                      label_field=Label,
                                      train='train',
                                      test='test',
                                      path='data/aclImdb')
test, _ = test.split(split_ratio = 0.04)


# Setting up vocabulary for word embeddings

# In[25]:


from torchtext import vocab

loaded_vectors = vocab.GloVe(name='6B', dim=50)
TEXT.vocab.set_vectors(stoi=loaded_vectors.stoi, vectors=loaded_vectors.vectors, dim=loaded_vectors.dim)
Label.build_vocab(train)


# In[26]:


print('Vectors Length: ', len(TEXT.vocab))


# In[27]:


PAD_IND = TEXT.vocab.stoi['pad']
PAD_IND


# The instance of `TokenReferenceBase` class will allow us to create sample based reference for text

# In[28]:


token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)


# Forward method that supports sigmoid

# In[29]:


def forward_with_sigmoid(input):
    return torch.sigmoid(model(input))


# In order to explain text features, we introduce interpretable embedding layers which allows access word embeddings and generate meaningful attributions for each embedding dimension.
# 
# `configure_interpretable_embedding_layer` function separates embedding layer from the model and precomputes word embeddings in advance. The embedding layer of our model is then being replaced by an Interpretable Embedding Layer which wraps original embedding layer and takes word embedding vectors as inputs of the forward function. This allows to generate baselines for word embeddings and compute attributions for each embedding dimension.
# 
# Note: After finishing interpretation it is important to call `remove_interpretable_embedding_layer` which removes the Interpretable Embedding Layer that we added for interpretation purposes and sets the original embedding layer back in the model.

# In[30]:


interpretable_embedding = configure_interpretable_embedding_layer(model, 'embedding')


# In[31]:


ig = IntegratedGradients(forward_with_sigmoid)


# In[32]:


# accumalate couple samples in this array for visualization purposes
vis_data_records_ig = []

def interpret_sentence(model, sentence, min_len = 7, label = 0):
    model.eval()
    text = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(text) < min_len:
        text += ['pad'] * (min_len - len(text))
    indexed = [TEXT.vocab.stoi[t] for t in text]

    
    model.zero_grad()

    input_indices = torch.LongTensor(indexed)
    input_indices = input_indices.unsqueeze(0)
    
    # input_indices dim: [sequence_length]
    seq_length = min_len

    # pre-computing word embeddings
    input_embedding = interpretable_embedding.indices_to_embeddings(input_indices)

    # predict
    pred = forward_with_sigmoid(input_embedding).item()
    pred_ind = round(pred)

    # generate reference for each sample
    reference_indices = token_reference.generate_reference(seq_length, device=device).unsqueeze(0)
    reference_embedding = interpretable_embedding.indices_to_embeddings(reference_indices)

    # compute attributions and approximation delta using integrated gradients
    attributions_ig, delta = ig.attribute(input_embedding, reference_embedding, n_steps=500)

    print('pred: ', Label.vocab.itos[pred_ind], '(', '%.2f'%pred, ')', ', delta: ', delta)

    add_attributions_to_visualizer(attributions_ig, text, pred, pred_ind, label, delta, vis_data_records_ig)
    
def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().numpy()

    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            Label.vocab.itos[pred_ind],
                            Label.vocab.itos[label],
                            Label.vocab.itos[1],
                            attributions.sum(),       
                            text[:len(attributions)],
                            delta))


# Interpret simple sentences and phrases

# In[33]:


interpret_sentence(model, 'It was a fantastic performance !', label=1)
interpret_sentence(model, 'Best film ever', label=1)
interpret_sentence(model, 'Such a great show!', label=1)
interpret_sentence(model, 'It was a horrible movie', label=0)
interpret_sentence(model, 'I\'ve never watched something as bad', label=0)
interpret_sentence(model, 'It is a disgusting movie!', label=0)


# In[34]:


# This is an example how we can visualize attributions
# Feel free to implement your own visualizers

print('Visualize attributions based on Integrated Gradients')
visualization.visualize_text(vis_data_records_ig)


# Above cell generates an output similar to this:

# In[35]:


from IPython.display import Image
Image(filename='img/sentiment_analysis.png')


# As mentioned above, after we are done with interpretation, we have to remove Interpretable Embedding Layer and set the original embeddings layer back to the model.

# In[36]:


remove_interpretable_embedding_layer(model, interpretable_embedding)

