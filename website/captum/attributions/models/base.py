#!/usr/bin/env python3

import torch

from functools import reduce

from torch.nn import Embedding


class InterpretableEmbeddingBase(Embedding):
    r"""
        Since some embedding vectors, e.g. word are created and assigned in
        the embedding layers of Pytorch models we need a way to access
        those layers, generate the embeddings and subtract the baseline.
        To do so, we separate embedding layers from the model, compute the embeddings
        separately and do all operations needed outside of the model.
        The original embedding layer is being replaced by
        `InterpretableEmbeddingBase` layer which passes already
        precomputed embedding vectors to the layers below.
    """

    def __init__(self, embedding):
        super().__init__(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding = embedding

    def forward(self, input):
        r"""
         The forward pass of embedding layer. This can be for the text or any
         type of embedding.

         Args

            input: Input embeddings tensor

         Return

            output: Output tensor is the same as input. It passes through
                    the embedding tensors to lower layers without any
                    modifications
        """
        return input

    def indices_to_embeddings(self, input):
        r"""
        Maps indices to corresponding embedding vectors

        Args

            input: a tensor of input indices. A typical example of an input
                   index is word index.

        Returns

            tensor: A tensor of word embeddings corresponding to the indices
                    specified in the input
        """
        return self.embedding(input)


class TokenReferenceBase:
    r"""
    A base class for creating reference tensor for a sequence of tokens. A typical
    example of such token is `PAD`. Users need to provide the index of the
    reference token in the vocabulary as an argument to `TokenReferenceBase`
    class.
    """

    def __init__(self, reference_token_idx=0):
        self.reference_token_idx = reference_token_idx

    def generate_reference(self, sequence_length, device):
        r"""
        Generated reference tensor of given `sequence_length` using
        `reference_token_idx`

        Returns

            tensor: a sequence of reference token with dimension [sequence_length]
        """
        return torch.tensor([self.reference_token_idx] * sequence_length, device=device)


def _get_deep_layer_name(obj, layer_names):
    r"""
    Traverses through the layer names that are separated by
    dot in order to access the embedding layer.
    """
    return reduce(getattr, layer_names.split("."), obj)


def _set_deep_layer_value(obj, layer_names, value):
    r"""
    Traverses through the layer names that are separated by
    dot in order to access the embedding layer and update its value.
    """
    layer_names = layer_names.split(".")
    setattr(reduce(getattr, layer_names[:-1], obj), layer_names[-1], value)


def configure_interpretable_embedding_layer(model, embedding_layer_name="embedding"):
    r"""
    This method wraps model's embedding layer with an interpretable embedding
    layer that allows us to access the embeddings through their indices.

    Args

        model: An instance of PyTorch model that contains embeddings
        embedding_layer_name: The name of the embedding layer in the `model`
                              that we would like to make interpretable

    Returns

        interpretable_emb: An instance of `InterpretableEmbeddingBase`
                           embedding layer that wraps model's
                           embedding layer - `embedding_layer_name`
    """
    embedding_layer = _get_deep_layer_name(model, embedding_layer_name)
    interpretable_emb = InterpretableEmbeddingBase(embedding_layer)
    _set_deep_layer_value(model, embedding_layer_name, interpretable_emb)
    return interpretable_emb
