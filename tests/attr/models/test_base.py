#!/usr/bin/env python3

from __future__ import print_function

import torch
import unittest

from torch.nn import Embedding

from captum.attr._models.base import (
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
    InterpretableEmbeddingBase,
)

from ..helpers.basic_models import BasicEmbeddingModel


class Test(unittest.TestCase):
    def test_interpretable_embedding_base(self):
        input = torch.tensor([2, 5, 0, 1])
        model = BasicEmbeddingModel()
        output = model(input)
        interpretable_embedding1 = configure_interpretable_embedding_layer(
            model, "embedding1"
        )
        self.assertEqual(model.embedding1, interpretable_embedding1)
        self._assert_embeddings_equal(
            input,
            output,
            interpretable_embedding1,
            model.embedding1.embedding_dim,
            model.embedding1.num_embeddings,
        )
        interpretable_embedding2 = configure_interpretable_embedding_layer(
            model, "embedding2.inner_embedding"
        )
        self.assertEqual(model.embedding2.inner_embedding, interpretable_embedding2)
        self._assert_embeddings_equal(
            input,
            output,
            interpretable_embedding2,
            model.embedding2.inner_embedding.embedding_dim,
            model.embedding2.inner_embedding.num_embeddings,
        )
        # configure another embedding when one is already configured
        with self.assertRaises(AssertionError):
            configure_interpretable_embedding_layer(model, "embedding2.inner_embedding")
        with self.assertRaises(AssertionError):
            configure_interpretable_embedding_layer(model, "embedding1")
        # remove interpretable embedding base
        self.assertTrue(
            model.embedding2.inner_embedding.__class__ is InterpretableEmbeddingBase
        )
        remove_interpretable_embedding_layer(model, interpretable_embedding2)
        self.assertTrue(model.embedding2.inner_embedding.__class__ is Embedding)

        self.assertTrue(model.embedding1.__class__ is InterpretableEmbeddingBase)
        remove_interpretable_embedding_layer(model, interpretable_embedding1)
        self.assertTrue(model.embedding1.__class__ is Embedding)

    def _assert_embeddings_equal(
        self, input, output, interpretable_embedding, embedding_dim, num_embeddings
    ):
        self.assertEqual(embedding_dim, interpretable_embedding.embedding_dim)
        self.assertEqual(num_embeddings, interpretable_embedding.num_embeddings)

        # dim - [4, 100]
        emb_shape = interpretable_embedding.indices_to_embeddings(input).shape
        self.assertEqual(emb_shape[0], input.shape[0])
        self.assertEqual(emb_shape[1], interpretable_embedding.embedding_dim)
        self.assertEqual(input.shape, output.shape)
