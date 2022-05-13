#!/usr/bin/env python3

from __future__ import print_function

import os
import tempfile
import unittest
from typing import Dict, List

import torch

HAS_PYTEXT = True
try:
    from captum.attr._models.pytext import (
        BaselineGenerator,
        configure_model_integ_grads_embeddings,
    )
    from pytext.common.constants import DatasetFieldName
    from pytext.config.component import create_featurizer, create_model
    from pytext.config.doc_classification import ModelInputConfig, TargetConfig
    from pytext.config.field_config import FeatureConfig, WordFeatConfig
    from pytext.data import CommonMetadata
    from pytext.data.doc_classification_data_handler import DocClassificationDataHandler
    from pytext.data.featurizer import SimpleFeaturizer
    from pytext.fields import FieldMeta
    from pytext.models.decoders.mlp_decoder import MLPDecoder
    from pytext.models.doc_model import DocModel_Deprecated
    from pytext.models.embeddings.word_embedding import WordEmbedding
    from pytext.models.representations.bilstm_doc_attention import BiLSTMDocAttention
except ImportError:
    HAS_PYTEXT = False


class VocabStub:
    def __init__(self) -> None:
        self.itos: List = []
        self.stoi: Dict = {}


# TODO add more test cases for dict features


class TestWordEmbeddings(unittest.TestCase):
    def setUp(self):
        if not HAS_PYTEXT:
            return self.skipTest("Skip the test since PyText is not installed")

        self.embedding_file, self.embedding_path = tempfile.mkstemp()
        self.word_embedding_file, self.word_embedding_path = tempfile.mkstemp()
        self.decoder_file, self.decoder_path = tempfile.mkstemp()
        self.representation_file, self.representation_path = tempfile.mkstemp()
        self.model = self._create_dummy_model()
        self.data_handler = self._create_dummy_data_handler()

    def tearDown(self):
        for f in (
            self.embedding_file,
            self.word_embedding_file,
            self.decoder_file,
            self.representation_file,
        ):
            os.close(f)
        for p in (
            self.embedding_path,
            self.word_embedding_path,
            self.decoder_path,
            self.representation_path,
        ):
            os.remove(p)

    def test_word_embeddings(self):
        embedding_list = configure_model_integ_grads_embeddings(self.model)
        integrated_gradients_embedding = embedding_list[0]
        input = torch.arange(0, 300).unsqueeze(0).unsqueeze(0)
        self.assertEqual(integrated_gradients_embedding.embedding_dim, 300)
        self.assertEqual(embedding_list.embedding_dim[0], 300)
        self.assertEqual(embedding_list(input).shape[2], input.shape[2])
        self.assertTrue(
            torch.allclose(
                integrated_gradients_embedding.get_attribution_map(input)["word"], input
            )
        )

    def test_baseline_generation(self):
        baseline_generator = BaselineGenerator(self.model, self.data_handler, "cpu")
        embedding_list = configure_model_integ_grads_embeddings(self.model)
        integrated_gradients_embedding = embedding_list[0]
        self.assertTrue(
            torch.allclose(
                baseline_generator.generate_baseline(integrated_gradients_embedding, 5)[
                    0
                ],
                torch.tensor([[1, 1, 1, 1, 1]]),
            )
        )

    def _create_dummy_data_handler(self):
        feat = WordFeatConfig(
            vocab_size=4,
            vocab_from_all_data=True,
            vocab_from_train_data=True,
            vocab_from_pretrained_embeddings=False,
            pretrained_embeddings_path=None,
        )
        featurizer = create_featurizer(
            SimpleFeaturizer.Config(), FeatureConfig(word_feat=feat)
        )
        data_handler = DocClassificationDataHandler.from_config(
            DocClassificationDataHandler.Config(),
            ModelInputConfig(word_feat=feat),
            TargetConfig(),
            featurizer=featurizer,
        )
        train_data = data_handler.gen_dataset(
            [{"text": "<pad>"}], include_label_fields=False
        )
        eval_data = data_handler.gen_dataset(
            [{"text": "<pad>"}], include_label_fields=False
        )
        test_data = data_handler.gen_dataset(
            [{"text": "<pad>"}], include_label_fields=False
        )
        data_handler.init_feature_metadata(train_data, eval_data, test_data)

        return data_handler

    def _create_dummy_model(self):
        return create_model(
            DocModel_Deprecated.Config(
                representation=BiLSTMDocAttention.Config(
                    save_path=self.representation_path
                ),
                decoder=MLPDecoder.Config(save_path=self.decoder_path),
            ),
            FeatureConfig(
                word_feat=WordEmbedding.Config(
                    embed_dim=300, save_path=self.word_embedding_path
                ),
                save_path=self.embedding_path,
            ),
            self._create_dummy_meta_data(),
        )

    def _create_dummy_meta_data(self):
        text_field_meta = FieldMeta()
        text_field_meta.vocab = VocabStub()
        text_field_meta.vocab_size = 4
        text_field_meta.unk_token_idx = 1
        text_field_meta.pad_token_idx = 0
        text_field_meta.pretrained_embeds_weight = None
        label_meta = FieldMeta()
        label_meta.vocab = VocabStub()
        label_meta.vocab_size = 3
        metadata = CommonMetadata()
        metadata.features = {DatasetFieldName.TEXT_FIELD: text_field_meta}
        metadata.target = label_meta
        return metadata
