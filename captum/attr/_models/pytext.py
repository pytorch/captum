#!/usr/bin/env python3
from collections import defaultdict

import torch
from pytext.models.embeddings.dict_embedding import DictEmbedding
from pytext.models.embeddings.word_embedding import WordEmbedding
from pytext.models.model import EmbeddingBase, EmbeddingList


class PyTextInterpretableEmbedding(EmbeddingBase):
    r"""
    In PyText DocNN models we need a way to access word embedding layers,
    generate the embeddings and subtract the baseline.
    To do so, we separate embedding layers from the model, compute the embeddings
    separately and do all operations needed outside of the model.
    The original embedding layer is being replaced by `PyTextInterpretableEmbedding`
    layer which passes precomputed embedding vectors to lower layers.
    """

    def __init__(self, embeddings) -> None:
        self.embedding_dims = [embedding.embedding_dim for embedding in embeddings]
        super().__init__(sum(self.embedding_dims))
        self.embeddings = embeddings

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

    def get_attribution_map(self, attributions):
        r"""
        After attribution scores are computed for an input embedding vector
        we need to split it up into attribution sub tensors for each
        feature type: word, dict and other types

        TODO: we can potentally also output tuples of attributions. This might be
        a better option. We'll work on this in a separate diff.

        Args

           attributions: A tensor that contains attribution values for each input
                         field. It usually has the same dimensions as the input
                         tensor

        Return

           attribution_map: A dictionary of feature_type and attribution values

        """
        begin = 0
        attribution_map = defaultdict()
        for embedding, embedding_size in zip(self.embeddings, self.embedding_dims):
            end = begin + embedding_size
            if isinstance(embedding, WordEmbedding):
                attribution_map["word"] = attributions[:, :, begin:end]
            elif isinstance(embedding, DictEmbedding):
                attribution_map["dict"] = attributions[:, :, begin:end]
            else:
                raise NotImplementedError(
                    "Currently only word and dict " "embeddings are supported"
                )
            begin = end

        return attribution_map


class BaselineGenerator:
    r"""
    This is an example input baseline generator for DocNN model which uses
    word and dict features.
    """
    PAD = "<pad>"

    def __init__(self, model, data_handler, device) -> None:
        self.model = model
        self.data_handler = data_handler
        if "dict_feat" in data_handler.features:
            self.vocab_dict = data_handler.features["dict_feat"].vocab
        if "word_feat" in data_handler.features:
            self.vocab_word = data_handler.features["word_feat"].vocab

        self.baseline_single_word_feature = self._generate_baseline_single_word_feature(
            device
        )
        self.baseline_single_dict_feature = self._generate_baseline_single_dict_feature(
            device
        )

    def generate_baseline(self, integ_grads_embeddings, seq_length):
        r"""
        Generates baseline for input word and dict features. In the future we
        will extend it to support char and other features as well.
        This baseline is entirely based on the `<pad>` token.

        Args

            integ_grads_embeddings: A reference to integrated gradients embedding
                                    layer
            seq_length: The length of each sequence which depends on batch size

        Return
                baseline: A tuple of feature baselines
                          Each feature type has a corresponding baseline tensor
                          in the tuple.
                          Currently only Dict and Word feature types are supported
        """
        baseline = []
        for embedding in integ_grads_embeddings.embeddings:
            if isinstance(embedding, WordEmbedding):
                baseline.append(self._generate_word_baseline(seq_length))
            elif isinstance(embedding, DictEmbedding):
                baseline.append(self._generate_dict_baseline(seq_length))
            else:
                raise NotImplementedError(
                    "Currently only word and dict " "embeddings are supported"
                )
        return tuple(baseline)

    def _generate_baseline_single_word_feature(self, device):
        return (
            torch.tensor(
                [self.vocab_word.stoi[self.PAD] if hasattr(self, "vocab_word") else 0]
            )
            .unsqueeze(0)
            .to(device)
        )

    def _generate_baseline_single_dict_feature(self, device):
        r"""Generate dict features based on Assistant's case study by using
         sia_transformer:
         fbcode/assistant/sia/transformer/sia_transformer.py
         sia_transformer generates dict features in a special gazetter format
         See `fbsource/fbcode/pytext/models/embeddings/dict_embedding.py`

         It generates word dict feature embeddings for each word token.

         The output of SIATransformer after running it on `<pad>` token
         looks as following:
        OutputRecord(tokens=['<', 'pad', '>'],
                     token_ranges=[(0, 1), (1, 4), (4, 5)],
                     gazetteer_feats=['<pad>', '<pad>', '<pad>'],
                     gazetteer_feat_lengths=[1, 1, 1],
                     gazetteer_feat_weights=[0.0, 0.0, 0.0],
                     characters=[['<', '<pad>', '<pad>'],
                                ['p', 'a', 'd'], ['>', '<pad>', '<pad>']],
                     pretrained_token_embedding=[ ], dense_feats=None)
        """
        gazetteer_feats = [self.PAD, self.PAD, self.PAD]
        gazetteer_feat_lengths = [1, 1, 1]
        gazetteer_feat_weights = [0.0, 0.0, 0.0]
        gazetteer_feat_id = (
            torch.tensor(
                [
                    self.vocab_dict.stoi[gazetteer_feat]
                    if hasattr(self, "vocab_dict")
                    else 0
                    for gazetteer_feat in gazetteer_feats
                ]
            )
            .unsqueeze(0)
            .to(device)
        )
        gazetteer_feat_weights = (
            torch.tensor(gazetteer_feat_weights).unsqueeze(0).to(device)
        )
        gazetteer_feat_lengths = (
            torch.tensor(gazetteer_feat_lengths).to(device).view(1, -1)[:, 1]
        )

        return (gazetteer_feat_id, gazetteer_feat_weights, gazetteer_feat_lengths)

    def _generate_word_baseline(self, seq_length):
        return self.baseline_single_word_feature.repeat(1, seq_length)

    def _generate_dict_baseline(self, seq_length):
        return (
            self.baseline_single_dict_feature[0].repeat(1, seq_length),
            self.baseline_single_dict_feature[1].repeat(1, seq_length),
            self.baseline_single_dict_feature[2].repeat(1, seq_length),
        )


def configure_task_integ_grads_embeddings(task):
    r"""
    Wraps Pytext's DocNN model embedding with `IntegratedGradientsEmbedding` for
    a given input task.
    IntegratedGradientsEmbedding allows to perform baseline related operations

    Args

        task: DocNN task reference

    Returns

        integrated_gradients_embedding_lst: The embedding layer which contains
                    IntegratedGradientsEmbedding as a wrapper over the original
                    embeddings of the model

    """
    integrated_gradients_embedding_lst = configure_model_integ_grads_embeddings(
        task.model
    )
    task.model.embedding = integrated_gradients_embedding_lst
    return integrated_gradients_embedding_lst[0]


def configure_model_integ_grads_embeddings(model):
    r"""
    Wraps Pytext's DocNN model embedding with `IntegratedGradientsEmbedding`
    IntegratedGradientsEmbedding allows to perform baseline related operations

    Args

        model: a reference to DocModel

    Returns

        integrated_gradients_embedding_lst: The embedding layer which contains
                    IntegratedGradientsEmbedding as a wrapper over the original
                    embeddings of the model

    """
    embeddings = model.embedding
    integrated_gradients_embedding = PyTextInterpretableEmbedding(embeddings)
    return EmbeddingList([integrated_gradients_embedding], False)


def reshape_word_features(word_features):
    r"""
     Creates one-sample batch for word features for sanity check purposes

    Args

        word_features: A tensor of diemnsions #words x #embeddings

    Return

        word_features: A tensor of dimensions 1 x #words x #embeddings

    """
    return word_features.unsqueeze(0)


def reshape_dict_features(
    dict_feature_id_batch, dict_weight_batch, dict_seq_len_batch, seq_length, idx
):
    r"""
    Creates one-sample batch for dict features for sanity check purposes
    It reads and reshapes id, weight and seq_length feature arrays for given
    input index `idx` from the input batch

    Args

        dict_feature_id_batch: The batch tensor for ids
        dict_weight_matrix: The batch tensor for weights
        dict_seq_len_matrix: The batch tensor for sequence length
        seq_length: The number of tokens per sequence
        idx: The index of sample in the batch

    Return

        dict_feature_ids: A tensor of dimensions [ bsz x # dict feature embeddings]
        dict_feature_weights: [ bsz x # dict feature embeddings]
        dict_feature_lens: [ bsz * seq_length ]

    """
    dict_feature_ids = dict_feature_id_batch[idx].unsqueeze(0)
    dict_feature_weights = dict_weight_batch[idx].unsqueeze(0)
    dict_feature_lens = dict_seq_len_batch[idx].unsqueeze(0)
    return (dict_feature_ids, dict_feature_weights, dict_feature_lens)
