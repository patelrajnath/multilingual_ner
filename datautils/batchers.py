"""This module contains batch iterators that are used in EncoderClassifiers

Copyright PolyAI Limited.
"""
from collections import abc
from typing import Dict, Optional, List

import flair
import numpy as np
import torch

_MAX_PER_BATCH = 3


class SamplingBatcher(abc.Iterator):
    """Batcher that samples according to a given distribution.

    It defaults to sampling from the data distribution.

    WARNING: this class is not deterministic. if you want deterministic
    behaviour, just freeze the numpy seed.
    """

    def __init__(
            self,
            examples_a: np.ndarray,
            examples_b: np.ndarray,
            batch_size: int,
            pad_id: int,
            pad_id_labels: int,
    ):
        """Create a new BalancedBatcher.

        Args:
            examples_a: np.ndarray containing examples
            examples_b: np.ndarray containing examples
            batch_size: int size of a single batch
        """
        self._examples_a = examples_a
        self._examples_b = examples_b
        self._num_items = examples_a.shape[0]
        self._pad_id = pad_id
        self.pad_id_labels = pad_id_labels
        self._batch_size = batch_size
        self._indices = np.arange(self._num_items)
        self.rnd = np.random.RandomState(0)
        self.ptr = 0

    def __next__(self):
        """Generates the next batch.

        Returns:
            (batch_of_examples, batch_of_labels) - a tuple of ndarrays
        """
        if self.ptr + self._batch_size > self._num_items:
            self.rnd.shuffle(self._indices)
            self.ptr = 0
            raise StopIteration  # ugly Python
        else:
            batch_indices = \
                self._indices[self.ptr:self.ptr + self._batch_size]
            self.ptr += self._batch_size

            batch_sentences, batch_tags = self._examples_a[batch_indices], self._examples_b[batch_indices]
            # compute length of longest sentence in batch
            batch_max_len = max([len(s) for s in batch_sentences])

            # prepare a numpy array with the data, initializing the data with 'PAD'
            # and all labels with -1; initializing labels to -1 differentiates tokens
            # with tags from 'PAD' tokens
            input_ = self._pad_id * np.ones((len(batch_sentences), batch_max_len))
            labels = self.pad_id_labels * np.ones((len(batch_sentences), batch_max_len))

            # copy the data to the numpy array
            data_len = []
            for j in range(len(batch_sentences)):
                cur_len = len(batch_sentences[j])
                data_len.append(cur_len)
                input_[j][:cur_len] = batch_sentences[j]
                labels[j][:cur_len] = batch_tags[j]

            # since all data are indices, we convert them to torch LongTensors
            input_ = torch.LongTensor(input_)
            labels = torch.LongTensor(labels)

            data_len = np.asarray(data_len)
            batch_len = torch.LongTensor(data_len)

            maxlen = input_.shape[1]
            input_mask = torch.arange(maxlen)[None, :] < batch_len[:, None]

            # For labels we use 0 for padding
            labels_mask = (labels > 0).long()

            return input_, labels, batch_len, input_mask, labels_mask

    def __iter__(self):
        """Gets an iterator for this iterable

        Returns:
            self because the class is an iterator itself
        """
        return self

    def __len__(self):
        return self._num_items


class SamplingBatcherFlair(abc.Iterator):
    """Batcher that samples according to a given distribution.

    It defaults to sampling from the data distribution.

    WARNING: this class is not deterministic. if you want deterministic
    behaviour, just freeze the numpy seed.
    """

    def __init__(
            self,
            examples_a: np.ndarray,
            examples_b: np.ndarray,
            batch_size: int,
            pad_id: int,
            pad_id_labels: int,
            embedding_length: int,
    ):
        """Create a new BalancedBatcher.

        Args:
            examples_a: np.ndarray containing examples
            examples_b: np.ndarray containing examples
            batch_size: int size of a single batch
        """
        self._examples_a = examples_a
        self._examples_b = examples_b
        self._num_items = examples_a.shape[0]
        self._pad_id = pad_id
        self.pad_id_labels = pad_id_labels
        self._batch_size = batch_size
        self._indices = np.arange(self._num_items)
        self.rnd = np.random.RandomState(0)
        self.ptr = 0
        self.embedding_length = embedding_length

    def __next__(self):
        """Generates the next batch.

        Returns:
            (batch_of_examples, batch_of_labels) - a tuple of ndarrays
        """
        if self.ptr + self._batch_size > self._num_items:
            self.rnd.shuffle(self._indices)
            self.ptr = 0
            raise StopIteration  # ugly Python
        else:
            batch_indices = \
                self._indices[self.ptr:self.ptr + self._batch_size]
            self.ptr += self._batch_size

            batch_sentences, batch_tags = self._examples_a[batch_indices], self._examples_b[batch_indices]
            # compute length of longest sentence in batch
            batch_max_len = max([len(s) for s in batch_sentences])

            # prepare a numpy array with the data, initializing the data with 'PAD'
            # and all labels with -1; initializing labels to -1 differentiates tokens
            # with tags from 'PAD' tokens
            sentences = batch_sentences
            lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
            longest_token_sequence_in_batch: int = max(lengths)

            pre_allocated_zero_tensor = torch.full(
                (self.embedding_length * longest_token_sequence_in_batch,),
                self._pad_id,
                dtype=torch.float,
                device=flair.device,
            )

            all_embs: List[torch.Tensor] = list()
            labels = self.pad_id_labels * np.ones((len(sentences), batch_max_len))

            for j, sentence in enumerate(sentences):
                cur_len = len(sentences[j])
                labels[j][:cur_len] = batch_tags[j]

                all_embs += [
                    emb for token in sentence for emb in token.get_each_embedding()
                ]
                nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

                if nb_padding_tokens > 0:
                    t = pre_allocated_zero_tensor[
                        : self.embedding_length * nb_padding_tokens
                        ]
                    all_embs.append(t)

            sentence_tensor = torch.cat(all_embs).view(
                [
                    len(sentences),
                    longest_token_sequence_in_batch,
                    self.embedding_length,
                ]
            )

            # since all data are indices, we convert them to torch LongTensors
            labels = torch.LongTensor(labels)

            # For labels we use 0 for padding
            labels_mask = (labels > 0).long()

            return sentence_tensor, labels, labels_mask

    def __iter__(self):
        """Gets an iterator for this iterable

        Returns:
            self because the class is an iterator itself
        """
        return self

    def __len__(self):
        return self._num_items


class SamplingBatcherStackedTransformers(abc.Iterator):
    """Batcher that samples according to a given distribution.

    It defaults to sampling from the data distribution.

    WARNING: this class is not deterministic. if you want deterministic
    behaviour, just freeze the numpy seed.
    """

    def __init__(
            self,
            examples_a: np.ndarray,
            examples_b: np.ndarray,
            batch_size: int,
            pad_id: int,
            pad_id_labels: int,
            embedding_length: int,
            device: str,
    ):
        """Create a new BalancedBatcher.

        Args:
            examples_a: np.ndarray containing examples
            examples_b: np.ndarray containing examples
            batch_size: int size of a single batch
        """
        self._examples_a = examples_a
        self._examples_b = examples_b
        self._num_items = examples_a.shape[0]
        self._pad_id = pad_id
        self.pad_id_labels = pad_id_labels
        self._batch_size = batch_size
        self._indices = np.arange(self._num_items)
        self.rnd = np.random.RandomState(0)
        self.ptr = 0
        self.embedding_length = embedding_length
        self.device = device

    def __next__(self):
        """Generates the next batch.

        Returns:
            (batch_of_examples, batch_of_labels) - a tuple of ndarrays
        """
        if self.ptr + self._batch_size > self._num_items:
            self.rnd.shuffle(self._indices)
            self.ptr = 0
            raise StopIteration  # ugly Python
        else:
            batch_indices = \
                self._indices[self.ptr:self.ptr + self._batch_size]
            self.ptr += self._batch_size

            batch_sentences, batch_tags = self._examples_a[batch_indices], self._examples_b[batch_indices]

            # prepare a numpy array with the data, initializing the data with 'PAD'
            # and all labels with -1; initializing labels to -1 differentiates tokens
            # with tags from 'PAD' tokens
            sentences = batch_sentences
            lengths: List[int] = [sentence.shape[0] for sentence in sentences]
            longest_token_sequence_in_batch: int = max(lengths)

            pre_allocated_zero_tensor = torch.full(
                (self.embedding_length * longest_token_sequence_in_batch,),
                self._pad_id,
                dtype=torch.float,
                device=self.device,
            )

            all_embs: List[torch.Tensor] = list()
            labels = self.pad_id_labels * np.ones((len(sentences), longest_token_sequence_in_batch))

            for j, sentence in enumerate(sentences):
                cur_len = sentence.shape[0]
                labels[j][:cur_len] = batch_tags[j]

                all_embs += [emb for emb in sentence]
                nb_padding_tokens = longest_token_sequence_in_batch - cur_len

                if nb_padding_tokens > 0:
                    t = pre_allocated_zero_tensor[
                        : self.embedding_length * nb_padding_tokens
                        ]
                    all_embs.append(t)

            sentence_tensor = torch.cat(all_embs).view(
                [
                    len(sentences),
                    longest_token_sequence_in_batch,
                    self.embedding_length,
                ]
            )

            # since all data are indices, we convert them to torch LongTensors
            labels = torch.LongTensor(labels)

            # For labels we use 0 for padding
            labels_mask = (labels > 0).long()

            return sentence_tensor, labels, labels_mask

    def __iter__(self):
        """Gets an iterator for this iterable

        Returns:
            self because the class is an iterator itself
        """
        return self

    def __len__(self):
        return self._num_items
