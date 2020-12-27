"""This module contains batch iterators that are used in EncoderClassifiers

Copyright PolyAI Limited.
"""
from collections import abc
from typing import Dict, Optional

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
            pad_id = int
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
            batch_data = self._pad_id * np.ones((len(batch_sentences), batch_max_len))
            batch_labels = -1 * np.ones((len(batch_sentences), batch_max_len))

            # copy the data to the numpy array
            data_len = []
            for j in range(len(batch_sentences)):
                cur_len = len(batch_sentences[j])
                data_len.append(cur_len)
                batch_data[j][:cur_len] = batch_sentences[j]
                batch_labels[j][:cur_len] = batch_tags[j]

            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.LongTensor(batch_data)
            batch_labels = torch.LongTensor(batch_labels)

            data_len = np.asarray(data_len)
            batch_len = torch.LongTensor(data_len)

            maxlen = batch_data.shape[1]
            mask_x = torch.arange(maxlen)[None, :] < batch_len[:, None]

            # reshape labels to give a flat vector of length batch_size*seq_len
            # batch_labels = batch_labels.view(-1)

            # For labels we use -1 for padding
            mask_y = (batch_labels >= 0).long()

            return batch_data, batch_labels, batch_len, mask_x, mask_y

    def __iter__(self):
        """Gets an iterator for this iterable

        Returns:
            self because the class is an iterator itself
        """
        return self

    def __len__(self):
        return self._num_items
