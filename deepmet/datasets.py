#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2021 Jack Gisby, Ralf Weber
#
# This file is part of DeepMet.
#
# DeepMet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepMet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepMet.  If not, see <https://www.gnu.org/licenses/>.
#
# This file incorporates work covered by the following copyright and
# permission notice:
#
#   Copyright (c) 2018 Lukas Ruff
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import os
import random
import numpy as np
from math import floor
from typing import Union, Tuple
from torch.utils.data import Dataset, Subset

from deepmet.base import LoadableDataset


def unison_shuffled_copies(a: np.array, b: np.array) -> Tuple[np.array, np.array]:
    """
    Shuffle two datasets in unison, using :py:meth:`numpy.random.permutation`.

    :param a: numpy vector or matrix

    :param b: numpy vector or matrix

    :return: A tuple containing a and b, each after shuffling.
    """

    assert len(a) == len(b)

    p = np.random.permutation(len(a))

    return a[p], b[p]


def get_data_from_csv(dataset_path: str, meta_path: str, shuffle: bool = True,
                      add_labels: float = None) -> Tuple[np.array, np.array]:
    """
    Get fingerprints and metadata from CSV files.

    :param dataset_path: The path of the input fingerprints, as a CSV file.

    :param meta_path: The path of the input metadata, as a CSV file.

    :param shuffle: Whether to shuffle the input fingerprints and labels.

    :param add_labels: If None, the labels present in column three of the file at `meta_path` will be used as labels.
        Else, `add_labels` should be a number, where 0 represents a "normal" class and any other positive integer
        represents a "non-normal" class.

    :return: A tuple containing the loaded fingerprints and metadata as a numpy matrix and vector, respectively.
    """

    if not os.path.exists(dataset_path):
        raise FileNotFoundError

    if not os.path.exists(meta_path):
        raise FileNotFoundError

    self_x_data = np.loadtxt(dataset_path, delimiter=",", comments=None)
    self_labels = np.loadtxt(meta_path, delimiter=",", dtype=str, comments=None)

    num_rows, num_cols = self_labels.shape

    if add_labels is not None or num_cols < 3:

        if num_cols == 3:
            extra_column = np.ones(num_rows, dtype=np.int)
        else:
            extra_column = np.ones((num_rows, 1), dtype=np.int)

        if add_labels is not None:
            extra_column *= add_labels

        if num_cols == 3:
            self_labels[:, 2] = extra_column

        else:
            assert num_cols == 2
            self_labels = np.append(self_labels, extra_column, axis=1)

    if shuffle:
        return unison_shuffled_copies(self_x_data, self_labels)
    else:
        return self_x_data, self_labels


class LoadableMolKeyDataset(LoadableDataset):
    """
    A loadable pytorch dataset. Can contain a test and training set - the test set may be used as a test or validation
    set.

    :param root: The directory containing the datasets.

    :param train_idx: If provided, represents a list of indices indicating observations
        that belong to the training data split.

    :param test_idx: If provided, represents a list of indices indicating observations
        that belong to the testing data split.

    :param data: A 2D numpy array containing the dataset.

    :param labels: Group labels for each observation in `data`.
    """

    def __init__(self, root: str, train_idx: Union[range, None] = None, test_idx: Union[range, None] = None,
                 data: Union[np.array, None] = None, labels: Union[np.array, None] = None):
        super().__init__(root)

        self.train_set = MolKeyDataset(root=self.root, train=True, data=data, labels=labels)

        if train_idx is not None:
          self.train_set = Subset(self.train_set, train_idx)

        self.test_set = MolKeyDataset(root=self.root, train=False, data=data, labels=labels)

        if test_idx is not None:
            self.test_set = Subset(self.test_set, test_idx)


class MolKeyDataset(Dataset):
    """
    A pytorch dataset representing a training, validation or test set.

    :param root: The directory containing the dataset.

    :param train: Whether the dataset represents "training" data.

    :param data: A 2D numpy array containing the dataset.

    :param labels: Group labels for each observation in `data`.
    """

    def __init__(self, root: str, train: bool, data: Union[np.array, None] = None,
                 labels: Union[np.array, None] = None):
        super(MolKeyDataset, self).__init__()

        self.train = train

        self.data = data

        if labels is None:
            self.labels = np.zeros(self.data.shape[0])
        else:
            self.labels = labels.astype(float)

    # This is used to return a single datapoint. A requirement from pytorch
    def __getitem__(self, index):
        return self.data[index], self.labels[index], index

    # For Pytorch to know how many datapoints are in the dataset
    def __len__(self):
        return len(self.data)


def load_training_dataset(normal_dataset_path: str, normal_meta_path: str,
                          non_normal_dataset_path: Union[None, str] = None,
                          non_normal_dataset_meta_path: Union[None, str] = None, seed: int = 1,
                          validation_split: float = 0.8, test_split: float = 0.9
                          ) -> Tuple[LoadableMolKeyDataset, np.array, LoadableMolKeyDataset]:
    """
    Gets pytorch dataset classes for training and testing. A set of "normal" compounds must be supplied for use as a
    training dataset. Some of these structures must be reserved for the validation dataset and some may also be used
    in the test set. A comparator group of "non-normal" structures can also be provided - although these will be solely
    reserved for testing purposes and will not be involved in the training or validation of the model.

    :param normal_dataset_path: Path to the fingerprints of the "normal" structures, as a CSV file.

    :param normal_meta_path: Path to the metadata for the "normal" structures, as a CSV file.

    :param non_normal_dataset_path: Optional, path to the fingerprints of the "non-normal" structures, as a CSV file.
        The "non-normal" structures are solely reserved for the test set - they are not used in training or validation.

    :param non_normal_dataset_meta_path: Must be given if `non_normal_dataset_path` is given. Path to the metadata of
        the "non-normal" structures, as a CSV file.

    :param seed: Random seed to ensure dataset shuffling is consistent. If set to -1, the seed will not be set.

    :param validation_split: Proportion of the "normal" dataset to be used as training data.

    :param test_split: If None, 1 - `validation_split` will be the proportion of "normal" data used for the validation
        set and there will be no normal structures in the test set. Else, `test_split - validation_split" will be the
        proportion of data used for the validation set and `1.0 - test_split` will be the proportion of normal
        structures used for the test set. Any "non-normal" structures supplied will be reserved solely for use in
        the test set.

    :return: Tuple containing:
     - A :py:meth:`deepmet.datasets.LoadableMolKeyDataset` object containing the training set and the test set.
     - Metadata for the training, validation and test sets.
     - A :py:meth:`deepmet.datasets.LoadableMolKeyDataset` object containing the training set and the validation set.
    """
    
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)

    x_data, labels = get_data_from_csv(normal_dataset_path, normal_meta_path, add_labels=0)

    num_rows, num_cols = x_data.shape
    train_val_split_index = floor(num_rows * validation_split)
    train_index = range(0, train_val_split_index)

    if test_split is None:
        val_test_split_index = num_rows
    else:
        val_test_split_index = floor(num_rows * test_split)

    val_index = range(train_val_split_index, val_test_split_index)

    if non_normal_dataset_path is not None:

        other_x_data, other_labels = get_data_from_csv(non_normal_dataset_path, non_normal_dataset_meta_path, add_labels=1, shuffle=False)

        x_data = np.concatenate([x_data, other_x_data])
        labels = np.concatenate([labels, other_labels])

        assert num_cols == other_x_data.shape[1]

    num_rows, num_cols = x_data.shape
    test_index = range(val_test_split_index, num_rows)

    assert len(train_index) + len(val_index) + len(test_index) == num_rows
    for i, a in enumerate((train_index, val_index, test_index)):
        for j, b in enumerate((train_index, val_index, test_index)):
            if i != j:
                assert all([a_item not in b for a_item in a])

    full_dataset = LoadableMolKeyDataset(
        root=normal_dataset_path,
        train_idx=train_index,
        test_idx=test_index,
        data=x_data,
        labels=labels[:, 2]
    )

    val_dataset = LoadableMolKeyDataset(
        root=normal_dataset_path,
        train_idx=train_index,
        test_idx=val_index,
        data=x_data,
        labels=labels[:, 2]
    )

    return full_dataset, labels, val_dataset


def load_testing_dataset(input_dataset_path: str, input_meta_path: str) -> Tuple[LoadableMolKeyDataset, np.array]:
    """
    Load a test set in the absence of training data.
    
    :param input_dataset_path: Path to the fingerprints of the input structures, as a CSV file.

    :param input_meta_path: Path to the metadata for the input structures, as a CSV file.

    :return: A tuple containing:
     - A :py:meth:`deepmet.datasets.LoadableMolKeyDataset` object containing the test set.
     - Metadata for the training, validation and test sets.
    """

    x_data, labels = get_data_from_csv(input_dataset_path, input_meta_path, shuffle=False)
    num_rows, num_cols = x_data.shape

    full_dataset = LoadableMolKeyDataset(root=input_dataset_path, data=x_data, labels=np.zeros((num_rows, 1)))

    return full_dataset, labels
