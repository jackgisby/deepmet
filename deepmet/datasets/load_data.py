#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2021 Ralf Weber
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

from deepmet.datasets.mol_key_dataset import LoadableMolKeyDataset


def unison_shuffled_copies(a, b):

    assert len(a) == len(b)

    p = np.random.permutation(len(a))

    return a[p], b[p]


def get_data_from_csv(dataset_path, meta_path, shuffle=True):

    if not os.path.exists(dataset_path):
        raise FileNotFoundError

    if not os.path.exists(meta_path):
        raise FileNotFoundError

    self_x_data = np.loadtxt(dataset_path, delimiter=",", comments=None)
    self_labels = np.loadtxt(meta_path, delimiter=",", dtype=str, comments=None)

    if shuffle:
        return unison_shuffled_copies(self_x_data, self_labels)
    else:
        return self_x_data, self_labels


def load_training_dataset(normal_dataset_path, normal_meta_path, non_normal_dataset_path=None,
                          non_normal_dataset_meta_path=None, seed=1, validation_split=0.8, test_split=0.9):
    """ Loads the dataset. """
    
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)

    x_data, labels = get_data_from_csv(normal_dataset_path, normal_meta_path)

    num_rows, num_cols = x_data.shape
    train_val_split_index = floor(num_rows * validation_split)
    train_index = range(0, train_val_split_index)

    if test_split is None:
        val_test_split_index = num_rows
    else:
        val_test_split_index = floor(num_rows * test_split)

    val_index = range(train_val_split_index, val_test_split_index)

    if non_normal_dataset_path is not None:

        other_x_data, other_labels = get_data_from_csv(non_normal_dataset_path, non_normal_dataset_meta_path, shuffle=False)

        x_data = np.concatenate([x_data, other_x_data])
        labels = np.concatenate([labels, other_labels])

    num_rows, num_cols = x_data.shape
    test_index = range(val_test_split_index, num_rows)

    assert len(train_index) + len(val_index) + len(test_index) == num_rows
    for i, a in enumerate((train_index, val_index, test_index)):
        for j, b in enumerate((train_index, val_index, test_index)):
            if i != j:
                assert all([a_item not in b for a_item in a])

    full_dataset = LoadableMolKeyDataset(root=normal_dataset_path, train_idx=train_index, test_idx=test_index, data=x_data, labels=labels[:, 2])
    val_dataset = LoadableMolKeyDataset(root=normal_dataset_path, train_idx=train_index, test_idx=val_index, data=x_data, labels=labels[:, 2])

    return full_dataset, labels, val_dataset


def load_testing_dataset(normal_dataset_path, normal_meta_path):

    x_data, labels = get_data_from_csv(normal_dataset_path, normal_meta_path, shuffle=False)
    num_rows, num_cols = x_data.shape

    full_dataset = LoadableMolKeyDataset(root=normal_dataset_path, data=x_data, labels=np.zeros((num_rows, 1)))

    return full_dataset, labels
