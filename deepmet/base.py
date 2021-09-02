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

import logging
import numpy as np
import torch.nn as nn
from typing import Callable, Union
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class BaseADDataset(ABC):
    """ Anomaly detection dataset base class. """

    def __init__(self, root: str):
        super().__init__()
        self.root = root  # Root path to data

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # Tuple with original class labels that define the normal class
        self.outlier_classes = None  # Tuple with original class labels that define the outlier class

        self.train_set = None  # Must be of type torch.auxiliary.data.Dataset
        self.test_set = None  # Must be of type torch.auxiliary.data.Dataset

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
                DataLoader, DataLoader):
        """ Implement data loaders of type torch.auxiliary.data.DataLoader for train_set and test_set. """

        pass

    def __repr__(self):
        return self.__class__.__name__


class BaseNet(nn.Module):
    """ Base class for all neural networks. """

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # Representation dimensionality, i.e. dim of the last layer

    def forward(self, *input):
        """
        Forward pass logic

        :return: Network output
        """

        raise NotImplementedError

    def summary(self):
        """ Network summary. """

        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])

        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


class BaseTrainer(ABC):
    """ Trainer base class. """

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple, batch_size: int,
                 weight_decay: float, device: str, n_jobs_dataloader: int, loss_function: Union[None, Callable]):
        super().__init__()

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        self.loss_function = loss_function

    @abstractmethod
    def train(self, dataset: BaseADDataset, net: BaseNet) -> BaseNet:
        """
        Implement train method that trains the given network using the train_set of dataset.

        :return: Trained net
        """

        pass

    @abstractmethod
    def test(self, dataset: BaseADDataset, net: BaseNet):
        """ Implement test method that evaluates the test_set of dataset on the given network. """

        pass


class LoadableDataset(BaseADDataset):
    """ Class for loading datasets into a usable format. """

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
                DataLoader, DataLoader):

        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)

        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)

        return train_loader, test_loader
