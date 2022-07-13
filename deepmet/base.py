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

import logging
import numpy as np
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class BaseADDataset(ABC):
    """ Anomaly detection dataset base class. """

    def __init__(self, root: str):
        super().__init__()
        self.root = root

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None
        self.outlier_classes = None

        self.train_set = None
        self.test_set = None

    @abstractmethod
    def loaders(self, batch_size: int, num_workers: int = 0) -> (DataLoader, DataLoader):
        """
        Implements data loaders of type :class:`~torch.utils.data.DataLoader` for
        `self.train_set` and `self.test_set`.
        """

        pass

    def __repr__(self):
        return self.__class__.__name__


class BaseNet(nn.Module):
    """ Base class for a neural network. """

    def __init__(self, rep_dim: int, in_features: int):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)

        self.rep_dim = rep_dim
        self.in_features = in_features

    def forward(self, *input):
        """ Forward pass logic. """

        raise NotImplementedError

    def summary(self):
        """ Network summary. """

        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])

        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


class BaseTrainer(ABC):
    """ Base class for a one-group model trainer. """

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple,
                 batch_size: int, weight_decay: float, device: str, n_jobs_dataloader: int):
        super().__init__()

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

    @abstractmethod
    def train(self, dataset: BaseADDataset, net: BaseNet) -> BaseNet:
        """ Implements train method that trains the given network using the `train_set` of dataset. """

        pass

    @abstractmethod
    def test(self, dataset: BaseADDataset, net: BaseNet):
        """ Implements test method that evaluates the `test_set` of dataset on the given network. """

        pass


class LoadableDataset(BaseADDataset):
    """ Class for loading datasets into :class:`~torch.utils.data.DataLoader` objects. """

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size: int, num_workers: int = 0) -> (DataLoader, DataLoader):
        """
        Loads the dataset to :class:`~torch.utils.data.DataLoader` objects.

        :param batch_size: How many samples per batch should be loaded.

        :param num_workers: The number of processes to be used for loading.

        :return: A tuple of two :class:`~torch.utils.data.DataLoader` objects, the first
            representing the training dataset and the second representing the test data.
        """

        train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        test_loader = DataLoader(
            dataset=self.test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, test_loader
