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
import json
import time
import torch
import logging
import numpy as np
import torch.optim as optim
from typing import BinaryIO, Union
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

from deepmet.networks import build_network
from deepmet.base import BaseADDataset, BaseNet, BaseTrainer


class DeepMet(object):
    """
    Class for the DeepSVDD method adapted for compound anomaly detection.

    :param objective: One of "one-class" and "soft-boundary".

    :param nu: The proportion of samples in the training set to be classified as outliers.

    :param in_features: The number of input features.
    """

    def __init__(self, objective: str = 'one-class', nu: float = 0.1, rep_dim: int = 100, in_features: int = 2048):
        """ Constructor method. """

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu

        self.R = 0.0  # Hypersphere radius R
        self.c = None  # Hypersphere center c

        self.rep_dim = rep_dim
        self.in_features = in_features

        self.net_name = None
        self.net = None  # Neural network \phi

        self.trainer = None
        self.optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
            'test_loss': None
        }

        self.visualisation = None

    def set_network(self, net_name):
        """
        Builds the neural network \\phi.

        :param net_name: The name of the network architecture - see :py:meth:`deepmet.networks.build_network`.
        """

        self.net_name = net_name
        self.net = build_network(net_name, self.rep_dim, self.in_features)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """
        Trains the DeepMet model on the training data.

        :param dataset: Pytorch dataset class. May be loaded with :py:meth:`deepmet.datasets.load_training_dataset`.

        :param optimizer_name: optimisation method for training the network. Set to "amsgrad" to use the AMSGrad variant 
            of the adam optimisation algorithm. 
            
        :param lr: Learning rate of the optimisation process.

        :param n_epochs: Number of epochs to be used in the training process.

        :param lr_milestones: If specified, a multi-step learning rate decay can be used during training at the
            specified epochs. The tuple is passed to the `milestones` parameter of
            :py:meth:`torch.optim.lr_scheduler.MultiStepLR`.

        :param batch_size: The number of training samples to be used in each batch.

        :param weight_decay: The L2 penalty to be applied to weights during the training phase.

        :param device: The device to be used to train the model. One of "cuda" or "cpu".

        :param n_jobs_dataloader: The number of cpus to be utilised when loading the training and test sets.
        """

        self.optimizer_name = optimizer_name
        self.trainer = DeepMetTrainer(self.objective, self.R, self.c, self.nu, optimizer_name, lr=lr,
                                      n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                      weight_decay=weight_decay, device=device, n_jobs_dataloader=n_jobs_dataloader)

        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.R = float(self.trainer.R.cpu().data.numpy())  # Get float
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # Get list

        # Save results
        self.results['train_time'] = self.trainer.train_time
        self.results['R'] = self.R
        self.results['c'] = self.c

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """
        Tests the DeepMet model on the test data. Calls :py:meth:`deepmet.core.DeepMetTrainer` to carry out the
        training process.

        :param device: The device to be used to train the model. One of "cuda" or "cpu".

        :param n_jobs_dataloader: The number of cpus to be utilised when loading the training and test sets.

        :param dataset: Pytorch dataset class. May be loaded with :py:meth:`deepmet.datasets.load_testing_dataset`.
        """

        if self.trainer is None:
            self.trainer = DeepMetTrainer(self.objective, self.R, self.c, self.nu,
                                          device=device, n_jobs_dataloader=n_jobs_dataloader)

        # Test the model
        self.trainer.test(dataset, self.net)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores
        self.results['test_loss'] = self.trainer.test_loss

    def save_model(self, export_model: Union[str, os.PathLike, BinaryIO]):
        """
        Save DeepMet model to export_model. Saves the model's radius, centre and network state dictionary.

        :param export_model: File path for the model to be exported to.
        """

        net_dict = self.net.state_dict()

        torch.save({'R': self.R,
                    'c': self.c,
                    'net_dict': net_dict},
                   export_model)

    def load_model(self, model_path):
        """
        Load DeepMet model from model_path. Gets the model's radius, centre and network state dictionary.

        :param model_path: File path for the model to be loaded from.
        """

        model_dict = torch.load(model_path)

        self.R = model_dict['R']
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

    def save_results(self, export_json):
        """ Save results dict to a JSON-file. """

        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)


class DeepMetTrainer(BaseTrainer):
    """
    Trainer for the :py:meth:`deepmet.core.DeepMet` model.

    :param objective: One of "one-class" and "soft-boundary".

    :param R: The radius of the hypersphere.

    :param c: The centre of the hypersphere.

    :param nu: The proportion of samples in the training set to be classified as outliers.

    :param optimizer_name: optimisation method for training the network. Set to "amsgrad" to use the AMSGrad variant
        of the adam optimisation algorithm.

    :param lr: Learning rate of the optimisation process.

    :param n_epochs: Number of epochs to be used in the training process.

    :param lr_milestones: If specified, a multi-step learning rate decay can be used during training at the
        specified epochs. The tuple is passed to the `milestones` parameter of
        :py:meth:`torch.optim.lr_scheduler.MultiStepLR`.

    :param batch_size: The number of training samples to be used in each batch.

    :param weight_decay: The L2 penalty to be applied to weights during the training phase.

    :param device: The device to be used to train the model. One of "cuda" or "cpu".

    :param n_jobs_dataloader: The number of cpus to be utilised when loading the training and test sets.
    """

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device, n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # optimisation parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.test_loss = None

        # Visualisation
        self.latent_visualisation = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        """
        Method for training the :py:meth:`deepmet.core.DeepMet` model.

        :param dataset: Pytorch dataset class. May be loaded with :py:meth:`deepmet.datasets.load_training_dataset`.

        :param net: The NN to be trained.
        """

        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.2)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs.float())
                dist = torch.sum((outputs - self.c) ** 2, dim=1)

                # SVDD loss function
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)

                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        """
        Method for testing the :py:meth:`deepmet.core.DeepMet` model.

        :param dataset: Pytorch dataset class. May be loaded with :py:meth:`deepmet.datasets.load_testing_dataset`.

        :param net: The NN to be tested.
        """
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:

                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs.float())
                dist = torch.sum((outputs - self.c) ** 2, dim=1)

                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                loss = torch.mean(dist)

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1

        self.test_loss = loss_epoch / n_batches
        logger.info('Test set Loss: {:.8f}'.format(self.test_loss))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        try:
            self.test_auc = roc_auc_score(labels, scores)
            logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))
        except ValueError:
            print("Only one class present in y_true. ROC AUC score is not defined in that case.")

        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """
        Initialize hypersphere center c as the mean from an initial forward pass on the data.

        :param train_loader: Pytorch object for loading data.

        :param net: The NN to be initialised.

        :param eps: If the centre is too close to 0, it is set to +-eps. Else, zero units can be trivially matched with
            zero weights.
        """

        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs.float())
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """
    Optimally solve for radius R via the (1-nu)-quantile of distances.

    :param dist: A pytorch tensor.

    :param nu: The proportion of samples in the training set to be classified as outliers.

    :return: The value of the quantile.
    """

    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
