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

import os
from torch.utils.data import DataLoader

import unittest

from tests.utils import *
from deepmet.base import *


class BaseTestCase(unittest.TestCase):
    temp_results_dir = None

    @classmethod
    def to_test_results(cls, *args):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), cls.temp_results_dir.name, *args)

    @classmethod
    def setUpClass(cls):

        # create temporary directory for testing
        cls.temp_results_dir = make_temp_results_dir()

    def test_base_add_dataset(self):

        # can't create instance of ABC
        self.assertRaises(TypeError, BaseADDataset)

        # check that loaders must be implemented
        class ADDDatasetWithoutLoaders(BaseADDataset):
            pass

        self.assertRaises(TypeError, ADDDatasetWithoutLoaders)

        class ADDDatasetWithLoaders(BaseADDataset):

            def loaders(self):
                pass

        add_dataset_with_loaders = ADDDatasetWithLoaders("./")
        self.assertEqual(str(add_dataset_with_loaders), "ADDDatasetWithLoaders")
        self.assertEqual(add_dataset_with_loaders.root, "./")

    def test_loadable_dataset(self):

        # inherits from BaseADDDataset
        loadable_dataset = LoadableDataset("./")
        data_loader = loadable_dataset.loaders(5, 1)

        # creates data loaders
        self.assertIsInstance(data_loader[0], DataLoader)
        self.assertIsInstance(data_loader[1], DataLoader)

        # no data given
        self.assertIsNone(data_loader[0].dataset)
        self.assertIsNone(data_loader[1].dataset)

    def test_base_net(self):

        # can create instance, inherits from nn.Module not ABC
        base_net = BaseNet(200, 1000)
        base_net.summary()
        self.assertRaises(NotImplementedError, base_net.forward)

    def test_base_trainer(self):

        # can't create instance of ABC
        self.assertRaises(TypeError, BaseTrainer)

        # check we can create class that inherits from BaseTrainer when train and test are implemented
        class TrainerWithoutTrainTest(BaseTrainer):
            pass

        self.assertRaises(TypeError, TrainerWithoutTrainTest)

        class TrainerWithTrainTest(BaseTrainer):

            def train(self):
                pass

            def test(self):
                pass

        trainer_with_train_test = TrainerWithTrainTest("opt", 0.1, 5, tuple(), 5, 0.1, "cpu", 1)


if __name__ == '__main__':
    unittest.main()
