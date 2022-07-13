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

import unittest

from torch.utils.data import DataLoader

from tests.utils import *
from deepmet.datasets import *
from deepmet.auxiliary import get_fingerprints_from_meta


class DatasetTestCase(unittest.TestCase):
    temp_results_dir = None

    @classmethod
    def to_test_results(cls, *args):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), cls.temp_results_dir.name, *args)

    @classmethod
    def setUpClass(cls):

        # create temporary directory for testing
        cls.temp_results_dir = make_temp_results_dir()

        # get a subset of the full input data
        cls.normal_meta_path, cls.non_normal_meta_path = get_normal_non_normal_subsets(cls.to_test_results(), normal_sample_size=50)

        # get a test dataset
        cls.normal_fingerprints_path = get_fingerprints_from_meta(
            cls.normal_meta_path,
            cls.to_test_results("normal_fingerprints.csv")
        )

        cls.non_normal_fingerprints_path = get_fingerprints_from_meta(
            cls.non_normal_meta_path,
            cls.to_test_results("non_normal_fingerprints.csv")
        )

    def test_unison_shuffled_copies(self):

        normal_meta = np.loadtxt(self.normal_meta_path, delimiter=",", dtype=str, comments=None)
        normal_fingerprints = np.loadtxt(self.normal_fingerprints_path, delimiter=",", comments=None)

        # shuffle the datasets
        random.seed(1)
        np.random.seed(1)
        shuffled_meta, shuffled_fingerprints = unison_shuffled_copies(normal_meta, normal_fingerprints)

        # check the first 5 rows are as expected
        hmdb_ids_to_check = ['HMDB0097649', 'HMDB0124840', 'HMDB0128276', 'HMDB0127036', 'HMDB0007646']

        self.assertEqual([r[0] for i, r in enumerate(shuffled_meta) if i < 5], hmdb_ids_to_check)
        self.assertEqual(shuffled_fingerprints[1:5].sum(), 8970)

        # check the fingerprints have been shuffled in the same way the meta has been
        for hmdb_id in hmdb_ids_to_check:
            before_idx = normal_meta[:, 0] == hmdb_id
            after_idx = shuffled_meta[:, 0] == hmdb_id

            self.assertTrue((normal_fingerprints[before_idx] == shuffled_fingerprints[after_idx]).all())

    def test_get_data_from_csv(self):

        acquired_data = {}

        # default parameters
        random.seed(1)
        np.random.seed(1)
        acquired_data["standard"] = get_data_from_csv(self.normal_fingerprints_path, self.normal_meta_path)

        # without shuffling
        acquired_data["non_shuffled"] = get_data_from_csv(self.normal_fingerprints_path, self.normal_meta_path, shuffle=False)

        # add custom labels
        random.seed(1)
        np.random.seed(1)
        acquired_data["add_labels"] = get_data_from_csv(self.normal_fingerprints_path, self.normal_meta_path, add_labels=range(50))

        shuffled_hmdb_ids = ['HMDB0124840', 'HMDB0128276', 'HMDB0127036', 'HMDB0007646']
        non_shuffled_hmdb_ids = ['HMDB0003134', 'HMDB0007646', 'HMDB0007831', 'HMDB0007891']

        # perform checks
        for argument_type, (fingerprints, meta) in acquired_data.items():

            hmdb_ids = non_shuffled_hmdb_ids if argument_type is "non_shuffled" else shuffled_hmdb_ids
            self.assertTrue((meta[1:5, 0] == hmdb_ids).all())

            self.assertEqual(fingerprints[1:5].sum(), 5281 if argument_type is "non_shuffled" else 8970)

    def test_mol_key_dataset(self):

        # get some data
        random.seed(1)
        np.random.seed(1)
        fingerprints, meta = get_data_from_csv(self.normal_fingerprints_path, self.normal_meta_path)

        # inherits from LoadableDataset
        loadable_mol_key_dataset = LoadableMolKeyDataset("./", fingerprints,
                                                         test_idx=range(1, 5),
                                                         train_idx=range(10, 15))

        # test loaders with no labels
        train_loader, test_loader = loadable_mol_key_dataset.loaders(batch_size=10)
        self.assertIsInstance(train_loader, DataLoader)

        # data is the same
        self.assertTrue((train_loader.dataset.dataset.data == test_loader.dataset.dataset.data).all())

        # labels are set to 0
        self.assertTrue((train_loader.dataset.dataset.labels == 0).all())

        # check indices
        self.assertEqual(test_loader.dataset.indices, range(1, 5))
        self.assertEqual(train_loader.dataset.indices, range(10, 15))

        # now try with labels
        loadable_mol_key_dataset_labelled = LoadableMolKeyDataset("./", fingerprints,
                                                                  labels=np.ones(len(meta)),
                                                                  test_idx=range(1, 5),
                                                                  train_idx=range(10, 15))
        train_loader, test_loader = loadable_mol_key_dataset_labelled.loaders(batch_size=10)

        # labels are now ones
        self.assertTrue((train_loader.dataset.dataset.labels == 1).all())

    def test_load_training_dataset(self):

        # without testing data
        full_dataset, labels, val_dataset = load_training_dataset(
            self.normal_fingerprints_path,
            self.normal_meta_path
        )

        # check data splits
        self.assertEqual(full_dataset.train_set.indices, range(40))
        self.assertEqual(full_dataset.test_set.indices, range(45, 50))
        self.assertEqual(val_dataset.train_set.indices, range(40))
        self.assertEqual(val_dataset.test_set.indices, range(40, 45))

        # check labels
        self.assertTrue((full_dataset.train_set.dataset.labels == 0).all())

        # without test split
        full_dataset, labels, val_dataset = load_training_dataset(
            self.normal_fingerprints_path,
            self.normal_meta_path,
            self.non_normal_fingerprints_path,
            self.non_normal_meta_path,
            test_split=None
        )

        # check data splits
        self.assertEqual(full_dataset.train_set.indices, range(40))
        self.assertEqual(full_dataset.test_set.indices, range(50, 100))
        self.assertEqual(val_dataset.test_set.indices, range(40, 50))

        # check labels
        self.assertTrue((full_dataset.train_set.dataset.labels[range(50)] == 0).all())
        self.assertTrue((full_dataset.test_set.dataset.labels[range(50, 100)] == 1).all())

        # with both testing data and test split
        full_dataset, labels, val_dataset = load_training_dataset(
            self.normal_fingerprints_path,
            self.normal_meta_path,
            self.non_normal_fingerprints_path,
            self.non_normal_meta_path
        )

        # check data splits
        self.assertEqual(full_dataset.train_set.indices, range(40))
        self.assertEqual(full_dataset.test_set.indices, range(45, 100))
        self.assertEqual(val_dataset.test_set.indices, range(40, 45))

        # check labels
        self.assertTrue((full_dataset.train_set.dataset.labels[range(50)] == 0).all())
        self.assertTrue((full_dataset.test_set.dataset.labels[range(50, 100)] == 1).all())


    def test_load_testing_dataset(self):

        test_data, labels = load_testing_dataset(self.normal_fingerprints_path, self.normal_meta_path)
        self.assertEqual(test_data.test_set.data.shape, (50, 13155))


if __name__ == '__main__':
    unittest.main()
