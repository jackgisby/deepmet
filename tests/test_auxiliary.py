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

import unittest

from tests.utils import *
from deepmet.auxiliary import *


class AuxiliaryTestCase(unittest.TestCase):
    temp_results_dir = None

    @classmethod
    def to_test_results(cls, *args):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), cls.temp_results_dir.name, *args)

    @classmethod
    def setUpClass(cls):

        # create temporary directory for testing
        cls.temp_results_dir = make_temp_results_dir()

        # get a subset of the full input data
        cls.normal_meta_path, cls.non_normal_meta_path = get_normal_non_normal_subsets(cls.to_test_results())

        cls.test_smis = ('NCCc1ccc(O)c(O)c1', 'NCCc1cc(O)ccc1O', 'NCCc1cc(O)cc(O)c1')
        cls.test_mols = (Chem.MolFromSmiles(smi) for smi in cls.test_smis)

    def test_config(self):

        # load and re-save config
        loaded_cfg = Config({})
        loaded_cfg.load_config(self.to_test_results("data", "models", "config.json"))
        loaded_cfg.save_config(self.to_test_results("re_saved_config.json"))

        # re-load config and check nothing has changed
        re_loaded_cfg = Config({})
        re_loaded_cfg.load_config(self.to_test_results("re_saved_config.json"))

        self.assertEqual(loaded_cfg.settings, re_loaded_cfg.settings)

        # initialise with dictionary and check nothing has changed
        initialised_with_dict_cfg = Config(loaded_cfg.settings)

        self.assertEqual(initialised_with_dict_cfg.settings, re_loaded_cfg.settings)

        # check config contents are as expected
        self.assertEqual(len(loaded_cfg.settings["selected_features"]), 10355)
        del loaded_cfg.settings["selected_features"]

        self.assertEqual(loaded_cfg.settings, {
            'net_name': 'cocrystal_transformer',
            'objective': 'one-class',
            'nu': 0.1,
            'rep_dim': 200,
            'seed': 1,
            'optimizer_name': 'amsgrad',
            'lr': 0.000155986,
            'n_epochs': 20,
            'lr_milestones': [],
            'batch_size': 2000,
            'weight_decay': 1e-05,
            'pretrain': False,
            'in_features': 2800,
            'device': 'cpu'
        })

    def test_molecular_fingerprints(self):

        test_fingerprint_sums = (785, 809, 715)
        test_fingerprints = []

        fingerprint_methods = get_fingerprint_methods()

        # create meta file with test smiles
        with open(self.to_test_results("sample_meta.csv"), "w", newline="") as sample_meta_file:

            sample_meta_csv = csv.writer(sample_meta_file)

            for smi, mol, fingerprint_sum in zip(self.test_smis, self.test_mols, test_fingerprint_sums):

                # get the fingerprint
                long_fingerprint = smiles_to_matrix(smi, mol, fingerprint_methods)

                # check fingerprint properties
                self.assertEqual(len(long_fingerprint), 13155)
                self.assertEqual(sum(long_fingerprint), fingerprint_sum)

                sample_meta_csv.writerow([smi, smi])
                test_fingerprints.append(long_fingerprint)

        # use in-built function for getting fingerprints
        sample_fingerprints_out_path = get_fingerprints_from_meta(
            self.to_test_results("sample_meta.csv"),
            self.to_test_results("sample_fingerprints.csv")
        )

        # check get_fingerprints_from_meta gets the correct output
        fingerprints_from_file = []

        with open(sample_fingerprints_out_path, "r") as sample_fingerprints_file:

            sample_fingerprints_csv = csv.reader(sample_fingerprints_file)

            for fingerprint in sample_fingerprints_csv:
                fingerprints_from_file.append([int(s) for s in fingerprint])

        self.assertTrue(test_fingerprints == fingerprints_from_file)

    def test_select_features(self):

        # get features to remove
        cfg = Config({})
        cfg.load_config(self.to_test_results("data", "models", "config.json"))

        #  get a smaller subset of the test molecules
        normal_meta_path, non_normal_meta_path = get_normal_non_normal_subsets(
            self.to_test_results(),
            normal_sample_size=50
        )

        # get fingerprints for sample molecules
        sample_fingerprints_path = get_fingerprints_from_meta(
            normal_meta_path,
            self.to_test_results("sample_fingerprints.csv")
        )

        sample_non_normal_fingerprints_path = get_fingerprints_from_meta(
            non_normal_meta_path,
            self.to_test_results("sample_non_normal_fingerprints.csv")
        )

        # drop the features, check the output is as we expect
        sample_fingerprints_processed = drop_selected_features(
            sample_fingerprints_path,
            self.to_test_results("sample_fingerprints_processed.csv"),
            cfg.settings["selected_features"]
        )

        sample_fingerprints_processed = pd.read_csv(
            sample_fingerprints_processed,
            header=None,
            index_col=False
        )

        self.assertEqual(sample_fingerprints_processed.shape, (50, 2800))
        self.assertEqual(sample_fingerprints_processed.iloc[0].sum(), 169)

        # repeat for the non-normal fingerprints
        sample_non_normal_fingerprints_processed = drop_selected_features(
            sample_non_normal_fingerprints_path,
            self.to_test_results("sample_non_normal_fingerprints_processed.csv"),
            cfg.settings["selected_features"]
        )

        sample_non_normal_fingerprints_processed = pd.read_csv(
            sample_non_normal_fingerprints_processed,
            header=None,
            index_col=False
        )

        self.assertEqual(sample_non_normal_fingerprints_processed.shape, (50, 2800))
        self.assertEqual(sample_non_normal_fingerprints_processed.iloc[0].sum(), 929)

        # re-calculate features to drop, do it for multiple ways of passing non-normal inputs
        non_normal_inputs = {
            "list": [sample_non_normal_fingerprints_path],
            "string": sample_non_normal_fingerprints_path,
            "none": None
        }

        for non_normal_input in non_normal_inputs.keys():

            # calculate the features to drop
            regenerated_fingerprints, regenerated_non_normal_fingerprints, reselected_features = select_features(
                normal_fingerprints_path=sample_fingerprints_path,
                normal_fingerprints_out_path=self.to_test_results("regenerated_fingerprints.csv"),
                non_normal_fingerprints_paths=non_normal_inputs[non_normal_input],
                non_normal_fingerprints_out_paths=self.to_test_results("regenerated_non_normal_fingerprints.csv")
            )

            # check properties of reselected_features
            self.assertEqual(sum(reselected_features), 69458545)
            self.assertEqual(len(reselected_features), 10183)

            # check properties of regenerated_fingerprints
            regenerated_fingerprints = pd.read_csv(
                regenerated_fingerprints,
                header=None,
                index_col=False
            )

            self.assertEqual(regenerated_fingerprints.shape, (50, 2972))
            self.assertEqual(regenerated_fingerprints.iloc[0].sum(), 156)

            # check properties of regenerated_non_normal_fingerprints
            if non_normal_inputs[non_normal_input] is not None:

                regenerated_non_normal_fingerprints = pd.read_csv(
                    regenerated_non_normal_fingerprints[0],
                    header=None,
                    index_col=False
                )

                self.assertEqual(regenerated_non_normal_fingerprints.shape, (50, 2972))
                self.assertEqual(regenerated_non_normal_fingerprints.iloc[0].sum(), 938)


if __name__ == '__main__':
    unittest.main()
