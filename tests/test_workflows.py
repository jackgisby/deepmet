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
import pandas as pd

import unittest

from tests.utils import *
from deepmet.auxiliary import Config
from deepmet.datasets import load_training_dataset
from deepmet.workflows import train_likeness_scorer, train_single_model, get_likeness_scores


class TrainModelTestCase(unittest.TestCase):
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

        # create a folder for a newly trained model
        cls.fresh_results_path = cls.to_test_results("deepmet_model_fresh")
        os.mkdir(cls.fresh_results_path)

        # train the deepmet model using a subset of the data
        cls.deepmet_model_fresh = train_likeness_scorer(
            cls.normal_meta_path,
            cls.fresh_results_path,
            non_normal_meta_path=cls.non_normal_meta_path,
            normal_fingerprints_path=None,
            non_normal_fingerprints_path=None,
            net_name="cocrystal_transformer",
            objective="one-class",
            nu=0.1,
            rep_dim=200,
            device="cuda",
            seed=1,
            optimizer_name="amsgrad",
            lr=0.000100095,
            n_epochs=20,
            lr_milestones=tuple(),
            batch_size=10,
            weight_decay=1e-5,
            validation_split=0.8,
            test_split=None,
            filter_features=True
        )

        # create folder for the scoring
        cls.rescore_results_path = cls.to_test_results("deepmet_model_rescored")
        os.mkdir(cls.rescore_results_path)

        # perform scoring using the new model
        cls.deepmet_results_rescore = get_likeness_scores(
            dataset_path=cls.non_normal_meta_path,
            results_path=cls.rescore_results_path,
            load_model=os.path.join(cls.fresh_results_path, "model.tar"),
            load_config=os.path.join(cls.fresh_results_path, "config.json"),
            device="cpu"
        )

    def test_trained_deepmet(self):

        # does the newly trained DeepMet model have the expected test results
        self.assertTrue(0.5 < self.deepmet_model_fresh.results["test_loss"] < 2)

    def test_rescored_deepmet(self):

        # ensure expected test subset has been selected
        zinc_test_set = [
            'ZINC_416', 'ZINC_645', 'ZINC_1545', 'ZINC_2402', 'ZINC_3260', 'ZINC_3665', 'ZINC_4107', 'ZINC_4212',
            'ZINC_4908', 'ZINC_5544', 'ZINC_5874', 'ZINC_6399', 'ZINC_6480', 'ZINC_6545', 'ZINC_6683', 'ZINC_7084',
            'ZINC_7660', 'ZINC_7691', 'ZINC_7807', 'ZINC_7840', 'ZINC_8111', 'ZINC_8203', 'ZINC_9584', 'ZINC_9587',
            'ZINC_10447', 'ZINC_11455', 'ZINC_11456', 'ZINC_12471', 'ZINC_12498', 'ZINC_13429', 'ZINC_13587',
            'ZINC_13691', 'ZINC_14704', 'ZINC_14833', 'ZINC_15104', 'ZINC_15575', 'ZINC_15592', 'ZINC_16830',
            'ZINC_16968', 'ZINC_17275', 'ZINC_17878', 'ZINC_17894', 'ZINC_18018', 'ZINC_18058', 'ZINC_18285',
            'ZINC_18334', 'ZINC_19005', 'ZINC_19548', 'ZINC_19628', 'ZINC_19715'
        ]

        self.assertEqual([score_entry[0] for score_entry in self.deepmet_results_rescore], zinc_test_set)

        # get the scores on the test set
        original_scores = [score_entry[2] for score_entry in self.deepmet_model_fresh.results["test_scores"]]
        rescores = [score_entry[2] for score_entry in self.deepmet_results_rescore]

        # check that the re-loaded model gives the same test results as the original model
        for original_score, rescore in zip(original_scores, rescores):

            self.assertAlmostEqual(original_score, rescore, places=5)

    def test_feature_processing(self):

        # files containing fingerprints
        fingerprint_csvs = (
            "normal_fingerprints.csv",
            "non_normal_fingerprints.csv",
            "normal_fingerprints_processed.csv",
            "non_normal_fingerprints_processed.csv"
        )

        # the fingerprint sum of the first molecule for each fingerprint set
        fingerprint_first_row_sums = (1640, 1355, 1088, 917)

        for fingerprint_csv, fingerprint_first_row_sum in zip(fingerprint_csvs, fingerprint_first_row_sums):

            # load the fingerprint as a pandas dataframe
            fingerprints = pd.read_csv(
                os.path.join(self.fresh_results_path, fingerprint_csv),
                dtype=int,
                header=None,
                index_col=False
            )

            # check the first row has the correct values
            self.assertEqual(fingerprints.iloc[0].sum(), fingerprint_first_row_sum)

            # check the fingerprints have the right shape
            num_rows, num_cols = fingerprints.shape
            self.assertEqual(num_rows, 50 if "non_normal" in fingerprint_csv else 500)
            self.assertEqual(num_cols, 2746 if "processed" in fingerprint_csv else 13155)

    def test_train_with_modified_parameters(self):

        # train without non-normal class
        train_likeness_scorer(
            self.normal_meta_path,
            self.to_test_results()
        )

        # train without filtering
        train_likeness_scorer(
            self.normal_meta_path,
            self.to_test_results(),
            filter_features=False
        )

        # train without test_split
        train_likeness_scorer(
            self.normal_meta_path,
            self.to_test_results(),
            non_normal_meta_path=self.non_normal_meta_path,
            test_split=None
        )

    def test_train_single_model(self):

        # load the model config so we can provide it to train_single_model
        cfg = Config(dict())
        cfg.load_config(os.path.join(self.fresh_results_path, "config.json"))

        # load the dataset so we can provide it to train_single_model
        dataset, dataset_labels, validation_dataset = load_training_dataset(
            normal_dataset_path=os.path.join(self.fresh_results_path, "normal_fingerprints_processed.csv"),
            normal_meta_path=self.normal_meta_path,
            non_normal_dataset_path=os.path.join(self.fresh_results_path, "non_normal_fingerprints_processed.csv"),
            non_normal_dataset_meta_path=self.non_normal_meta_path,
            test_split=None
        )

        # train the model with train_single_model
        single_trained_model = train_single_model(cfg, validation_dataset)
        single_trained_model.test(dataset)

        # get the scores on the test set
        original_scores = [score_entry[2] for score_entry in self.deepmet_model_fresh.results["test_scores"]]
        train_single_model_scores = [score_entry[2] for score_entry in single_trained_model.results["test_scores"]]

        # check that the re-loaded model gives the same test results as the original model
        for original_score, train_single_model_score in zip(original_scores, train_single_model_scores):

            self.assertAlmostEqual(original_score, train_single_model_score, places=5)


class ScoreModelTestCase(unittest.TestCase):
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

        # carry out likeness scoring using pretrained model
        cls.pretrained_results_path = cls.to_test_results("pretrained_results_path")
        os.mkdir(cls.pretrained_results_path)

        cls.pretrained_deepmet_scores = get_likeness_scores(
            dataset_path=cls.non_normal_meta_path,
            results_path=cls.pretrained_results_path,
            load_model=cls.to_test_results("data", "models", "deep_met_model.tar"),
            load_config=cls.to_test_results("data", "models", "config.json"),
            device="cpu"
        )

    def test_pretrained_deepmet(self):

        # these are the expected scores
        expected_scores = [
            2.3567147254943848, 1.4115852117538452, 1.079946517944336, 1.8736340999603271,
            1.9399359226226807, 1.3693122863769531, 1.4716355800628662, 1.4825979471206665,
            1.2511224746704102, 1.088046908378601, 1.5147550106048584, 1.2984886169433594,
            1.6910569667816162, 1.295636534690857, 1.4793989658355713, 1.4867863655090332,
            1.3723686933517456, 2.045121431350708, 1.230161190032959, 1.7905741930007935,
            1.9460512399673462, 1.0301955938339233, 0.7690515518188477, 1.4042847156524658,
            1.8687483072280884, 1.278300404548645, 0.9926234483718872, 0.96748286485672,
            1.1696834564208984, 1.5597151517868042, 1.12373948097229, 1.6772441864013672,
            1.4601625204086304, 1.06545090675354, 1.3930270671844482, 1.7143746614456177,
            1.3054900169372559, 1.6427769660949707, 1.8970110416412354, 0.8174448013305664,
            1.3852347135543823, 1.7129510641098022, 1.4169374704360962, 1.4309706687927246,
            1.0717730522155762, 1.1838876008987427, 1.3994414806365967, 1.539568305015564,
            1.7585017681121826, 1.5800352096557617
        ]

        # these are the calculated scores, generated by supplying model to load_model
        calculated_scores = [score_entry[2] for score_entry in self.pretrained_deepmet_scores]

        # now we calculate scores without specifying load_model
        self.pretrained_deepmet_scores_no_model = get_likeness_scores(
            dataset_path=self.non_normal_meta_path,
            results_path=self.pretrained_results_path,
            device="cpu"
        )

        calculated_scores_no_model = [score_entry[2] for score_entry in self.pretrained_deepmet_scores_no_model]

        zip_scores = zip(expected_scores, calculated_scores, calculated_scores_no_model)

        # check that the re-scores are the same as the scores for the original model
        for original_score, rescores, rescores_no_model in zip_scores:

            self.assertAlmostEqual(original_score, rescores, places=5)
            self.assertAlmostEqual(original_score, rescores_no_model, places=5)

    def test_feature_processing(self):

        # csv files containing fingerprints
        fingerprint_csvs = ("input_fingerprints.csv", "input_fingerprints_processed.csv")

        # the fingerprint sum of the first molecule for each fingerprint set
        fingerprint_first_row_sums = (1355, 929)

        for fingerprint_csv, fingerprint_first_row_sum in zip(fingerprint_csvs, fingerprint_first_row_sums):

            # get the csv as a pandas dataframe
            fingerprints = pd.read_csv(
                os.path.join(self.pretrained_results_path, fingerprint_csv),
                dtype=int,
                header=None,
                index_col=False
            )

            # check the first row has the correct values
            self.assertEqual(fingerprints.iloc[0].sum(), fingerprint_first_row_sum)

            # check the dimensions of the fingerprints
            num_rows, num_cols = fingerprints.shape
            self.assertEqual(num_rows, 50)
            self.assertEqual(num_cols, 2800 if "processed" in fingerprint_csv else 13155)


if __name__ == '__main__':
    unittest.main()
