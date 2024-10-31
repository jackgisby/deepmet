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
from json import load

import unittest
from click.testing import CliRunner

from tests.utils import *
from deepmet.__main__ import *


class CLITestCase(unittest.TestCase):
    temp_results_dir = None

    @classmethod
    def to_test_results(cls, *args):
        return os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            cls.temp_results_dir.name,
            *args
        )

    @classmethod
    def setUpClass(cls):

        # create temporary directory for testing
        cls.temp_results_dir = make_temp_results_dir()

        # get a CliRunner to test the command line
        cls.runner = CliRunner()

        # create temporary directory for testing
        cls.temp_results_dir = make_temp_results_dir()

        # get a subset of the full input data
        cls.normal_meta_path, cls.non_normal_meta_path = get_normal_non_normal_subsets(
            cls.to_test_results(), normal_sample_size=50
        )

        # create a folder for a newly trained model
        cls.deepmet_results_path = cls.to_test_results("deepmet_model")
        os.mkdir(cls.deepmet_results_path)

        # train the deepmet model using a subset of the data
        cls.deepmet_model = train_likeness_scorer(
            cls.normal_meta_path,
            cls.deepmet_results_path,
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
            filter_features=True,
        )

        # perform scoring using the new model
        cls.deepmet_scores = get_likeness_scores(
            dataset_path=cls.non_normal_meta_path,
            results_path=cls.deepmet_results_path,
            load_model=os.path.join(cls.deepmet_results_path, "model.tar"),
            load_config=os.path.join(cls.deepmet_results_path, "config.json"),
            device="cpu",
        )

    def test_cli(self):

        # test help
        cli_help = self.runner.invoke(cli, ["--help"])

        self.assertEqual(cli_help.exit_code, 0)
        self.assertEqual(cli_help.output[0:35], "Usage: cli [OPTIONS] COMMAND [ARGS]")

    def test_score_command(self):

        # test help
        score_help = self.runner.invoke(score, ["--help"])

        self.assertEqual(score_help.exit_code, 0)
        self.assertEqual(
            score_help.output[0:39], "Usage: score [OPTIONS] NORMAL_META_PATH"
        )

        # try scoring with model
        os.mkdir(self.to_test_results("cli_scoring_results"))

        score_with_model = self.runner.invoke(
            score,
            [
                self.non_normal_meta_path,
                "--output_path",
                self.to_test_results("cli_scoring_results"),
                "--load_model",
                os.path.join(self.deepmet_results_path, "model.tar"),
                "--load_config",
                os.path.join(self.deepmet_results_path, "config.json"),
            ],
        )

        self.assertEqual(score_with_model.exit_code, 0)

        with open(
            self.to_test_results("cli_scoring_results", "scores.json"), "r"
        ) as scores_json:
            cli_scores = load(scores_json)

        for cli_score, deepmet_score in zip(cli_scores, self.deepmet_scores):
            self.assertEqual(cli_score[0], deepmet_score[0])
            self.assertAlmostEqual(cli_score[2], deepmet_score[2], places=5)

        # check without load_ arguments
        score_no_model = self.runner.invoke(
            score,
            [
                self.non_normal_meta_path,
                "--output_path",
                self.to_test_results("cli_scoring_results"),
            ],
        )

        self.assertEqual(score_no_model.exit_code, 0)

    def test_train_command(self):

        # test help
        train_help = self.runner.invoke(train, ["--help"])

        self.assertEqual(train_help.exit_code, 0)
        self.assertEqual(
            train_help.output[0:39], "Usage: train [OPTIONS] NORMAL_META_PATH"
        )

        # create model
        os.mkdir(self.to_test_results("cli_training_results"))

        train_model = self.runner.invoke(
            train,
            [
                self.normal_meta_path,
                "--output_path",
                self.to_test_results("cli_training_results"),
                "--non_normal_path",
                self.non_normal_meta_path,
                "--lr",
                0.000100095,
                "--batch_size",
                10,
                "--optimizer_name",
                "amsgrad",
                "--seed",
                1,
            ],
        )

        self.assertEqual(train_model.exit_code, 0)

        # get the scores for the newly trained model
        score_cli_model = self.runner.invoke(
            score,
            [
                self.non_normal_meta_path,
                "--output_path",
                self.to_test_results("cli_training_results"),
                "--load_model",
                self.to_test_results("cli_training_results", "model.tar"),
                "--load_config",
                self.to_test_results("cli_training_results", "config.json"),
            ],
        )

        self.assertEqual(score_cli_model.exit_code, 0)

        # check scores match expected
        with open(
            self.to_test_results("cli_training_results", "scores.json"), "r"
        ) as scores_json:
            cli_scores = load(scores_json)

        for cli_score, deepmet_score in zip(cli_scores, self.deepmet_scores):
            self.assertEqual(cli_score[0], deepmet_score[0])
            self.assertAlmostEqual(cli_score[2], deepmet_score[2], places=5)


if __name__ == "__main__":
    unittest.main()
