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

import os
import sys
import csv
import unittest
import tempfile
import numpy as np
from shutil import copytree

from deepmet.workflows.training import train_likeness_scorer
from deepmet.workflows.scoring import get_likeness_scores


def get_meta_subset(full_meta_path, reduced_meta_path, sample_size=200):
    with open(full_meta_path, "r", encoding="utf8") as full_meta_file:
        full_meta_csv = csv.reader(full_meta_file, delimiter=",")

        meta_rows = []

        for i, meta_row in enumerate(full_meta_csv):
            meta_rows.append(meta_row)

    random_choices = np.random.choice(i, size=sample_size, replace=False)

    with open(reduced_meta_path, "w", newline="") as reduced_meta_file:
        reduced_meta_csv = csv.writer(reduced_meta_file)

        for i, meta_row in enumerate(meta_rows):

            if i in random_choices:
                reduced_meta_csv.writerow(meta_row)

    return reduced_meta_path


class TrainModelTestCase(unittest.TestCase):
    temp_results_dir = None

    @classmethod
    def to_test_results(cls, *args):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), cls.temp_results_dir.name, *args)

    @classmethod
    def setUpClass(cls):
        # create temporary directory for testing
        cls.temp_results_dir = tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.realpath(__file__)))

        # create a copy of relevant data in the test's temporary folder
        copytree(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data"), cls.to_test_results("data"))

        # path to input metadata
        cls.normal_meta_path = cls.to_test_results("normal_meta.csv")
        cls.non_normal_meta_path = cls.to_test_results("non_normal_meta.csv")

        # get a subset of the full input data

        np.random.seed(1)
        cls.reduced_normal_meta_path = get_meta_subset(
            cls.to_test_results("data", "test_set", "hmdb_meta.csv"),
            cls.to_test_results("normal_meta.csv"),
            sample_size=500
        )

        np.random.seed(1)
        cls.reduced_non_normal_meta_path = get_meta_subset(
            cls.to_test_results("data", "test_set", "zinc_meta.csv"),
            cls.to_test_results("non_normal_meta.csv"),
            sample_size=50
        )

        cls.pretrained_results_path = cls.to_test_results("pretrained_results_path")
        os.mkdir(cls.pretrained_results_path)

        cls.pretrained_deep_met_scores = get_likeness_scores(
            dataset_path=cls.non_normal_meta_path,
            results_path=cls.pretrained_results_path,
            load_model=cls.to_test_results("data", "models", "deep_met_model.tar"),
            load_config=cls.to_test_results("data", "models", "config.json"),
            device="cpu"
        )

    def test_pretrained_deep_met(self):

        expected_scores = [2.3567147254943848, 1.4115852117538452, 1.079946517944336, 1.8736340999603271, 1.9399359226226807, 1.3693122863769531, 1.4716355800628662, 1.4825979471206665, 1.2511224746704102, 1.088046908378601, 1.5147550106048584, 1.2984886169433594, 1.6910569667816162, 1.295636534690857, 1.4793989658355713, 1.4867863655090332, 1.3723686933517456, 2.045121431350708, 1.230161190032959, 1.7905741930007935, 1.9460512399673462, 1.0301955938339233, 0.7690515518188477, 1.4042847156524658, 1.8687483072280884, 1.278300404548645, 0.9926234483718872, 0.96748286485672, 1.1696834564208984, 1.5597151517868042, 1.12373948097229, 1.6772441864013672, 1.4601625204086304, 1.06545090675354, 1.3930270671844482, 1.7143746614456177, 1.3054900169372559, 1.6427769660949707, 1.8970110416412354, 0.8174448013305664, 1.3852347135543823, 1.7129510641098022, 1.4169374704360962, 1.4309706687927246, 1.0717730522155762, 1.1838876008987427, 1.3994414806365967, 1.539568305015564, 1.7585017681121826, 1.5800352096557617]
        calculated_scores = [score_entry[2] for score_entry in self.pretrained_deep_met_scores]

        # check that the re-loaded model gives the same test results as the original model
        for original_score, rescore in zip(expected_scores, calculated_scores):

            self.assertAlmostEqual(original_score, rescore, places=5)


if __name__ == '__main__':
    unittest.main()
