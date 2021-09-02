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

import unittest
import pandas as pd

from tests.auxiliary import *
from deepmet.workflows import get_likeness_scores


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

        # carry out likeness scoring using pretrained model
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

    def test_feature_processing(self):

        fingerprint_csvs = ("input_fingerprints.csv", "input_fingerprints_processed.csv")

        for fingerprint_csv in fingerprint_csvs:

            fingerprints = pd.read_csv(
                os.path.join(self.pretrained_results_path, fingerprint_csv),
                dtype=int,
                header=None,
                index_col=False
            )

            num_rows, num_cols = fingerprints.shape

            if "processed" in fingerprint_csv:
                self.assertEqual(num_cols, 2800)
            else:
                self.assertEqual(num_cols, 13155)


if __name__ == '__main__':
    unittest.main()
