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
        return os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            cls.temp_results_dir.name,
            *args
        )

    @classmethod
    def setUpClass(cls):

        # create temporary directory for testing
        cls.temp_results_dir = make_temp_results_dir()

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

        self.assertEqual(
            loaded_cfg.settings,
            {
                "net_name": "cocrystal_transformer",
                "objective": "one-class",
                "nu": 0.1,
                "rep_dim": 200,
                "seed": 1,
                "optimizer_name": "amsgrad",
                "lr": 0.000155986,
                "n_epochs": 20,
                "lr_milestones": [],
                "batch_size": 2000,
                "weight_decay": 1e-05,
                "pretrain": False,
                "in_features": 2800,
                "device": "cpu",
            },
        )


if __name__ == "__main__":
    unittest.main()
