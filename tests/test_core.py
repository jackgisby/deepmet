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

from tests.utils import *
from deepmet.core import *
from deepmet.networks import CocrystalTransformer


class CoreTestCase(unittest.TestCase):
    temp_results_dir = None

    @classmethod
    def to_test_results(cls, *args):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), cls.temp_results_dir.name, *args)

    @classmethod
    def setUpClass(cls):

        # create temporary directory for testing
        cls.temp_results_dir = make_temp_results_dir()

    def test_deepmet(self):

        # some parameters aren't allowed
        with self.assertRaises(AssertionError):
            DeepMet(nu=-1)

        with self.assertRaises(AssertionError):
            DeepMet(objective="two-class")

        # create DeepMet class
        deepmet = DeepMet(rep_dim=200, in_features=2800)

        # set network
        deepmet.set_network("cocrystal_transformer")
        self.assertIsInstance(deepmet.net, CocrystalTransformer)

        # load a model
        deepmet.load_model(self.to_test_results("data", "models", "deep_met_model.tar"))
        self.assertEqual(deepmet.R, 0)
        self.assertAlmostEqual(sum(deepmet.c), 15.8816223219, places=5)

        deepmet.save_model(self.to_test_results("reloaded_deep_met_model.tar"))

        # reload model, check again
        deepmet_reloaded = DeepMet(rep_dim=200, in_features=2800)
        deepmet_reloaded.set_network("cocrystal_transformer")
        deepmet_reloaded.load_model(self.to_test_results("reloaded_deep_met_model.tar"))

        self.assertEqual(deepmet_reloaded.R, 0)
        self.assertAlmostEqual(sum(deepmet_reloaded.c), 15.8816223219, places=5)
        self.assertEqual(str(deepmet.net), str(deepmet_reloaded.net))

    def test_get_radius(self):

        c = torch.ones(200, device="cpu") * 10
        self.assertAlmostEqual(get_radius(c, 0.5), 3.1622776985168457, places=5)


if __name__ == '__main__':
    unittest.main()
