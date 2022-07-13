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
from deepmet.networks import *


class NetworksTestCase(unittest.TestCase):
    temp_results_dir = None

    @classmethod
    def to_test_results(cls, *args):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), cls.temp_results_dir.name, *args)

    @classmethod
    def setUpClass(cls):

        # create temporary directory for testing
        cls.temp_results_dir = make_temp_results_dir()

    def test_build_network(self):

        # try building basic network
        net = build_network("basic_multilayer", 200, 2800)
        self.assertIsInstance(net.forward(torch.Tensor(50, 2800)), torch.Tensor)

        # try building transformer network
        net = build_network("cocrystal_transformer", 200, 2800)
        self.assertIsInstance(net.forward(torch.Tensor(50, 2800)), torch.Tensor)

    def test_networks(self):

        # try building basic network
        net = BasicMultilayer(200, 2800)
        self.assertIsInstance(net.forward(torch.Tensor(50, 2800)), torch.Tensor)

        # try building transformer network
        net = CocrystalTransformer(200, 2800)
        self.assertIsInstance(net.forward(torch.Tensor(50, 2800)), torch.Tensor)


if __name__ == '__main__':
    unittest.main()
