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
import csv
import tempfile
import numpy as np
from shutil import copytree


def make_temp_results_dir():

    temp_results_dir = tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.realpath(__file__)))

    # create a copy of relevant data in the test's temporary folder
    copytree(
        os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "deepmet", "data"),
        os.path.join(temp_results_dir.name, "data")
    )

    return temp_results_dir


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


def get_normal_non_normal_subsets(test_results_path, seed=1):

    np.random.seed(seed)
    reduced_normal_meta_path = get_meta_subset(
        os.path.join(test_results_path, "data", "test_set", "hmdb_meta.csv"),
        os.path.join(test_results_path, "normal_meta.csv"),
        sample_size=500
    )

    np.random.seed(seed)
    reduced_non_normal_meta_path = get_meta_subset(
        os.path.join(test_results_path, "data", "test_set", "zinc_meta.csv"),
        os.path.join(test_results_path, "non_normal_meta.csv"),
        sample_size=50
    )

    return reduced_normal_meta_path, reduced_non_normal_meta_path
