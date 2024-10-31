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

import setuptools
import deepmet


def main():

    setuptools.setup(
        name="deepmet",
        version=deepmet.__version__,
        long_description=open("README.rst").read(),
        url="https://github.com/computational-metabolomics/deepmet",
        license="GPLv3",
        platforms=["Windows, UNIX"],
        keywords=["Metabolomics", "Lipidomics", "Mass spectrometry", "Metabolite Identification"],
        packages=setuptools.find_packages(),
        python_requires=">=3.8",
        include_package_data=True,
        classifiers=[
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.8",
          "Topic :: Scientific/Engineering :: Bio-Informatics",
          "Topic :: Scientific/Engineering :: Chemistry",
          "Topic :: Utilities",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent",
        ],
        entry_points={
         "console_scripts": [
             "deepmet = deepmet.__main__:cli"
         ]
        }
    )


if __name__ == "__main__":
    main()
