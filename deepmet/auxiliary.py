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
import json
import jpype


class Config(object):
    """
    Class for storing model configuration.

    :param settings: Dictionary containing model configuration settings.
    """

    def __init__(self, settings: dict):

        self.settings = settings

    def load_config(self, import_json: str):
        """
        Load settings dict from import_json (path/filename.json) JSON-file.

        :param import_json: The location of the json file to load settings from.
        """

        with open(import_json, "r") as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            self.settings[key] = value

    def save_config(self, export_json: str):
        """
        Save settings dict to export_json (path/filename.json) JSON-file. The
        configuration class can later be restored using `load_config`.

        :param export_json: Path at which to store a json file based on the
            configuration settings.
        """

        with open(export_json, "w") as fp:
            json.dump(self.settings, fp)


def start_jpype() -> jpype.JPackage:
    """
    Start `jpype` process for interface with the CDK jar executable.

    :return: Gateway to CDK java class.
    """

    if not jpype.isJVMStarted():

        cdk_path = os.path.join(os.environ["CONDA_PREFIX"], "share", "java", "cdk.jar")

        jpype.startJVM(
            jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % cdk_path
        )

    return jpype.JPackage("org").openscience.cdk
