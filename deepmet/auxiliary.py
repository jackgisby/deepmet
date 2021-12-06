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
#
# This file incorporates work covered by the following copyright and
# permission notice:
#
#   Copyright (c) 2018 Lukas Ruff
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import os
import csv
import json
import jpype
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, AllChem


class Config(object):
    """
    Class for storing model configuration.

    :param settings: Dictionary containing model configuration settings.
    """

    def __init__(self, settings):
        """ Constructor method """

        self.settings = settings

    def load_config(self, import_json):
        """
        Load settings dict from import_json (path/filename.json) JSON-file.

        :param import_json: The location of the json file to load settings from.
        """

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            self.settings[key] = value

    def save_config(self, export_json):
        """
        Save settings dict to export_json (path/filename.json) JSON-file. The configuration class can later be restored
        using `load_config`.

        :param export_json: Path at which to store a json file based on the configuration settings.
        """

        with open(export_json, 'w') as fp:
            json.dump(self.settings, fp)


def start_jpype():
    """
    Start `jpype` process for interface with the CDK jar executable.

    :return: Gateway to CDK java class.
    """

    if not jpype.isJVMStarted():

        cdk_path = os.path.join(os.environ["CONDA_PREFIX"], "share", "java", "cdk.jar")

        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % cdk_path)

    return jpype.JPackage('org').openscience.cdk


def cdk_fingerprint(smi, cdk, fp_type="pubchem"):
    """
    Get CDK fingerprints.

    :param smi: SMILES string representing a molecule to be converted into a fingerprint.

    :param cdk: Gateway to CDK java class, as returned :py:meth:`deepmet.auxiliary.start_jpype`.

    :param fp_type: Fingerprint to get from CDK. One of `estate`, `pubchem` or `klekota-roth`.

    :return: A bit-based fingerprint as a list.
    """

    smiles_parser = cdk.smiles.SmilesParser(cdk.DefaultChemObjectBuilder.getInstance())

    if fp_type == "estate":
        nbit = 79
        fingerprinter = cdk.fingerprint.EStateFingerprinter()

    elif fp_type == "pubchem":
        nbit = 881
        fingerprinter = cdk.fingerprint.PubchemFingerprinter(cdk.silent.SilentChemObjectBuilder.getInstance())

    elif fp_type == "klekota-roth":
        nbit = 4860
        fingerprinter = cdk.fingerprint.KlekotaRothFingerprinter()

    mol = smiles_parser.parseSmiles(smi)

    fingerprint = fingerprinter.getBitFingerprint(mol).asBitSet()

    # convert the fingerprint bits to a bit vector
    bits = []
    idx = fingerprint.nextSetBit(0)

    while idx >= 0:
        bits.append(idx)
        idx = fingerprint.nextSetBit(idx + 1)

    bit_vec = np.zeros(nbit)
    bit_vec[bits] = 1

    return bit_vec.astype(int)


def get_mol_fingerprint(smiles, mol, method_name, method, cdk, nbit=1024):
    """
    Get molecular fingerprint via a set of different methods. :py:meth:`deepmet:auxiliary:get_fingerprint_methods` can
    be used to generate a dictionary, the keys of which refer to possible values of `method_name` and values which
    refer to possible values of `method`.

    :param smiles: String containing the smiles to be converted to a bit-based fingerprint.

    :param mol: A :py:meth:`rdkit.Chem.Mol` object calculated from the input `smiles` string.

    :param method_name: Name of the method to be used to calculate fingerprints. Can be one of morgan, estate, pubchem,
        klekota-roth, maccs, mol_descriptors. Else, `method` is applied as a function to calculate fingerprints with
        the argument `fpSize` set to `nbit`.

    :param method: Function to be used to calculate fingerprints. Not required for CDK fingerprints (estate, pubchem
        and klekota-roth).

    :param nbit: The size of the fingerprint to be generated. Does not apply to fixed length fingerprints - including
        morgan, maccs or mol_descriptors fingerprint methods.

    :param cdk: Gateway to CDK java class, as returned :py:meth:`deepmet.auxiliary.start_jpype`.

    :return: A bit-based fingerprint as a list.
    """

    if "morgan" in method_name:
        fingerprint = method[0](mol, method[1], nBits=nbit)

    elif method_name in ("estate", "pubchem", "klekota-roth"):
        fingerprint = list(cdk_fingerprint(smiles, cdk, method_name))

    elif method_name in ("maccs", "mol_descriptors"):
        fingerprint = method(mol)

    else:
        fingerprint = method(mol, fpSize=nbit)

    return fingerprint


def smiles_to_matrix(smiles, mol, fingerprint_methods=None):
    """
    Get the final matrix of fingerprints from the smiles

    :param smiles: String containing the smiles to be converted to a bit-based fingerprint.

    :param mol: A :py:meth:`rdkit.Chem.Mol` object calculated from the input `smiles` string.

    :param fingerprint_methods: A dictionary describing the fingerprint methods to be used. Should be a subset of
        the dictionary created by :py:meth:`deepmet:auxiliary:get_fingerprint_methods` - if `fingerprint_methods` is
        `None`, then the entire set of fingerprints will be used by default.

    :return: A set of concatenated bit-based fingerprints as a list.
    """

    if fingerprint_methods is None:
        fingerprint_methods = get_fingerprint_methods()

    cdk = start_jpype()

    fingerprint = []
    for fingerprint_method in fingerprint_methods.keys():
        fingerprint += get_mol_fingerprint(smiles, mol, fingerprint_method, fingerprint_methods[fingerprint_method], cdk)

    return fingerprint


def get_fingerprint_methods():
    """
    Generate a dictionary containing fingerprints to be used by :py:meth:`deepmet:auxiliary:get_mol_fingerprint`

    :return: A dictionary specifying fingerprint names and methods.
    """

    return {
        "morgan_1": [AllChem.GetMorganFingerprintAsBitVect, 1],
        "morgan_2": [AllChem.GetMorganFingerprintAsBitVect, 2],
        "morgan_3": [AllChem.GetMorganFingerprintAsBitVect, 3],
        "morgan_4": [AllChem.GetMorganFingerprintAsBitVect, 4],
        "rdk": Chem.RDKFingerprint,
        "layered": Chem.LayeredFingerprint,
        "pattern": Chem.PatternFingerprint,
        "klekota-roth": "klekota-roth",
        "pubchem": "pubchem",
        "estate": "estate",
        "maccs": MACCSkeys.GenMACCSKeys
    }


def get_fingerprints_from_meta(meta_path, fingerprints_out_path):
    """
    Takes a file containing smiles and generates a file containing molecular fingerprints.

    :param meta_path: The path of a CSV file, the first column of which contains structure IDs and the second contains
        smiles.

    :param fingerprints_out_path: The path at which a new CSV file will be written containing molecular fingerprints.

    :return: The value of `fingerprints_out_path`.
    """

    # get the methods to be used for generating fingerprints
    fingerprint_methods = get_fingerprint_methods()

    # disable unnecessary RDKit messages
    RDLogger.DisableLog('rdApp.*')

    with open(meta_path, "r", encoding="utf8") as meta_file, \
         open(fingerprints_out_path, "w", newline="") as structure_fingerprint_matrix:

        # 0 - ID, 1 - smiles
        meta_csv = csv.reader(meta_file, delimiter=",")  # Input smiles
        structure_matrix_csv = csv.writer(structure_fingerprint_matrix)  # Output matrix

        for meta_row in meta_csv:

            # get RDKit Mol object
            mol = Chem.MolFromSmiles(meta_row[1])

            # write fingerprints based on smiles and RDKit Mol objects
            structure_matrix_csv.writerow(smiles_to_matrix(meta_row[1], mol, fingerprint_methods))

    return fingerprints_out_path


def select_features(normal_fingerprints_path, normal_fingerprints_out_path,
                    non_normal_fingerprints_paths=None, non_normal_fingerprints_out_paths=None,
                    unbalanced=0.1):
    """
    Carry out feature selection based on a set of molecular fingerprints.

    :param normal_fingerprints_path: The path of a set of "normal" input fingerprints, in CSV format. Feature selection
        will be performed on these data.

    :param normal_fingerprints_out_path: The path at which to write the new, reduced set of "normal" fingerprints, as a
        CSV file.

    :param non_normal_fingerprints_paths: A list of paths to CSVs containing "non-normal" input fingerprints. Feature
        selection, calculated based on the "normal" input fingerprints, will also be applied to these. If set to None,
        will solely perform feature selection on the "normal" set of fingerprints.

    :param non_normal_fingerprints_out_paths: A list of paths at which write the reduced sets of "non-normal"
        fingerprints, as CSV files.

    :param unbalanced: Float that controls the stringency of feature selection. If left as default, 0.1, features
        that are 90% a single value will be removed.

    :return: A tuple of three values, the first two of which are the values of `normal_fingerprints_out_path`,
        `non_normal_fingerprints_out_paths`. The third is a vector specifying which columns were selected by the method.
    """

    normal_fingerprints = pd.read_csv(normal_fingerprints_path, dtype=int, header=None, index_col=False)

    # Get inital dataset shape
    normal_num_rows, normal_num_cols = normal_fingerprints.shape
    normal_index = normal_fingerprints.index

    # Rename columns
    normal_fingerprints.columns = range(0, normal_num_cols)

    if non_normal_fingerprints_paths is not None:

        if not isinstance(non_normal_fingerprints_paths, list):
            non_normal_fingerprints_paths = [non_normal_fingerprints_paths]

        if not isinstance(non_normal_fingerprints_out_paths, list):
            non_normal_fingerprints_out_paths = [non_normal_fingerprints_out_paths]

        non_normal_fingerprints, non_normal_num_rows, normal_num_cols, non_normal_index = [], [], [], []

        for i in range(len(non_normal_fingerprints_paths)):
            non_normal_fingerprints.append(
                pd.read_csv(non_normal_fingerprints_paths[i], dtype=int, header=None, index_col=False))

            num_rows, num_cols = non_normal_fingerprints[i].shape
            non_normal_num_rows.append(num_rows)
            normal_num_cols.append(num_cols)

            non_normal_index.append(non_normal_fingerprints[i].index)
            non_normal_fingerprints[i].columns = range(0, normal_num_cols[i])

            # Make sure both the columns are the same
            assert all(normal_fingerprints.columns == non_normal_fingerprints[i].columns)

    # Remove unbalanced features - https://doi.org/10.1021/acs.analchem.0c01450
    # Do this just for the normal features
    cols_to_remove = []
    for i, cname in enumerate(normal_fingerprints):

        n_unique = normal_fingerprints[cname].nunique()
        if n_unique == 1:  # var=0 so remove this column
            cols_to_remove.append(i)

        elif n_unique == 2:  # var>0 so check if unbalanced

            # Get the proportion of features that match the first value
            balance_table = normal_fingerprints[cname].value_counts()
            balance = balance_table[0] / (balance_table[1] + balance_table[0])

            # Remove those that are mostly "1" or mostly "0"
            if balance > (1 - unbalanced) or balance < unbalanced:
                cols_to_remove.append(i)

        else:  # Binary features so should only be max of two different values
            assert False

    # Remove columns that are unbalanced
    normal_fingerprints.drop(cols_to_remove, axis=1, inplace=True)

    # Check that we haven't removed any samples
    normal_num_rows_processed, normal_num_cols_processed = normal_fingerprints.shape

    assert normal_num_rows_processed == normal_num_rows
    assert all(normal_fingerprints.index == normal_index)

    # Save processed matrix
    normal_fingerprints.to_csv(normal_fingerprints_out_path, header=False, index=False)

    if non_normal_fingerprints_paths is not None:
        for i in range(len(non_normal_fingerprints_paths)):
            # Remove columns that are unbalanced in the normal dataset
            non_normal_fingerprints[i].drop(cols_to_remove, axis=1, inplace=True)

            # Check no samples have been removed
            non_normal_num_rows_processed, non_normal_num_cols_processed = non_normal_fingerprints[i].shape

            assert non_normal_num_rows_processed == non_normal_num_rows[i]
            assert all(non_normal_fingerprints[i].index == non_normal_index[i])

            # Check all the columns are the same for each dataset
            assert all(normal_fingerprints.columns == non_normal_fingerprints[i].columns)

            # Save
            non_normal_fingerprints[i].to_csv(non_normal_fingerprints_out_paths[i], header=False, index=False)

    return normal_fingerprints_out_path, non_normal_fingerprints_out_paths, cols_to_remove


def drop_selected_features(fingerprints_path, fingerprints_out_path, cols_to_remove):
    """
    Drops features that have already been pre-selected by :py:meth:`deepmet:auxiliary:select_features`.

    :param fingerprints_path: The path of a set of input fingerprints, in CSV format. Features will be selected based
        on the value of `cols_to_remove`.

    :param fingerprints_out_path: The path at which to write the new, reduced set of fingerprints, as a CSV file.

    :param cols_to_remove: A vector specifying which columns are to be removed.

    :return: The value of `fingerprints_out_path`.
    """

    normal_fingerprints = pd.read_csv(fingerprints_path, dtype=int, header=None, index_col=False)

    # Remove columns that are unbalanced
    normal_fingerprints.drop(cols_to_remove, axis=1, inplace=True)

    # Save processed matrix
    normal_fingerprints.to_csv(fingerprints_out_path, header=False, index=False)

    return fingerprints_out_path
