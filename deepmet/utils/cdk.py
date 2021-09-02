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
import jpype
import numpy as np

import deepmet

if not jpype.isJVMStarted():

    cdk_path = os.path.join(deepmet.__path__[0], os.pardir, 'tools', 'CDK', 'cdk-2.2.jar')

    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % cdk_path)
    cdk = jpype.JPackage('org').openscience.cdk


def cdk_parser_smiles(smi):
    sp = cdk.smiles.SmilesParser(cdk.DefaultChemObjectBuilder.getInstance())
    try:
        mol = sp.parseSmiles(smi)
    except:
        raise IOError('invalid smiles input')
    return mol


def cdk_fingerprint(smi, fp_type="pubchem"):

    if fp_type == 'estate':
        nbit = 79
    elif fp_type == 'pubchem':
        nbit = 881
    elif fp_type == 'klekota-roth':
        nbit = 4860

    _fingerprinters = {
        "pubchem": cdk.fingerprint.PubchemFingerprinter(cdk.silent.SilentChemObjectBuilder.getInstance()),
        "estate": cdk.fingerprint.EStateFingerprinter(),
        "klekota-roth": cdk.fingerprint.KlekotaRothFingerprinter()
    }

    mol = cdk_parser_smiles(smi)
    if fp_type in _fingerprinters:
        fingerprinter = _fingerprinters[fp_type]
    else:
        raise IOError('invalid fingerprint type')

    fp = fingerprinter.getBitFingerprint(mol).asBitSet()

    bits = []
    idx = fp.nextSetBit(0)

    while idx >= 0:
        bits.append(idx)
        idx = fp.nextSetBit(idx + 1)

    vec = np.zeros(nbit)
    vec[bits] = 1

    return vec.astype(int)
