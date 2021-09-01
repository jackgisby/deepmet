import os
import jpype
import numpy as np

if not jpype.isJVMStarted():

    # TODO: use DeepMet reference when these functions are properly packaged
    # cdk_path = os.path.join(DeepMet.__path__[0], 'tools', 'CDK', 'cdk-2.2.jar')
    cdk_path = os.path.join("..", "..", "DeepMet", 'tools', 'CDK', 'cdk-2.2.jar')
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
