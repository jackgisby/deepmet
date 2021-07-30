from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from .cdk import cdk_fingerprint


def get_mol_fingerprint(smiles, mol, method_name, method, nbit=1024):

    if "morgan" in method_name:
        fingerprint = method[0](mol, method[1], nBits=nbit)

    elif isinstance(method, str):
        fingerprint = list(cdk_fingerprint(smiles, method))

    elif method_name in ("maccs", "mol_descriptors"):
        fingerprint = method(mol)

    else:
        fingerprint = method(mol, fpSize=nbit)

    return fingerprint


def smiles_to_matrix(smiles, mol, fingerprint_methods):
    """Get the final matrix of fingerprints from the smiles"""

    fingerprint = []
    for fingerprint_method in fingerprint_methods.keys():
        fingerprint += get_mol_fingerprint(smiles, mol, fingerprint_method, fingerprint_methods[fingerprint_method])

    assert len(fingerprint) == 13155

    return fingerprint


def get_fingerprint_methods():

    return {
        "morgan_1": [AllChem.GetMorganFingerprintAsBitVect, 1],
        "morgan_2": [AllChem.GetMorganFingerprintAsBitVect, 2],
        "morgan_3": [AllChem.GetMorganFingerprintAsBitVect, 3],
        "morgan_4": [AllChem.GetMorganFingerprintAsBitVect, 4],
        "rdk": Chem.RDKFingerprint,
        "layered": Chem.LayeredFingerprint,
        "pattern": Chem.PatternFingerprint,
        "klekota_roth": "klekota-roth",
        "pubchem": "pubchem",
        "estate": "estate",
        "maccs": MACCSkeys.GenMACCSKeys
    }
