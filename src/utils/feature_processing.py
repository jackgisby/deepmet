import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem

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

def select_features(normal_fingerprints_path, non_normal_fingerprints_path, unbalanced=0.1):

    normal_fingerprints = pd.read_csv(normal_fingerprints_path, dtype=int, header=None, index_col=False)
    non_normal_fingerprints = pd.read_csv(non_normal_fingerprints_path, dtype=int, header=None, index_col=False)

    # Get inital dataset shape
    normal_num_rows, normal_num_cols = normal_fingerprints.shape
    normal_index = normal_fingerprints.index
    
    non_normal_num_rows, normal_num_cols = non_normal_fingerprints.shape
    non_normal_index = non_normal_fingerprints.index

    # Rename columns
    normal_fingerprints.columns = range(0, normal_num_cols)
    non_normal_fingerprints.columns = range(0, normal_num_cols)

    # Make sure both the columns are the same
    assert all(normal_fingerprints.columns == non_normal_fingerprints.columns)

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

    # Remove columns based on those that are unbalanced in the normal dataset
    normal_fingerprints.drop(cols_to_remove, axis=1, inplace=True)
    non_normal_fingerprints.drop(cols_to_remove, axis=1, inplace=True)

    # Check that we haven't removed any samples
    normal_num_rows_processed, normal_num_cols_processed = normal_fingerprints.shape
    non_normal_num_rows_processed, non_normal_num_cols_processed = normal_fingerprints.shape

    assert normal_num_rows_processed == normal_num_rows
    assert all(normal_fingerprints.index == normal_index)
    assert non_normal_num_rows_processed == non_normal_num_rows
    assert all(non_normal_fingerprints.index == non_normal_index)

    # Check all the columns are the same for each dataset
    assert all(normal_fingerprints.columns == non_normal_fingerprints.columns)

    # Save
    new_normal_fingerprints_path = "../data/mol_key_test/normal_fingerprints_processed.csv"
    new_non_normal_fingerprints_path = "../data/mol_key_test/non_normal_fingerprints_processed.csv"

    normal_fingerprints.to_csv(new_normal_fingerprints_path, header=False, index=False)
    normal_fingerprints.to_csv(new_non_normal_fingerprints_path, header=False, index=False)

    return new_normal_fingerprints_path, new_non_normal_fingerprints_path
