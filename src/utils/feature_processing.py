import csv
import pandas as pd
from rdkit import Chem, RDLogger
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


def get_fingerprints_from_meta(meta_path, fingerprints_out_path):

    fingerprint_methods = get_fingerprint_methods()

    RDLogger.DisableLog('rdApp.*')

    with open(meta_path, "r", encoding="utf8") as meta_file, \
         open(fingerprints_out_path, "w", newline="") as structure_fingerprint_matrix:

        # 0 - ID, 1 - smiles
        meta_csv = csv.reader(meta_file, delimiter=",")  # Input smiles
        structure_matrix_csv = csv.writer(structure_fingerprint_matrix)  # Output matrix

        for meta_row in meta_csv:

            mol = Chem.MolFromSmiles(meta_row[1])
            structure_matrix_csv.writerow(smiles_to_matrix(meta_row[1], mol, fingerprint_methods))

    return fingerprints_out_path


def select_features(normal_fingerprints_path, normal_fingerprints_out_path,
                    non_normal_fingerprints_paths=None, non_normal_fingerprints_out_paths=None,
                    unbalanced=0.1):

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

        non_normal_fingerprints = []

        for i in range(len(non_normal_fingerprints_paths)):
            non_normal_fingerprints[i] = pd.read_csv(non_normal_fingerprints_paths[i], dtype=int, header=None, index_col=False)

            non_normal_num_rows, normal_num_cols = non_normal_fingerprints[i].shape
            non_normal_index = non_normal_fingerprints[i].index

            non_normal_fingerprints[i].columns = range(0, normal_num_cols)

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
            non_normal_num_rows_processed, non_normal_num_cols_processed = normal_fingerprints.shape
    
            assert non_normal_num_rows_processed == non_normal_num_rows[i]
            assert all(non_normal_fingerprints[i].index == non_normal_index[i])
    
            # Check all the columns are the same for each dataset
            assert all(normal_fingerprints.columns == non_normal_fingerprints[i].columns)
    
            # Save
            non_normal_fingerprints[i].to_csv(non_normal_fingerprints_out_paths[i], header=False, index=False)

    return normal_fingerprints_out_path, non_normal_fingerprints_out_paths
