import os
import csv
import shutil
from random import sample
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, AllChem, Descriptors
# from rdkit.ML.Descriptors import MoleculeDescriptors
from PyFingerprint.All_Fingerprint import get_fingerprint


def get_mol_fingerprint(smiles, mol, method_name, method, nbit=1024):

    if "morgan" in method_name:
        fingerprint = method[0](mol, method[1], nBits=nbit)

    elif isinstance(method, str):
        fingerprint = list(get_fingerprint(smiles, method, output="cats", nbit=nbit))

    elif method_name == "maccs":
        fingerprint = method(mol)

    else:
        fingerprint = method(mol, fpSize=nbit)

    return fingerprint


def smiles_qc(smiles):

    mol = Chem.MolFromSmiles(smiles)

    try:
        Chem.SanitizeMol(mol)
    except:
        return

    if mol is None:
        return

    if mol.GetNumHeavyAtoms() < 4:
        return

    exact_mass = Descriptors.ExactMolWt(mol)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    if exact_mass < 100 or exact_mass > 800:
        return

    if len(smiles) < 4:
        print(smiles)
        assert False

    return smiles


def smiles_to_matrix(smiles, mol, fingerprint_methods):

    fingerprint = []
    for fingerprint_method in fingerprint_methods.keys():
        fingerprint += get_mol_fingerprint(smiles, mol, fingerprint_method, fingerprint_methods[fingerprint_method])

    assert len(fingerprint) == 12131

    return fingerprint


def get_fingerprint_methods():

    # descriptor_list = [x[0] for x in Descriptors._descList]
    # calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_list)

    return {
        "morgan_2": [AllChem.GetMorganFingerprintAsBitVect, 2],
        "morgan_3": [AllChem.GetMorganFingerprintAsBitVect, 3],
        "morgan_4": [AllChem.GetMorganFingerprintAsBitVect, 4],
        "rdk": Chem.RDKFingerprint,
        # "mol_descriptors": calculator.CalcDescriptors,
        "layered": Chem.LayeredFingerprint,
        "pattern": Chem.PatternFingerprint,
        "klekota_roth": "klekota-roth",
        "pubchem": "pubchem",
        "estate": "estate",
        "maccs": MACCSkeys.GenMACCSKeys
    }


def get_fingerprints_from_smiles(filename, prefix, smiles_col, first_row, subset_n=20000):

    with open("../../../../Data/other_databases/" + filename, "r", encoding="utf8") as structures, \
         open("../data/mol_key_test/" + prefix.lower() + "_fingerprints.csv", "w", newline="") as structure_fingerprint_matrix, \
         open("../data/mol_key_test/" + prefix.lower() + "_meta.csv", "w", newline="") as structure_meta:

        structure_csv = csv.reader(structures, delimiter="\t")
        structure_matrix_csv = csv.writer(structure_fingerprint_matrix)
        structure_meta_csv = csv.writer(structure_meta)

        structure_subset = set()

        for row in structure_csv:

            if row[0] == first_row:
                continue

            smiles = smiles_qc(row[smiles_col])

            if smiles is not None and smiles != "":
                structure_subset.add(smiles)

        if subset_n is not None and subset_n != 0:
            structure_subset = sample(structure_subset, subset_n)

        for i, smiles in enumerate(structure_subset):

            mol = Chem.MolFromSmiles(smiles)

            structure_meta_csv.writerow([prefix + "_" + str(i), smiles])
            structure_matrix_csv.writerow(smiles_to_matrix(smiles, mol, fingerprint_methods))


if __name__ == "__main__":

    RDLogger.DisableLog('rdApp.*')

    # if os.path.exists("../data/mol_key_test"):
    #     shutil.rmtree("../data/mol_key_test")
    #
    # os.mkdir("../data/mol_key_test")

    fingerprint_methods = get_fingerprint_methods()

    filenames = ["chembl_28_chemreps.txt"]  # , "10_prop.csv", "metabolites-2021-06-20"
    prefixes = ["CHEMBL"],  # , "ZINC", "HMDB"
    smiles_cols = [1],  # , 10, 2
    first_rows = ["chembl_id"]  # , "ZINC_ID", "HMDB_ID"

    for filename, prefix, smiles_col, first_row in zip(filenames, prefixes, smiles_cols, first_rows):

        if prefix in ("HMDB",):
            subset_n = None
        else:
            subset_n = 20000

        get_fingerprints_from_smiles(filename, prefix, smiles_col, first_row, subset_n)
