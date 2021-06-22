import os
import csv
import shutil
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from PyFingerprint.All_Fingerprint import get_fingerprint


def get_mol_fingerprint(smiles, mol, method):

    if isinstance(method, list):
        fingerprint = method[0](mol, method[1])

    elif isinstance(method, str):
        fingerprint = list(get_fingerprint(smiles, method, output="cats"))

    else:
        fingerprint = method(mol)

    return fingerprint


if __name__ == "__main__":

    RDLogger.DisableLog('rdApp.*')

    if os.path.exists("../data/mol_key_test"):
        shutil.rmtree("../data/mol_key_test")

    os.mkdir("../data/mol_key_test")

    descriptor_list = [x[0] for x in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_list)

    fingerprint_methods = {
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
        "circular": "circular",
        "maccs": MACCSkeys.GenMACCSKeys
    }

    with open("../../../../Data/hmdb/hmdb_metabolites/metabolites-2021-06-20", "r", encoding="utf8") as hmdbs, \
         open("../data/mol_key_test/fingerprints.csv", "w", newline="") as fingerprint_matrix, \
         open("../data/mol_key_test/meta.csv", "w", newline="") as meta:

        hmdb_csv = csv.reader(hmdbs)
        matrix_csv = csv.writer(fingerprint_matrix)
        meta_csv = csv.writer(meta)

        # 0: HMDB_ID, 2: smiles, 6: mono mass
        for i, row in enumerate(hmdb_csv):

            smiles = None
            mol = None

            if row[0] == "HMDB_ID":
                continue

            mol = Chem.MolFromSmiles(row[2])

            try:
                Chem.SanitizeMol(mol)
            except:
                continue

            if mol is None:
                continue

            if mol.GetNumHeavyAtoms() < 4:
                continue

            atom_check = [True for atom in mol.GetAtoms() if atom.GetSymbol() not in ["C", "H", "N", "O", "P", "S"]]
            if len(atom_check) > 0:
                continue

            smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

            if "+" in smiles or "-" in smiles:
                continue

            mol = Chem.MolFromSmiles(smiles)

            if len(smiles) < 4:
                print(smiles)
                assert False

            meta_csv.writerow([row[0], smiles])

            fingerprint = []
            for fingerprint_method in fingerprint_methods.keys():

                fingerprint += get_mol_fingerprint(smiles, mol, fingerprint_methods[fingerprint_method])

            assert len(fingerprint) == 19299

            matrix_csv.writerow(fingerprint)
