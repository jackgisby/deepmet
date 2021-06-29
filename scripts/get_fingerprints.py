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


if __name__ == "__main__":

    RDLogger.DisableLog('rdApp.*')

    if os.path.exists("../data/mol_key_test"):
        shutil.rmtree("../data/mol_key_test")

    os.mkdir("../data/mol_key_test")

    fingerprint_methods = get_fingerprint_methods()

    # with open("../../../../Data/zinc12/10_prop.csv", "r", encoding="utf8") as zincs, \
    #      open("../data/mol_key_test/zinc_fingerprints.csv", "w", newline="") as zinc_fingerprint_matrix, \
    #      open("../data/mol_key_test/zinc_meta.csv", "w", newline="") as zinc_meta:
    #
    #     zinc_csv = csv.reader(zincs, delimiter="\t")
    #     zinc_matrix_csv = csv.writer(zinc_fingerprint_matrix)
    #     zinc_meta_csv = csv.writer(zinc_meta)
    #
    #     zinc_subset = set()
    #
    #     # 0: ZINC_ID, 10: smiles
    #     for row in zinc_csv:
    #
    #         if row[0] == "ZINC_ID":
    #             continue
    #
    #         smiles = smiles_qc(row[10])
    #
    #         if smiles is not None and smiles != "":
    #             zinc_subset.add(smiles)
    #
    #     zinc_subset = sample(zinc_subset, 20000)
    #
    #     for i, smiles in enumerate(zinc_subset):
    #
    #         mol = Chem.MolFromSmiles(smiles)
    #
    #         zinc_meta_csv.writerow(["ZINC_" + str(i), smiles])
    #         zinc_matrix_csv.writerow(smiles_to_matrix(smiles, mol, fingerprint_methods))

    with open("../../../../Data/hmdb/hmdb_metabolites/metabolites-2021-06-20", "r", encoding="utf8") as hmdbs, \
         open("../data/mol_key_test/hmdb_fingerprints.csv", "w", newline="") as hmdb_fingerprint_matrix, \
         open("../data/mol_key_test/hmdb_meta.csv", "w", newline="") as hmdb_meta:

        hmdb_csv = csv.reader(hmdbs)
        hmdb_matrix_csv = csv.writer(hmdb_fingerprint_matrix)
        hmdb_meta_csv = csv.writer(hmdb_meta)

        # 0: HMDB_ID, 2: smiles, 6: mono mass
        for row in hmdb_csv:

            if row[0] == "HMDB_ID":
                continue

            smiles = smiles_qc(row[2])

            if smiles is None:
                continue

            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None

            hmdb_meta_csv.writerow([row[0], smiles])
            hmdb_matrix_csv.writerow(smiles_to_matrix(smiles, mol, fingerprint_methods))
