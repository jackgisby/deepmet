import os
import numpy as np
from math import floor
from .mol_key_test import MolKeyDataset


def load_dataset(dataset_name, data_path):
    """Loads the dataset."""

    implemented_datasets = ("mol_key_test",)
    assert dataset_name in implemented_datasets

    dataset = None
    labels = None

    if dataset_name == 'mol_key_test':

        data_csv = os.path.join(data_path, "fingerprints_processed.csv")
        labels_csv = os.path.join(data_path, "meta.csv")


        if not os.path.exists(data_csv):
            raise FileNotFoundError
        elif not os.path.exists(labels_csv):
            raise FileNotFoundError

        x_data = np.loadtxt(data_csv, delimiter=",")
        labels = np.loadtxt(labels_csv, delimiter=",", dtype=str)
        np.random.seed(1)

        def unison_shuffled_copies(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        x_data, labels = unison_shuffled_copies(x_data, labels)

        num_rows, num_cols = x_data.shape
        train_test_split_index = floor(num_rows * 0.8)

        train_index = range(0, train_test_split_index)
        test_index = range(train_test_split_index, num_rows)

        dataset = MolKeyDataset(root=data_path, train_idx=train_index, test_idx=test_index, data=x_data)

    return dataset, labels
