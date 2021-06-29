import os
import numpy as np
from math import floor
from .mol_key_test import MolKeyDataset


def load_dataset(dataset_name, data_path):
    """Loads the dataset."""

    implemented_datasets = ("mol_key_test",)
    assert dataset_name in implemented_datasets

    if dataset_name == 'mol_key_test':

        self_data_csv = os.path.join(data_path, "hmdb_fingerprints_processed.csv")
        self_labels_csv = os.path.join(data_path, "hmdb_meta.csv")

        other_data_csv = os.path.join(data_path, "zinc_fingerprints_processed.csv")
        other_labels_csv = os.path.join(data_path, "zinc_meta.csv")

        if not os.path.exists(self_data_csv):
            raise FileNotFoundError
        elif not os.path.exists(self_labels_csv):
            raise FileNotFoundError
        if not os.path.exists(other_data_csv):
            raise FileNotFoundError
        elif not os.path.exists(other_labels_csv):
            raise FileNotFoundError

        self_x_data = np.loadtxt(self_data_csv, delimiter=",", comments=None)
        self_labels = np.loadtxt(self_labels_csv, delimiter=",", dtype=str, comments=None)

        other_x_data = np.loadtxt(other_data_csv, delimiter=",", comments=None)
        other_labels = np.loadtxt(other_labels_csv, delimiter=",", dtype=str, comments=None)

        np.random.seed(1)

        def unison_shuffled_copies(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        self_x_data, self_labels = unison_shuffled_copies(self_x_data, self_labels)
        other_x_data, other_labels = unison_shuffled_copies(other_x_data, other_labels)

        num_rows, num_cols = self_x_data.shape
        train_test_split_index = floor(num_rows * 0.8)
        train_index = range(0, train_test_split_index)

        x_data = np.concatenate([self_x_data, other_x_data])
        labels = np.concatenate([self_labels, other_labels])
        num_rows, num_cols = x_data.shape
        test_index = range(train_test_split_index, num_rows)

        dataset = MolKeyDataset(root=data_path, train_idx=train_index, test_idx=test_index, data=x_data, labels=labels[:,2])

        test_x_data = self_x_data[train_index]

        num_rows, num_cols = test_x_data.shape
        prev_index = 0
        fold_datasets = []

        for i in range(5):
            if i == 4:
                fold_split_index = num_rows
            else:
                fold_split_index = floor(num_rows * ((i + 1) / 5))

            fold_validation_index = range(prev_index, fold_split_index)
            fold_train_index = [index for index in range(num_rows) if index not in fold_validation_index]

            fold_datasets.append(MolKeyDataset(root=data_path, train_idx=fold_train_index, test_idx=fold_validation_index, data=test_x_data, labels=None))

            prev_index = fold_split_index

    return dataset, labels, fold_datasets
