import os
import numpy as np
from math import floor
from .mol_key_test import MolKeyDataset


def load_dataset(dataset_name, data_path, test_data_prefix=None):
    """Loads the dataset."""

    implemented_datasets = ("mol_key_test",)
    assert dataset_name in implemented_datasets

    if dataset_name == 'mol_key_test':

        self_data_csv = os.path.join(data_path, "hmdb_fingerprints_processed.csv")
        self_labels_csv = os.path.join(data_path, "hmdb_meta.csv")

        if not os.path.exists(self_data_csv):
            raise FileNotFoundError
        elif not os.path.exists(self_labels_csv):
            raise FileNotFoundError

        self_x_data = np.loadtxt(self_data_csv, delimiter=",", comments=None)
        self_labels = np.loadtxt(self_labels_csv, delimiter=",", dtype=str, comments=None)

        def unison_shuffled_copies(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        self_x_data, self_labels = unison_shuffled_copies(self_x_data, self_labels)

        if test_data_prefix is not None:
            other_data_csv = os.path.join(data_path, test_data_prefix + "_fingerprints_processed.csv")
            other_labels_csv = os.path.join(data_path, test_data_prefix + "_meta.csv")

            if not os.path.exists(other_data_csv):
                raise FileNotFoundError
            elif not os.path.exists(other_labels_csv):
                raise FileNotFoundError

            other_x_data = np.loadtxt(other_data_csv, delimiter=",", comments=None)
            other_labels = np.loadtxt(other_labels_csv, delimiter=",", dtype=str, comments=None)

            other_x_data, other_labels = unison_shuffled_copies(other_x_data, other_labels)

        num_rows, num_cols = self_x_data.shape
        train_val_split_index = floor(num_rows * 0.8)
        train_index = range(0, train_val_split_index)
        val_test_split_index = floor(num_rows * 0.9)
        val_index = range(train_val_split_index, val_test_split_index)

        if test_data_prefix is not None:
            x_data = np.concatenate([self_x_data, other_x_data])
            labels = np.concatenate([self_labels, other_labels])

        num_rows, num_cols = x_data.shape
        test_index = range(val_test_split_index, num_rows)

        assert len(train_index) + len(val_index) + len(test_index) == num_rows
        for i, a in enumerate((train_index, val_index, test_index)):
            for j, b in enumerate((train_index, val_index, test_index)):
                if i != j:
                    assert all([a_item not in b for a_item in a])

        full_dataset = MolKeyDataset(root=data_path, train_idx=train_index, test_idx=test_index, data=x_data, labels=labels[:,2])
        val_dataset = MolKeyDataset(root=data_path, train_idx=train_index, test_idx=val_index, data=x_data, labels=labels[:,2])


    return full_dataset, labels, val_dataset
