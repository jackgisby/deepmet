import numpy as np

from deepmet.base.loadable_dataset import LoadableDataset
from torch.utils.data import Dataset, Subset


class LoadableMolKeyDataset(LoadableDataset):

    def __init__(self, root: str, train_idx=None, test_idx=None, data=None, labels=None):
        super().__init__(root)

        self.train_set = MolKeyDataset(root=self.root, train=True, data=data, labels=labels)

        if train_idx is not None:
          self.train_set = Subset(self.train_set, train_idx)

        self.test_set = MolKeyDataset(root=self.root, train=False, data=data, labels=labels)

        if test_idx is not None:
            self.test_set = Subset(self.test_set, test_idx)


class MolKeyDataset(Dataset):

    def __init__(self, root, train, data=None, labels=None):
        super(MolKeyDataset, self).__init__()

        self.train = train

        self.data = data

        if labels is None:
            self.labels = np.zeros(self.data.shape[0])
        else:
            self.labels = labels.astype(float)

    # This is used to return a single datapoint. A requirement from pytorch
    def __getitem__(self, index):
        return self.data[index], self.labels[index], index

    # For Pytorch to know how many datapoints are in the dataset
    def __len__(self):
        return len(self.data)
