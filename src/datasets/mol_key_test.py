import numpy as np

from base.loadable_dataset import LoadableDataset
from torch.utils.data import Dataset, Subset


class MolKeyDataset(LoadableDataset):

    def __init__(self, root: str, train_idx=None, test_idx=None, data=None):
        super().__init__(root)

        self.train_set = MolKey(root=self.root, train=True, data=data)

        if train_idx is not None:
          self.train_set = Subset(self.train_set, train_idx)

        self.test_set = MolKey(root=self.root, train=False, data=data)

        if test_idx is not None:
            self.test_set = Subset(self.test_set, test_idx)


class MolKey(Dataset):

    def __init__(self, root, train, data=None):
        super(MolKey, self).__init__()

        self.train = train

        self.data = data

        self.labels = np.zeros(self.data.shape[0])

    # This is used to return a single datapoint. A requirement from pytorch
    def __getitem__(self, index):
        return self.data[index], self.labels[index], index

    # For Pytorch to know how many datapoints are in the dataset
    def __len__(self):
        return len(self.data)
