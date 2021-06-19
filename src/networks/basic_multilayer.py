import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class BasicMultilayer(BaseNet):

    def __init__(self):
        super().__init__()

        self.in_features = 2048  # in features
        self.rep_dim = 50  # autoencoder representation size

        # encoder
        self.deep1 = nn.Linear(self.in_features, 250)
        self.deep2 = nn.Linear(250, 100)
        self.fc_output = nn.Linear(100, self.rep_dim)

        # decoder
        self.deep4 = nn.Linear(self.rep_dim, 100)
        self.deep5 = nn.Linear(100, 250)
        self.deep6 = nn.Linear(250, self.in_features)

    def forward(self, x):

        x = F.relu(self.deep1(x))
        x = F.relu(self.deep2(x))
        x = F.relu(self.fc_output(x))

        return x


class BasicMultilayerAutoencoder(BasicMultilayer):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        x = F.relu(self.deep1(x))
        x = F.relu(self.deep2(x))
        x = F.relu(self.fc_output(x))

        x = F.relu(self.deep4(x))
        x = F.relu(self.deep5(x))
        x = F.sigmoid(self.deep6(x))

        return x
