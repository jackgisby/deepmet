import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class BasicMultilayer(BaseNet):

    def __init__(self, rep_dim, in_features):
        super().__init__()

        self.rep_dim = rep_dim  # autoencoder representation size
        self.in_features = in_features  # in features

        # encoder
        self.deep1 = nn.Linear(self.in_features, 500)
        self.deep2 = nn.Linear(500, 250)
        self.fc_output = nn.Linear(250, self.rep_dim, bias=False)

        # decoder
        self.deep4 = nn.Linear(self.rep_dim, 250)
        self.deep5 = nn.Linear(250, 500)
        self.deep6 = nn.Linear(500, self.in_features)

    def forward(self, x):

        x = F.relu(self.deep1(x))
        x = F.relu(self.deep2(x))
        x = F.relu(self.fc_output(x))

        return x


class BasicMultilayerAutoencoder(BasicMultilayer):

    def __init__(self, rep_dim, in_features):
        super().__init__(rep_dim=rep_dim, in_features=in_features)

    def forward(self, x):

        x = F.relu(self.deep1(x))
        x = F.relu(self.deep2(x))
        x = F.relu(self.fc_output(x))

        x = F.relu(self.deep4(x))
        x = F.relu(self.deep5(x))
        x = F.sigmoid(self.deep6(x))

        return x
