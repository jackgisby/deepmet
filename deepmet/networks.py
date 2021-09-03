#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2021 Ralf Weber
#
# This file is part of DeepMet.
#
# DeepMet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepMet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepMet.  If not, see <https://www.gnu.org/licenses/>.
#
# This file incorporates work covered by the following copyright and
# permission notice:
#
#   Copyright (c) 2018 Lukas Ruff
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepmet.base import BaseNet


class BasicMultilayer(BaseNet):
    """
    A basic fully connected neural network with three layers.

    :param rep_dim: The number of dimensions of the representation layer.

    :param in_features: The number of features within the input dataset.
    """

    def __init__(self, rep_dim, in_features):
        super().__init__()

        self.rep_dim = rep_dim
        self.in_features = in_features

        self.deep1 = nn.Linear(self.in_features, 500)
        self.deep2 = nn.Linear(500, 250)
        self.fc_output = nn.Linear(250, self.rep_dim, bias=False)

    def forward(self, x):
        """ Create and return a feed forward neural network. """

        x = F.relu(self.deep1(x))
        x = F.relu(self.deep2(x))
        x = F.relu(self.fc_output(x))

        return x


def build_network(net_name, rep_dim, in_features):
    """
    Builds the neural network.

    :param net_name: May be one of "basic_multilayer" or "cocrystal_transformer", representing the
        :py:meth:`deepmet.networks.BasicMultilayer` and :py:meth:`deepmet.networks.CocrystalTransformer` networks,
        respectively.

    :param in_features: The number of features within the input dataset.

    :return: The network class.
    """

    implemented_networks = ("basic_multilayer", "cocrystal_transformer")
    assert net_name in implemented_networks

    net = None

    if net_name == "basic_multilayer":
        net = BasicMultilayer(rep_dim=rep_dim, in_features=in_features)

    elif net_name == "cocrystal_transformer":
        net = CocrystalTransformer(rep_dim=rep_dim, in_features=in_features)

    return net


class CocrystalTransformer(BaseNet):
    """
    A transformer network containing three :py:meth:`deepmet.networks.SAB` layers and a :py:meth:`deepmet.networks.PAM`
    layer.

    :param rep_dim: The number of dimensions of the representation layer.

    :param in_features: The number of features within the input dataset.
    """

    def __init__(self, rep_dim, in_features):
        super().__init__()

        self.rep_dim = rep_dim
        self.in_features = in_features

        self.seq = nn.Sequential(SAB(dim_in=self.in_features, dim_out=1000, num_heads=10),
                                 SAB(dim_in=1000, dim_out=500, num_heads=5),
                                 SAB(dim_in=500, dim_out=self.rep_dim, num_heads=10),
                                 PMA(dim=self.rep_dim, num_heads=5, num_seeds=1))

    def forward(self, x):
        """ Create and return a feed forward neural network. """

        x = torch.split(x, self.in_features, dim=1)
        x = torch.stack(x).transpose(0, 1)

        return self.seq(x).squeeze()


class MAB(nn.Module):
    """ MAB module. """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()

        self.dim_V = dim_V
        self.num_heads = num_heads

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)

        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        """ Create and return the module. """

        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + torch.nn.functional.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        return O


class SAB(nn.Module):
    """ SAB module. """

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        """ Create and return the module. """

        return self.mab(X, X)


class PMA(nn.Module):
    """ PMA module. """

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()

        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)

        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        """ Create and return the module. """

        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
