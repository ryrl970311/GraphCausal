# -*- encoding: utf-8 -*-
"""
@Introduce  :
@File       : utils.py
@Author     : ryrl
@email      : ryrl970311@gmail.com
@Time       : 2024/9/30 21:54
@Describe   :
"""

import numpy as np
import pandas as pd
import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from torch import Tensor
from torch_geometric.nn import GATConv


class GATAutoEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, embedding_dim, nheads: int = 8):
        """

        :param input_dim:
        :param hidden_dim:
        :param embedding_dim:
        :param nheads:
        """
        super().__init__()
        self.encoder = GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=nheads, concat=True, dropout=.3)

        self.embedding = nn.Linear(in_features=hidden_dim * nheads, out_features=embedding_dim)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=input_dim)
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor):
        """

        :param x:
        :param edge_index: [2, n]
        :param edge_weight:(n, )
        :return:
        """
        x = self.encoder(x, edge_index, edge_weight)
        x = F.relu(x)
        x = z = self.embedding(x)
        x = self.decoder(x)
        return x, z


