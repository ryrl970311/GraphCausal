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
import torch_geometric.nn


class AnnData2Graph(object):
    pass


class GATAutoEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, nheads: int = 8, dropout: float = .3):
        """

        :param input_dim: the number of features, e.g.: data.x.shape
        :param hidden_dim:
        :param embedding_dim:
        :param nheads:
        :param dropout:
        """
        super().__init__()
        self.encoder = torch_geometric.nn.GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=nheads,
            concat=True,
            dropout=.3
        )

        self.encoder_ = torch_geometric.nn.GATConv(
            in_channels=hidden_dim * nheads,
            out_channels=hidden_dim,
            heads=nheads,
            concat=True,
            dropout=.3
        )

        self.embedding = nn.Linear(in_features=hidden_dim * nheads, out_features=embedding_dim)

        self.embedding_dropout = nn.Dropout(p=dropout)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=input_dim // 2),
            nn.ReLU(),
            nn.Linear(in_features=input_dim // 2, out_features=input_dim)
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

        x = self.encoder_(x, edge_index, edge_weight)
        x = F.relu(x)

        x = z = self.embedding(x)
        x = self.embedding_dropout(x)
        x = self.decoder(x)
        return x, z


