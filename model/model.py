# -*- encoding: utf-8 -*-
"""
@Introduce:
@File: model.py
@Author: ryrl
@email: ryrl970311@gmail.com
@Time: 2024/10/2 10:45
@Describe:
"""
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch_geometric.nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self):
        super().__init__()
    pass


class GATAutoEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, nheads: int = 8, dropout: float = .3):
        """

        Parameters
        ----------
        input_dim
        hidden_dim
        embedding_dim
        nheads
        dropout
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

        Parameters
        ----------
        x
        edge_index
        edge_weight

        Returns
        -------

        """
        x = self.encoder(x, edge_index, edge_weight)
        x = F.relu(x)

        x = self.encoder_(x, edge_index, edge_weight)
        x = F.relu(x)

        x = z = self.embedding(x)
        x = self.embedding_dropout(x)
        x = self.decoder(x)
        return x, z



