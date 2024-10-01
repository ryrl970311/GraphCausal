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
        self.encoder = GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=nheads, concat=True)
        self.embedding = nn.Linear(in_features=hidden_dim * nheads, out_features=embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, edge_index, edge_weight):
        """

        :param x:
        :param edge_index:
        :param edge_weight:
        :return:
        """
        x = self.encoder(x, edge_index, edge_weight)
        x = F.relu(x)
        x = z = self.embedding(x)
        x = self.decoder(x)
        return x, z

    def encode(self, x, edge_index, edge_weight):
        """

        :param x:
        :param edge_index:
        :param edge_weight:
        :return:
        """
        x = self.encoder(x, edge_index, edge_weight)
        x = self.embedding(x)
        return x

    def decode(self, x):
        return self.decoder(x)

    def loss(self, x, x_recon):
        """
        :param x: [num_cells, num_genes]
        :param x_recon: [num_cells, num_genes]
        :return:
        """
        loss = F.mse_loss(x, x_recon)
        return loss

    def save_model(self, path):
        """

        :param path:
        :return:
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """

        :param path:
        :return:
        """
        self.load_state_dict(torch.load(path))
