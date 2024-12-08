# -*- encoding: utf-8 -*-
"""
@Introduce:
@File: model.py
@Author: ryrl
@email: ryrl970311@gmail.com
@Time: 2024/10/2 10:45
@Describe:
"""
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch_geometric.nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

class GraphAttentionLayer(nn.Module):

    def __init__(self):
        super().__init__()
    pass

class EncoderLayer(nn.Module):

    def __init__(
            self, 
            in_channels: Union[int, tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True
    ) -> None:
        super().__init__()
        self.encoder = torch_geometric.nn.GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim,
            fill_value=fill_value,
            bias=bias
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
        x = F.leaky_relu(x)
        return x

class Embedding(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: float = 0.3,
            activation: str = 'relu',
            bias: bool = False,
            use_batch_norm: bool = True
    ):
        super().__init__()
        self.embedding = nn.Linear(in_features=in_channels, out_features=out_channels, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU()
        self.batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_features=out_channels, affine=True)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.bias is not None:
            nn.init.zeros_(self.embedding.bias)
        
    def forward(self, x: Tensor):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        x = z = self.embedding(x)
        if self.batch_norm: x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x, z
    
class Decoder(nn.Module):

    def __init__(self, embedding_dim, out_dim, activitation: str = 'relu') -> None:
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=out_dim),
            nn.ReLU() if activitation == 'relu' else nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=out_dim),

            nn.Linear(in_features=out_dim, out_features=out_dim // 2),
            nn.ReLU() if activitation == 'relu' else nn.LeakyReLU(),

            nn.Linear(in_features=out_dim // 2, out_features=out_dim),
        )
    
    def forward(self, x: Tensor):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        return self.decoder(x)
        
class GATAutoEncoderDecoder(nn.Module):

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            heads: int = 8,
            dropout: float = 0.3,
            concat: bool = True,
            negative_slope: float = 0.2,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True
    ):
        """

        Parameters
        ----------
        in_channels
        hidden_channels
        out_channels
        heads
        dropout
        concat
        negative_slope
        add_self_loops
        edge_dim
        fill_value
        bias
        """
        super().__init__()
        self.encoder1 = EncoderLayer(
            in_channels=in_channels, 
            out_channels=hidden_channels, 
            heads=heads, 
            concat=concat,
            dropout=dropout,
            edge_dim=edge_dim,
            fill_value=fill_value,
            bias=bias,
            negative_slope=negative_slope,
            add_self_loops=add_self_loops,
        )

        self.encoder2 = EncoderLayer(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels // 2,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim,
            fill_value=fill_value,
            bias=bias
        )

        self.encoder3 = EncoderLayer(
            in_channels=hidden_channels // 2 * heads,
            out_channels=hidden_channels * 2,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim,
            fill_value=fill_value,
            bias=bias
        )

        self.encoder4 = EncoderLayer(
            in_channels=hidden_channels * 2 * heads,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim,
            fill_value=fill_value,
            bias=bias
        )

        self.embedding = Embedding(
            in_channels=out_channels * heads,
            out_channels=out_channels,
            dropout=dropout,
            activation='relu',
            bias=bias,
            use_batch_norm=True
        )

        self.decoder = Decoder(
            embedding_dim=out_channels,
            out_dim=in_channels,
            activitation='relu'
        )  # 重构原始表达数据，学习数据的低纬表示

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
        x = self.encoder1(x, edge_index, edge_weight)
        x = self.encoder2(x, edge_index, edge_weight)
        x = self.encoder3(x, edge_index, edge_weight)
        x = self.encoder4(x, edge_index, edge_weight)

        x, embedding = self.embedding(x)
        x = self.decoder(x)
        return x, embedding



