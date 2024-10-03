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

class GraphAttention(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 nheads: int = 8,
                 nheads_reduction: str = 'concat',
                 dropout: float = .5,
                 activation = F.relu,
                 bias: bool = True
                 ):
        r"""
        Single-cell graph attentional clustering--scGAC, GAT layer pytorch implementation.

        References: Yi Cheng, Xiuli Ma, scGAC: a graph attentional architecture for clustering single-cell RNA-seq data,
        Bioinformatics, Volume 38, Issue 8, March 2022, Pages 2187–2193, https://doi.org/10.1093/bioinformatics/btac099.


        Parameters
        ----------
        in_features
        out_features
        nheads
        nheads_reduction
        dropout
        activation
        bias
        """
        super().__init__()

        if nheads_reduction not in ['concat', 'average']:
            raise ValueError('nhead_reduction must in ["concat", "average"]')

        self.in_features = in_features,
        self.out_features = out_features,
        self.F = out_features,

        self.nheads = nheads,
        self.nheads_reduction = nheads_reduction,
        self.dropout = dropout,
        self.activation = activation,
        self.bias = bias

        self.kernels = nn.ParameterList()
        if self.bias:
            self.biases = nn.ParameterList()
        else:
            self.bias = None

        self.attn_kernels = nn.ParameterList()
        for head in range(self.nheads):
            kernel = nn.Parameter(torch.Tensor(in_features, out_features))
            self.kernels.append(kernel)

            if self.bias:
                bias = nn.Parameter(torch.Tensor(out_features, 1))
                self.biases.append(bias)

            # Attention weight
            attn_kernel_self = nn.Parameter(torch.Tensor(out_features, 1))
            attn_kernel_neighs = nn.Parameter(torch.Tensor(out_features, 1))
            self.attn_kernels.append(nn.ParameterList([attn_kernel_self, attn_kernel_neighs]))


        # initial parameters
        self.reset_parameters()

        if nheads_reduction == 'concat':
            self.output_dim = self.F * self.nheads
        else:
            self.output_dim = self.F

    def reset_parameters(self):
        # initial weight and bias
        for kernel in self.kernels:
            nn.init.xavier_uniform_(kernel)

        if self.bias:
            for bias in self.biases:
                nn.init.zeros_(bias)

        for attn_kernel in self.attn_kernels:
            nn.init.xavier_uniform_(attn_kernel[0])
            nn.init.xavier_uniform_(attn_kernel[1])

    def forward(self, x, adj):
        outputs = []
        for head in range(self.nheads):
            kernel = self.kernels[head]
            attn_kernel = self.attn_kernels[head]

            # Linear transformation
            features = torch.mm(input=x, mat2=kernel)

            # Attention mechanism
            attn4self = torch.mm(input=features, mat2=attn_kernel[0])
            attn4neighs = torch.mm(input=features, mat2=attn_kernel[1])

            # Calculate Attention coefficient
            dense = torch.exp(-torch.pow(attn4self - attn4neighs.t(), 2))

            mask_pos = (adj > 0).float()
            dense = dense * mask_pos
            dense = dense / (dense.sum(dim=1, keepdim=True) + np.finfo(float).eps)

            # Dropout
            dropout_attn = F.dropout(input=dense, p=self.dropout, training=self.training)
            dropout_feat = F.dropout(input=features, p=self.dropout, training=self.training)

            # Aggregate neighbors' message
            node_features = torch.mm(input=dropout_attn, mat2=dropout_feat)

            if self.bias:
                node_features = node_features + self.biases[head]

            outputs.append(node_features)


        if self.nheads_reduction == 'concat':
            output = torch.cat(tensors=outputs, dim=-1)
        else:
            output = torch.mean(torch.stack(tensors=outputs), dim=0)
        return self.activation(output)


class ClusteringLayer(nn.Module):

    def __init__(self,
                 nclusters: int,
                 model: str = 'q2',
                 threshold: float = .1,
                 weights = None,
                 alpha: float = 1.0):
        """
        Single-cell graph attentional clustering--scGAC, GAT cluster layer pytorch implementation.

        References: Yi Cheng, Xiuli Ma, scGAC: a graph attentional architecture for clustering single-cell RNA-seq data,
        Bioinformatics, Volume 38, Issue 8, March 2022, Pages 2187–2193, https://doi.org/10.1093/bioinformatics/btac099.

        Parameters
        ----------
        nclusters: The number of cluster
        model:
        threshold:
        weights: initial clusters' center
        alpha: Hyper parameter
        """
        super().__init__()

        self.nclusters: int = nclusters,
        self.threshold: float = threshold,
        self.model: str = model,
        self.weights = weights,
        self.alpha: float = alpha,

        self.clusters = nn.Parameter(self.weights) if self.weights is not None else None

    def forward(self, x: Tensor):
        """

        Parameters
        ----------
        x:

        Returns
        -------

        """

        if self.clusters is None:
            idx: Tensor = torch.randperm(x.size(0))[:self.nclusters]
            initial_clusters: Tensor = x[idx].detach()  # Note Avoid the update of these data
            self.clusters = nn.Parameter(initial_clusters)

        # Calculate the square of the **Euclidean distance**
        x_: Tensor = x.unsqueeze(dim=1)  # (batch size, 1, feature dim)
        cluster_: Tensor = self.clusters.unsqueeze(dim=0)  # (1, nclusters, feature dim)
        distances: Tensor = torch.sum((x_ - cluster_) ** 2, dim=2)  # (batch size, nclusters)

        q = 1.0 / (1.0 + distances / self.alpha)
        q = q / torch.sum(q, dim=1, keepdim=True)  # Normalization for each sample

        if self.model == 'q2':
            q_sum: Tensor = torch.sum(input=q, dim=0, keepdim=True)  # (1, nclusters)
            q_: Tensor = (q ** 2) / q_sum
            q_: Tensor = q_ / torch.sum(q_, dim=1, keepdim=True)  # Standardized
        else:
            q_: Tensor = q + np.finfo(float).eps

        qidx = torch.argmax(input=q_, dim=1) # Find the max value for each row
        q_mask = F.one_hot(tensor=qidx, num_classes=self.nclusters).float()
        q_ = q_mask * q_

        q_ = F.relu(input=q_ - self.threshold)
        q_ += torch.sign(q_) * self.threshold

        numerator: Tensor = torch.mm(input=q_.transpose(dim0=0, dim1=1), mat2=x)  # (nclusters, feature dim)
        denominator: Tensor = torch.sum(q_, dim=0, keepdim=True).transpose(dim0=0, dim1=1)  # (nclusters, 1)
        denominator: Tensor = denominator + np.finfo(float).eps
        self.clusters.data = numerator / denominator
        return q

