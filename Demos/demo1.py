# -*- encoding: utf-8 -*-
"""
@Introduce  :
@File       : demo1.py
@Author     : ryrl
@email      : ryrl970311@gmail.com
@Time       : 2024/10/1 14:44
@Describe   :
"""
import os
import time
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from model.model import GATscGAC
from model.utils import NetworkEnhance
from model.scGAC_utils import getGraph, getNeMatrix


adata = sc.read_h5ad('/public/workspace/ryrl/projects/classmates/ryrl/GraphCausal/10X/GEMXHumanPBMC.h5ad')
genes = adata.var_names.values
cells = adata.obs_names.values
features = adata.to_df().values


coef = np.corrcoef(features, rowvar=True)

t0 = time.time()
W = getNeMatrix(coef)
t1 = time.time()

df = pd.DataFrame(W, columns=cells, index=cells)
# df.to_parquet('/public/workspace/ryrl/projects/classmates/ryrl/GraphCausal/10X/EnhancedNetworkMatrix.parquet')

adjacency = getGraph(coef_matrix=coef, enhance_matrix=W, K=20, L=0)

# np.save(file='/public/workspace/ryrl/projects/classmates/ryrl/GraphCausal/10X/Adjacency.npy', arr=adjacency)

adjacency = np.load('/public/workspace/ryrl/projects/classmates/ryrl/GraphCausal/10X/Adjacency.npy')

coef_ = pd.read_parquet(
'/public/workspace/ryrl/projects/classmates/ryrl/GraphCausal/10X/EnhancedNetworkMatrix.parquet'
).values

pca = PCA(n_components=512)
features = pca.fit_transform(features)

scaler = StandardScaler()
features = scaler.fit_transform(features)

edge_index = np.array(np.nonzero(adjacency))
edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_weight = coef_[edge_index[0], edge_index[1]]

X = torch.tensor(adata.to_df().values)
edge_weight = torch.tensor(edge_weight, dtype=torch.float)

data = Data(x=X, edge_index=edge_index, edge_weight=edge_weight)

edge_index_, edge_weight_ = add_self_loops(
    edge_index=data.edge_index,
    edge_attr=data.edge_weight,
    num_nodes=data.num_nodes,
    fill_value=1.0
)

data = Data(x=data.x, edge_index=edge_index_, edge_attr=edge_weight_)
torch.save(obj=data,
           f='/public/workspace/ryrl/projects/classmates/ryrl/GraphCausal/10X/GraphSelfLoops.pt')