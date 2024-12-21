# -*- encoding: utf-8 -*-
"""
@Introduce  :
@File       : demo1.py
@Author     : ryrl
@email      : ryrl970311@gmail.com
@Time       : 2024/10/1 14:44
@Describe   :
"""
import time
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import matplotlib.pylab as plt

import torch.nn as nn
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from torch_geometric.utils import add_self_loops

from model.model import GATAutoEncoderDecoder
from model.utils import NetworkEnhance
from model import utils
from model.scGAC_ import getGraph, getNeMatrix
from torch_geometric.data import Data, ClusterData, ClusterLoader


adata = sc.read_h5ad('/public/workspace/ryrl/projects/classmates/ryrl/GraphCausal/10X/GEMXHumanPBMC.h5ad')
genes = adata.var_names.values
cells = adata.obs_names.values
features = adata.to_df().values

sc.read_loom()

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

data = Data(x=X, edge_index=edge_index, edge_weight=edge_weight,)

edge_index_, edge_weight_ = add_self_loops(
    edge_index=data.edge_index,
    edge_attr=data.edge_weight,
    num_nodes=data.num_nodes,
    fill_value=1.0
)

data = Data(x=data.x, edge_index=edge_index_, edge_attr=edge_weight_)
data.adj = torch.tensor(data.Adj)
torch.save(obj=data,
           f='/public/workspace/ryrl/projects/classmates/ryrl/GraphCausal/10X/GraphSelfLoops.pt')

data = torch.load('/public/workspace/ryrl/projects/classmates/ryrl/GraphCausal/10X/GraphSelfLoops.pt')
cluster_data = ClusterData(
    data=data,
    num_parts=10,
    recursive=False,
    save_dir='/public/workspace/ryrl/projects/classmates/ryrl/GraphCausal/10X/cluster'
)
model = GATAutoEncoderDecoder(
    in_channels=data.x.shape[1],
    hidden_channels=512,
    out_channels=256,

)
optimizer = torch.optim.Adam(params=model.parameters(), lr=.001, weight_decay=5e-4)
loss_fn = torch.nn.MSELoss()

num_epochs = 500
lst = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    x, z = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight)
    loss = loss_fn(x, data.x)
    loss.backward()
    optimizer.step()

    lst.append((epoch, loss))
    if epoch % 10 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss.item(): .4f}')


torch.save(model.state_dict(),
           '/public/workspace/ryrl/projects/classmates/ryrl/GraphCausal/10X/model/model_state.pth')
df = pd.DataFrame(lst, columns=['Epoch', 'Loss'])
df = df.assign(Loss=df['Loss'].apply(lambda x: x.item()))

df.to_csv(
    '/public/workspace/ryrl/projects/classmates/ryrl/GraphCausal/10X/train_loss.txt', sep='\t', index=None
)

# df.head()
sns.lineplot(data=df.drop(index=1), x='Epoch', y='Loss')
plt.show()








