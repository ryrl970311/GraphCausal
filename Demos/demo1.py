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
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.decomposition import PCA

from model.scGAC_utils import getGraph
from model.utils import NetworkEnhancer
from model.model import GraphAttention, ClusteringLayer


adata = sc.read_h5ad('/public/workspace/ryrl/projects/classmates/ryrl/GraphCausal/10X/GEMXHumanPBMC.h5ad')
genes = adata.var_names.values
cells = adata.obs_names.values
features = adata.to_df().values

nclusters = 12

N = len(cells)
avg_N = N // nclusters
K = avg_N // 10
K = min(K, 20)
K = max(K, 6)


coef = np.corrcoef(features, rowvar=True)
enhance = NetworkEnhancer(W_in=coef, K=15)

