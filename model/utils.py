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

from typing import Union

from anndata import AnnData
from numpy import ndarray
from numpy.linalg import eig
from scipy.sparse import diags
from numpy.linalg import eig, inv

from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityMatrix:

    def __init__(self, X: Union[pd.DataFrame, ndarray]):
        """

        Parameters
        ----------
        X : data to calculate the similarity matrix
        """
        super().__init__()
        self.X = X.values if isinstance(X, pd.DataFrame) else X \
            if isinstance(X, ndarray) else ValueError('X must be dataframe or ndarray!')

    def similiarty_matrix(self, method: str = 'cos', func = None):
        """

        Parameters
        ----------
        method : choose from 'cosine_similarity', 'pearson', 'spearman' and 'custom',default: cosine_similarity
        func : must combine with method = 'custom' and supply the function you want to calculate

        Returns
        -------
        similarities : ndarray or sparse matrix of shape (n_samples_X, n_samples_Y)
        Returns the cosine similarity between samples in X and Y.
        """
        if method == 'cos':
            return cosine_similarity(self.X)
        elif method == 'pearson':
            return np.corrcoef(self.X)
        elif method == 'spearman':
            return spearmanr(self.X)
        elif func and method == 'custom':
            return func(self.X)
        else:
            raise ValueError(f'method or func must be supplied.')


class NetworkEnhancer:
    def __init__(self, W_in: ndarray, order: float = 2, K: int = None, alpha: float = 0.9):
        """

        Parameters
        ----------
        W_in: Input Matrix, N x N
        order: Determine the degree of diffusion
        K: Number of neighbours
        alpha: Regularization parameters
        """
        self.W_in = W_in.astype(np.float64)
        self.order = order
        self.alpha = alpha
        self.N = self.W_in.shape[0]

        if K is None:
            self.K = min(20, int(np.ceil(self.N / 10)))
        else:
            self.K = K

        # Preprocess the matrix
        self.W_in_ = self.W_in * (1 - np.eye(self.N))
        self.zeroindex = np.where(np.sum(np.abs(self.W_in_), axis=0) > 0)[0]
        self.W = self.W_in[np.ix_(self.zeroindex, self.zeroindex)]

        self.degree_sum = np.sum(np.abs(self.W), axis=0)

        # 计算初始的 W 矩阵
        self.W = self.network_enhance_dn(W=self.W, typ='ave')
        self.W = (self.W + self.W.T) / 2


        if len(np.unique(self.W)) == 2:
            self.P = self.W.copy()
        else:
            self.P = self.dominateset(
                W=np.abs(self.W),
                NR_OF_KNN=min(self.K, self.W.shape[0] - 1)) * np.sign(self.W
            )

        # adjust P matrix
        self.P = self.P + np.eye(self.P.shape[0]) + np.diag(np.sum(np.abs(self.P.T), axis=0))

        # apply TransitionFields
        self.P = self.transitionfields(self.P)

        self.eigenvalues, self.eigenvectors = eig(self.P)
        self.eigenvalues = np.real(self.eigenvalues) - np.finfo(float).eps

    @staticmethod
    def network_enhance_dn(W: ndarray, typ: str = 'ave'):
        """
        Normalized network W
        Parameters
        ----------
        W: processed W matrix
        typ: type of normalized, 'ave' or 'gph'

        Returns
        -------
        A ndarray
        """
        w = W * max(W.shape)
        D = np.sum(np.abs(w), axis=1) + np.finfo(float).eps

        if typ == 'ave':
            D_inv = 1.0 / D
            D_diag = np.diag(D_inv)
            wn = D_diag @ w
        elif typ == 'gph':
            D_inv_sqrt = 1.0 / np.sqrt(D)
            D_diag = np.diag(D_inv_sqrt)
            wn = D_diag @ (w @ D_diag)
        else:
            raise ValueError("Unknown type: " + str(typ))

        return wn

    @staticmethod
    def dominateset(W: ndarray, NR_OF_KNN: int):
        r"""
        Keep the first NR_OF_KNN elements with the largest absolute value in each row,
        set the remaining elements to zero, and symmetrize the result.

        Parameters
        ----------
        W: numpy.ndarray, np.abs() transformed normalized matrix
        NR_OF_KNN: int, the number of nearest neighbors to retain for each row.

        Returns
        -------
        PNN_matrix: numpy.ndarray, symmetrized matrix.
        """
        A = -np.sort(-W, axis=1)
        B = np.argsort(-W, axis=1)
        res = A[:, :NR_OF_KNN]
        loc = B[:, :NR_OF_KNN]
        N = W.shape[0]
        inds = np.repeat(np.arange(N).reshape(N, 1), NR_OF_KNN, axis=1)
        PNN_matrix1 = np.zeros_like(W)
        PNN_matrix1[inds.flatten(), loc.flatten()] = res.flatten()
        PNN_matrix = (PNN_matrix1 + PNN_matrix1.T) / 2
        return PNN_matrix

    def transitionfields(self, W):
        """
        Perform transformation field operations on matrix W.

        Parameters
        ----------
        W: numpy.ndarray, adjusted matrix

        Returns
        -------
        W: numpy.ndarray, processed W。
        """
        zeroindex = np.where(np.sum(W, axis=1) == 0)[0]
        W = W * max(W.shape)
        W = self.network_enhance_dn(W, 'ave')
        w = np.sqrt(np.sum(np.abs(W), axis=0) + np.finfo(float).eps)
        W = W / w
        W = W @ W.T
        # Wnew = W.copy()
        W[zeroindex, :] = 0
        W[:, zeroindex] = 0
        return W

    def enhance(self):
        """
        Enhance the input network
        Returns
        -------
         W_out: numpy.ndarray, Enhanced W_in。
        """
        # 修改特征值
        d = (1 - self.alpha) * self.eigenvalues / (1 - self.alpha * self.eigenvalues ** self.order)
        D_matrix = np.diag(np.real(d))

        # 重构 W 矩阵
        W = self.eigenvectors @ D_matrix @ inv(self.eigenvectors)
        W = W * (1 - np.eye(W.shape[0]))
        diag_W = 1 - np.diag(W)
        W = W / diag_W[:, np.newaxis]
        D_diag = np.diag(self.degree_sum)
        W = D_diag @ W
        W[W < 0] = 0
        W = (W + W.T) / 2

        # 构建输出矩阵
        W_out = np.zeros_like(self.W_in)
        W_out[np.ix_(self.zeroindex, self.zeroindex)] = W
        return W_out





class AnnData2Graph(object):

    def __init__(self, adata: AnnData):
        super().__init__()
        self.adata = adata
        self.X = adata.to_df().values


    def adjacency_matrix(self, similarity: Union[np.array, pd.DataFrame], k: int = 15):
        """

        :param similarity:
        :param k:
        :return:
        """
        if isinstance(similarity, pd.DataFrame):
            similarity = similarity.values
        indices = np.argsort(-similarity, axis=1)[:, 1:k+1]
        adjacency_matrix = np.zeros_like(similarity)
        adjacency_matrix[np.arange(similarity.shape[0])[:, np.newaxis], indices] = 1
        return adjacency_matrix






