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
from numpy import ndarray, diag
from numpy.linalg import eig, inv
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
# from scGAC_utils import getGraph


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
    """
    A class for enhancing network matrices through normalization, constructing nearest neighbor sets,
    computing transition fields, and applying diffusion-based enhancement.

    Network Enhancement (NE), a novel method for improving the signal-to-noise ratio of a symmetric networks
    and thereby facilitating the downstream network analysis. NE leverages the transitive edges of a network by
    exploiting local structures to strengthen the signal within clusters and weaken the signal between clusters.
    At the same time, NE also alleviates the corrupted links in the network by imposing a normalization that removes
    weak edges by enforcing sparsity. NE is supported by theoretical justifications for its convergence and
    performance in improving community detection outcomes.
    The method provides theoretical guarantees as well as excellent empirical performance on many biological
    problems. The approach can be incorporated into any weighted network analysis pipeline and can lead to improved
    downstream analysis.

    Reference: Wang, B., Pourshafeie, A., Zitnik, M. et al. Network enhancement as a general method to denoise
    weighted biological networks. Nat Commun 9, 3108 (2018). https://doi.org/10.1038/s41467-018-05469-x.

    Methods
    -------
    dn(w, type_)
        Normalizes the input matrix based on the specified type.
    dominateset(aff_matrix, NR_OF_KNN)
        Constructs a symmetric Pairwise Nearest Neighbor (PNN) matrix from an affinity matrix.
    transition_fields(W)
        Computes transition fields based on the input matrix.
    network_enhancement(W_in, order=2, K=None, alpha=0.9)
        Enhances the input network matrix using diffusion and normalization techniques.
    """

    @staticmethod
    def dn(w: np.ndarray, type_: str) -> np.ndarray:
        """
        Normalize the input matrix based on the specified type.

        Parameters
        ----------
        w : np.ndarray
            The input matrix to be normalized.
        type_ : str
            The type of normalization to apply. Must be either 'ave' or 'gph'.

        Returns
        -------
        np.ndarray
            The normalized matrix.

        Raises
        ------
        ValueError
            If the normalization type is neither 'ave' nor 'gph'.
        """
        n = w.shape[0]
        w = w * n
        w = w.astype(float)
        D = np.sum(np.abs(w), axis=1) + np.finfo(float).eps

        if type_ == 'ave':
            D_inv = 1.0 / D
            D_sparse = diags(D_inv)
            wn_sparse = D_sparse.dot(w)
            wn = wn_sparse.toarray()
        elif type_ == 'gph':
            D_inv_sqrt = 1.0 / np.sqrt(D)
            D_sparse = diags(D_inv_sqrt)
            intermediate = D_sparse.dot(w)
            wn_sparse = intermediate.dot(D_sparse)
            wn = wn_sparse.toarray()
        else:
            raise ValueError("Invalid type. Must be 'ave' or 'gph'.")

        return wn

    @staticmethod
    def dominateset(aff_matrix: np.ndarray, NR_OF_KNN: int) -> np.ndarray:
        """
        Constructs a symmetric Pairwise Nearest Neighbor (PNN) matrix from an affinity matrix.

        Parameters
        ----------
        aff_matrix : np.ndarray
            The affinity matrix (2D array).
        NR_OF_KNN : int
            The number of nearest neighbors to consider for each element.

        Returns
        -------
        np.ndarray
            The symmetrized PNN matrix.

        Raises
        ------
        ValueError
            If the affinity matrix is not a 2D array.
        """
        if aff_matrix.ndim != 2:
            raise ValueError("aff_matrix must be a 2D array.")

        num_rows, num_cols = aff_matrix.shape
        sorted_indices = np.argsort(-aff_matrix, axis=1)  # Sort in descending order
        sorted_values = np.sort(-aff_matrix, axis=1) * -1  # Sort descending

        res = sorted_values[:, :NR_OF_KNN]
        loc = sorted_indices[:, :NR_OF_KNN]
        inds = np.repeat(np.arange(num_rows), NR_OF_KNN)
        loc_flat = loc.flatten()
        res_flat = res.flatten()

        PNN_matrix1 = np.zeros_like(aff_matrix, dtype=float)
        PNN_matrix1[inds, loc_flat] = res_flat

        PNN_matrix = (PNN_matrix1 + PNN_matrix1.T) / 2.0

        return PNN_matrix

    @staticmethod
    def transition_fields(W: np.ndarray) -> np.ndarray:
        """
        Computes transition fields based on the input matrix.

        Parameters
        ----------
        W : np.ndarray
            The input matrix.

        Returns
        -------
        np.ndarray
            The transition fields matrix.
        """
        W = np.array(W, dtype=float)
        w = np.sqrt(np.sum(np.abs(W), axis=0) + np.finfo(float).eps)
        w[w == 0] = np.finfo(float).eps  # Prevent division by zero
        W_normalized = W / w[np.newaxis, :]
        W = np.dot(W_normalized, W_normalized.T)

        Wnew = W.copy()
        zeroindex = np.where(np.sum(W, axis=1) == 0)[0]
        if zeroindex.size > 0:
            Wnew[zeroindex, :] = 0
            Wnew[:, zeroindex] = 0

        return Wnew

    def network_enhancement(
        self,
        W_in: np.ndarray,
        order: float = 2,
        K: int = None,
        alpha: float = 0.9
    ) -> np.ndarray:
        """
        Enhances the input network matrix using diffusion and normalization techniques.

        Parameters
        ----------
        W_in : np.ndarray
            The input network matrix of size N x N.
        order : float, optional
            Determines the extent of diffusion. Typical values are 0.5, 1, 2. Default is 2.
        K : int, optional
            The number of nearest neighbors. If None, it defaults to min(20, ceil(N / 10)).
        alpha : float, optional
            The regularization parameter. Default is 0.9.

        Returns
        -------
        np.ndarray
            The enhanced network matrix.

        Raises
        ------
        ValueError
            If input dimensions are inconsistent or parameters are invalid.
        """
        W_in = np.array(W_in, dtype=float)
        N = W_in.shape[0]

        # Set default K if not provided
        if K is None:
            K = min(20, int(np.ceil(N / 10)))

        # Step 1: Remove self-connections by setting diagonal to zero
        W_in1 = W_in * (1 - np.eye(N))

        # Step 2: Identify nodes with at least one non-zero connection
        zeroindex = np.where(np.sum(np.abs(W_in1), axis=1) > 0)[0]

        # Step 3: Extract the submatrix W0 corresponding to non-zero nodes
        W0 = W_in[np.ix_(zeroindex, zeroindex)]

        # Step 4: Normalize W0 using 'ave' type normalization
        W = self.dn(W0, 'ave')

        # Step 5: Symmetrize W
        W = (W + W.T) / 2

        # Step 6: Compute DD as the sum of absolute values of W0 across columns
        DD = np.sum(np.abs(W0), axis=0)

        # Step 7: Determine if W has only two unique values
        if len(np.unique(W)) == 2:
            P = W
        else:
            # Compute effective K to avoid exceeding matrix dimensions
            K_eff = min(K, W.shape[0] - 1)
            P = self.dominateset(np.abs(W).astype(float), K_eff) * np.sign(W)

        # Step 8: Update P by adding identity and diagonal matrix of row sums
        P = P + (np.eye(P.shape[0]) + np.diag(np.sum(np.abs(P), axis=1)))

        # Step 9: Apply Transition Fields
        P = self.transition_fields(P)

        # Step 10: Eigen decomposition of P
        eigenvalues, eigenvectors = np.linalg.eig(P)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        # Step 11: Adjust eigenvalues with regularization parameter alpha
        eps_val = np.finfo(float).eps
        d = eigenvalues - eps_val
        d = (1 - alpha) * d / (1 - alpha * (d ** order))

        # Step 12: Reconstruct W using modified eigenvalues
        D_matrix = np.diag(d)
        W = eigenvectors @ D_matrix @ eigenvectors.T

        # Step 13: Remove self-connections and normalize
        W = (W * (1 - np.eye(W.shape[0]))) / (1 - np.diag(W))[:, np.newaxis]

        # Step 14: Apply DD as a diagonal scaling factor
        D_sparse = diags(DD)
        W = D_sparse.dot(W)

        # Step 15: Set negative values to zero
        W[W < 0] = 0

        # Step 16: Symmetrize W
        W = (W + W.T) / 2

        # Step 17: Initialize W_out and assign the enhanced W to corresponding indices
        W_out = np.zeros_like(W_in)
        W_out[np.ix_(zeroindex, zeroindex)] = W

        return W_out


class AnnData2Graph(object):

    def __init__(self, adata: AnnData):
        super().__init__()
        self.adata = adata
        self.X = adata.to_df().values


    @staticmethod
    def adjacency_matrix(similarity: Union[np.array, pd.DataFrame], k: int = 15) -> ndarray:
        """

        Parameters
        ----------
        similarity: similarity matrix or enhanced similarity matrix
        k: The numbers of neighbors defined

        Returns
        -------
        A adjacency matrix.
        """
        if isinstance(similarity, pd.DataFrame):
            similarity = similarity.values
        indices = np.argsort(-similarity, axis=1)[:, 1:k+1]
        adjacency_matrix = np.zeros_like(similarity)
        adjacency_matrix[np.arange(similarity.shape[0])[:, np.newaxis], indices] = 1
        return adjacency_matrix


# def load_data(fname: str, pca_dim: int, is_NE=True, n_clusters=20, K=None):
#     # Get data
#
#     adata: AnnData = sc.read_h5ad(filename=fname)
#     cells = adata.obs_names.values
#     genes = adata.var_names.values
#     features = adata.to_df()
#
#     # Preprocess features
#     # features = normalization(features)
#
#     # Construct graph
#     N = len(cells)
#     avg_N = N // n_clusters
#     K = avg_N // 10
#     K = min(K, 20)
#     K = max(K, 6)
#
#     L = 0
#     if is_NE:
#         method = 'NE'
#     else:
#         method = 'pearson'
#     adj = getGraph(dataset_str, features, L, K, method)
#
#     # feature tranformation
#     if features.shape[0] > pca_dim and features.shape[1] > pca_dim:
#         pca = PCA(n_components=pca_dim)
#         features = pca.fit_transform(features)
#     else:
#         var = np.var(features, axis=0)
#         min_var = np.sort(var)[-1 * pca_dim]
#         features = features.T[var >= min_var].T
#         features = features[:, :pca_dim]
#     print('Shape after transformation:', features.shape)
#
#     features = (features - np.mean(features)) / (np.std(features))
#
#     return adj, features, cells, genes






