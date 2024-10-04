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


from numpy import ndarray
from anndata import AnnData
from scipy.linalg import eigh
from scipy.stats import spearmanr
from typing import Union, Optional
from sklearn.decomposition import PCA
from scipy.sparse import diags, csr_matrix
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

class NetworkEnhance:
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

    Parameters
    ----------
    diffusion_order : float, optional
        Determines the extent of diffusion. Typical values are 0.5, 1, 2. Default is 2.
    num_nearest_neighbors : int, optional
        The number of nearest neighbors. If None, it defaults to min(20, ceil(N / 10)).
    regularization_alpha : float, optional
        The regularization parameter. Default is 0.9.
    """

    def __init__(self,
                 diffusion_order: float = 2, num_nearest_neighbors: int = None, regularization_alpha: float = 0.1):
        self.diffusion_order = diffusion_order
        self.num_nearest_neighbors = num_nearest_neighbors
        self.regularization_alpha = regularization_alpha

    @staticmethod
    def normalize_matrix(input_matrix: np.ndarray, normalization_type: str) -> np.ndarray:
        """
        Normalize the input matrix based on the specified type.

        Parameters
        ----------
        input_matrix : np.ndarray
            The input matrix to be normalized.
        normalization_type : str
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
        num_nodes = input_matrix.shape[0]
        scaled_matrix = input_matrix * num_nodes
        scaled_matrix = scaled_matrix.astype(float)
        degree_sum = np.sum(np.abs(scaled_matrix), axis=1) + np.finfo(float).eps

        if normalization_type == 'ave':
            degree_inverse = 1.0 / degree_sum
            degree_sparse_matrix = diags(degree_inverse)
            normalized_sparse_matrix = degree_sparse_matrix.dot(scaled_matrix)
            normalized_matrix = normalized_sparse_matrix
        elif normalization_type == 'gph':
            degree_inverse_sqrt = 1.0 / np.sqrt(degree_sum)
            degree_sparse_matrix = diags(degree_inverse_sqrt)
            intermediate_matrix = degree_sparse_matrix.dot(scaled_matrix)
            normalized_sparse_matrix = intermediate_matrix.dot(degree_sparse_matrix)
            normalized_matrix = normalized_sparse_matrix
        else:
            raise ValueError("Invalid type. Must be 'ave' or 'gph'.")

        return normalized_matrix

    @staticmethod
    def construct_pairwise_nearest_neighbor_matrix(affinity_matrix: np.ndarray, num_nearest_neighbors: int) -> np.ndarray:
        """
        Constructs a symmetric Pairwise Nearest Neighbor (PNN) matrix from an affinity matrix.

        Parameters
        ----------
        affinity_matrix : np.ndarray
            The affinity matrix (2D array).
        num_nearest_neighbors : int
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
        if affinity_matrix.ndim != 2:
            raise ValueError("affinity_matrix must be a 2D array.")

        num_rows, num_cols = affinity_matrix.shape
        sorted_neighbor_indices = np.argsort(-affinity_matrix, axis=1)  # Sort in descending order
        sorted_affinity_values = np.sort(-affinity_matrix, axis=1) * -1  # Sort descending

        topk_affinity_values = sorted_affinity_values[:, :num_nearest_neighbors]
        topk_neighbor_indices = sorted_neighbor_indices[:, :num_nearest_neighbors]
        row_indices = np.repeat(np.arange(num_rows), num_nearest_neighbors)
        column_indices = topk_neighbor_indices.flatten()
        affinity_values = topk_affinity_values.flatten()

        pnn_matrix_partial = np.zeros_like(affinity_matrix, dtype=float)
        pnn_matrix_partial[row_indices, column_indices] = affinity_values

        pnn_matrix_symmetrized = (pnn_matrix_partial + pnn_matrix_partial.T) / 2.0

        return pnn_matrix_symmetrized

    @staticmethod
    def compute_transition_fields(input_matrix: np.ndarray) -> np.ndarray:
        """
        Computes transition fields based on the input matrix.

        Parameters
        ----------
        input_matrix : np.ndarray
            The input matrix.

        Returns
        -------
        np.ndarray
            The transition fields matrix.
        """
        input_matrix = np.array(input_matrix, dtype=float)
        normalization_factors = np.sqrt(np.sum(np.abs(input_matrix), axis=0) + np.finfo(float).eps)
        normalization_factors[normalization_factors == 0] = np.finfo(float).eps  # Prevent division by zero
        normalized_matrix = input_matrix / normalization_factors[np.newaxis, :]
        transition_matrix = np.dot(normalized_matrix, normalized_matrix.T)

        transition_matrix_copy = transition_matrix.copy()
        zero_sum_indices = np.where(np.sum(transition_matrix, axis=1) == 0)[0]
        if zero_sum_indices.size > 0:
            transition_matrix_copy[zero_sum_indices, :] = 0
            transition_matrix_copy[:, zero_sum_indices] = 0

        return transition_matrix_copy

    def network_enhancement(
        self,
        input_network_matrix: np.ndarray,
        diffusion_order: float = None,
        num_nearest_neighbors: int = None,
        regularization_alpha: float = None
    ) -> np.ndarray:
        """
        Enhances the input network matrix using diffusion and normalization techniques.

        Parameters
        ----------
        input_network_matrix : np.ndarray
            The input network matrix of size N x N.
        diffusion_order : float, optional
            Determines the extent of diffusion.
            Typical values are 0.5, 1, 2.
            If None, use the value set during initialization.
        num_nearest_neighbors : int, optional
            The number of nearest neighbors.
            If None, it defaults to min(20, ceil(N / 10)).
            If provided, it overrides the value set during initialization.
        regularization_alpha : float, optional
            The regularization parameter.
            If None, it defaults to 0.9.
            If provided, it overrides the value set during initialization.

        Returns
        -------
        np.ndarray
            The enhanced network matrix.

        Raises
        ------
        ValueError
            If input dimensions are inconsistent or parameters are invalid.
        """
        # Use instance parameters if arguments are not provided
        diffusion_order = self.diffusion_order if diffusion_order is None else diffusion_order
        num_nearest_neighbors = self.num_nearest_neighbors if num_nearest_neighbors is None else num_nearest_neighbors
        regularization_alpha = self.regularization_alpha if regularization_alpha is None else regularization_alpha

        input_network_matrix = np.array(input_network_matrix, dtype=float)
        num_nodes = input_network_matrix.shape[0]

        # Set default number of nearest neighbors if not provided
        if num_nearest_neighbors is None:
            num_nearest_neighbors = min(20, int(np.ceil(num_nodes / 10)))

        # Step 1: Remove self-connections by setting diagonal to zero
        network_without_self_loops = input_network_matrix * (1 - np.eye(num_nodes))

        # Step 2: Identify nodes with at least one non-zero connection
        active_node_indices = np.where(np.sum(np.abs(network_without_self_loops), axis=1) > 0)[0]

        # Step 3: Extract the submatrix corresponding to active nodes
        subnetwork_matrix = input_network_matrix[np.ix_(active_node_indices, active_node_indices)]
        #
        # # Step 4: Normalize the subnetwork using 'ave' type normalization
        # normalized_subnetwork_matrix = self.normalize_matrix(subnetwork_matrix, 'ave')

        # Step 5: Symmetrize the normalized subnetwork
        symmetrized_normalized_matrix = (subnetwork_matrix + subnetwork_matrix.T) / 2

        # Step 6: Compute the degree vector as the sum of absolute values across columns
        degree_vector = np.sum(np.abs(subnetwork_matrix), axis=0)

        # Step 7: Determine if the normalized subnetwork has only two unique values
        if len(np.unique(symmetrized_normalized_matrix)) == 2:
            pairwise_nearest_neighbor_matrix = symmetrized_normalized_matrix
        else:
            # Compute effective number of neighbors to avoid exceeding matrix dimensions
            effective_num_neighbors = min(num_nearest_neighbors, symmetrized_normalized_matrix.shape[0] - 1)
            pairwise_nearest_neighbor_matrix = self.construct_pairwise_nearest_neighbor_matrix(
                np.abs(symmetrized_normalized_matrix).astype(float),
                effective_num_neighbors
            ) * np.sign(symmetrized_normalized_matrix)

        # Step 8: Update P by adding identity and diagonal matrix of row sums
        row_sums = np.sum(np.abs(pairwise_nearest_neighbor_matrix), axis=1)
        updated_pairwise_matrix = pairwise_nearest_neighbor_matrix + (np.eye(pairwise_nearest_neighbor_matrix.shape[0]) + np.diag(row_sums))

        # Step 9: Apply Transition Fields
        transition_fields_matrix = self.compute_transition_fields(updated_pairwise_matrix)

        # Step 10: Eigen decomposition of P
        eigenvalues, eigenvectors = np.linalg.eig(transition_fields_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        # Step 11: Adjust eigenvalues with regularization parameter alpha
        epsilon = np.finfo(float).eps
        adjusted_eigenvalues = eigenvalues - epsilon
        adjusted_eigenvalues = (1 - regularization_alpha) * adjusted_eigenvalues / (1 - regularization_alpha * (adjusted_eigenvalues ** diffusion_order))

        # Step 12: Reconstruct W using modified eigenvalues
        eigenvalue_matrix = np.diag(adjusted_eigenvalues)
        reconstructed_matrix = eigenvectors @ eigenvalue_matrix @ eigenvectors.T

        # Step 13: Remove self-connections and normalize
        reconstructed_matrix = (reconstructed_matrix * (1 - np.eye(reconstructed_matrix.shape[0]))) / (1 - np.diag(reconstructed_matrix))[:, np.newaxis]

        # Step 14: Apply DD as a diagonal scaling factor
        degree_diagonal_matrix = diags(degree_vector)
        scaled_matrix = degree_diagonal_matrix.dot(reconstructed_matrix)

        # Step 15: Set negative values to zero
        scaled_matrix[scaled_matrix < 0] = 0

        # Step 16: Symmetrize W
        symmetrized_scaled_matrix = (scaled_matrix + scaled_matrix.T) / 2

        # Step 17: Initialize W_out and assign the enhanced W to corresponding indices
        enhanced_network_matrix = np.zeros_like(input_network_matrix)
        enhanced_network_matrix[np.ix_(active_node_indices, active_node_indices)] = symmetrized_scaled_matrix

        return enhanced_network_matrix

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







