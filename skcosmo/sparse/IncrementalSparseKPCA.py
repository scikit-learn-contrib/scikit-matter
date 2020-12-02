#!/usr/bin/env python

import os
import sys
import numpy as np
from regression import IncrementalSparseKRR
from tools import sorted_eigh

class IncrementalSparseKPCA(object):
    """
        Performs sparsified principal component analysis
        using batches. Example usage:

        KMM = build_kernel(Xm, Xm)
        iskpca = IncrementalSparseKPCA()
        iskpca.initialize_fit(KMM)
        for i in batches:
            KNMi = build_kernel(Xi, Xm)
            iskpca.fit_batch(KNMi)
        iskpca.finalize_fit()
        for i in batches:
            KNMi = build_kernel(Xi, Xm)
            iskpca.transform(KNMi)
        
        ---Attributes---
        n_components: number of principal components to retain
        T_mean: the column means of the approximate feature space
        n_samples: number of training points
        tiny: threshold for discarding small eigenvalues
        T_mean: auxiliary centering of the kernel matrix
            because the centering must be based on the
            feature space, which is approximated
        Um: eigenvectors of KMM
        Vm: eigenvalues of KMM
        Uc: eigenvalues of the covariance of T
        Vc: eigenvectors of the covariance of T
        V: projection matrix
        iskrr: SparseKRR object used to construct the inverse transform

        ---Methods---
        initialize_fit: initialize the sparse KPCA fit
            (i.e., compute eigendecomposition of KMM)
        fit_batch: fit a batch of training data
        finalize_fit: finalize the sparse KPCA fitting procedure
            (i.e., compute the KPCA projection vectors)
        transform: transform the sparse KPCA
        
        ---References---
        1.  https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
        2.  M. E. Tipping 'Sparse Kernel Principal Component Analysis',
            Advances in Neural Information Processing Systems 13, 633-639, 2001
        3.  C. Williams, M. Seeger, 'Using the Nystrom Method to Speed Up Kernel Machines',
            Avnaces in Neural Information Processing Systems 13, 682-688, 2001
        4.  K. Zhang, I. W. Tsang, J. T. Kwok, 'Improved Nystrom Low-Rank Approximation
            and Error Analysis', Proceedings of the 25th International Conference
            on Machine Learning, 1232-1239, 2008
        5.  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    
    def __init__(self, n_components=None, tiny=1.0E-15):
        self.n_components = n_components
        self.tiny = tiny
        self.C = None
        self.T_mean = None
        self.n_samples = None
        self.Um = None
        self.Vm = None
        self.Uc = None
        self.Vc = None
        self.V = None
        self.iskrr = None

    def initialize_fit(self, KMM):
        """
            Computes the eigendecomposition of the
            kernel matrix between the representative points

            ---Arguments---
            KMM: centered kernel between the representative points
        """

        self.Um, self.Vm = sorted_eigh(KMM, tiny=self.tiny)

        self.C = np.zeros((self.Um.size, self.Um.size))
        self.T_mean = np.zeros(self.Um.size)
        self.n_samples = 0
        
    def fit_batch(self, KNM):
        """
            Fits a batch for the sparse KPCA

            ---Arguments---
            KNM: centered kernel between all training points
                and the representative points
        """

        if self.Um is None or self.Vm is None:
            print("Error: must initialize the fit with a KMM"
                    "before fitting batches")
            return

        # Reshape 1D arrays
        if KNM.ndim < 2:
            KNM = np.reshape(KNM, (1, -1))

        # Don't need to do auxiliary centering of T or KNM
        # since the covariance matrix will be centered once
        # we are finished building it

        # TODO: also scale T?
        T = np.matmul(KNM, self.Vm)
        T = np.matmul(T, np.diagflat(1.0/np.sqrt(self.Um)))

        # Single iteration of one-pass covariance
        old_mean = self.T_mean
        self.n_samples += KNM.shape[0]
        self.T_mean = old_mean + np.sum(T-old_mean, axis=0)/self.n_samples

        self.C += np.matmul((T-self.T_mean).T, T-old_mean)

    def finalize_fit(self):
        """
            Finalize the sparse KPCA fitting procedure
        """

        if self.n_samples < 1:
            print("Error: must fit at least one batch"
                    "before finalizing the fit")
            return

        self.Uc, self.Vc = sorted_eigh(self.C, tiny=None)

        self.T_mean = np.matmul(self.T_mean, self.Vc)

        self.V = np.matmul(self.Vm, np.diagflat(1.0/np.sqrt(self.Um)))
        self.V = np.matmul(self.V, self.Vc)

        self.V = self.V[:, 0:self.n_components]
        self.T_mean = self.T_mean[0:self.n_components]

    def transform(self, KNM):
        """
            Transforms the sparse KPCA

            ---Arguments---
            KNM: centered kernel between the training/testing
                points and the representatitve points

            ---Returns---
            T: centered transformed KPCA scores
        """

        if self.V is None:
            print("Error: must fit the KPCA before transforming")
        else:
            T = np.matmul(KNM, self.V) - self.T_mean
            return T

    def initialize_inverse_transform(self, KMM, x_dim=1, sigma=1.0, 
            regularization=1.0E-12, regularization_type='scalar', rcond=None):
        """
            Initialize the sparse KPCA inverse transform

            ---Arguments---
            KMM: centered kernel between the transformed representative points
            x_dim: dimension of X data
            sigma: regulariztion parameter 
            regularization: additional regularization for the Sparse KRR solution
                for the inverse transform
            rcond: cutoff ratio for small singular values in the least squares
                solution to determine the inverse transform
        """

        # (can also do LR here)
        self.iskrr = IncrementalSparseKRR(sigma=sigma, regularization=regularization, 
                regularization_type=regularization_type, rcond=rcond)
        self.iskrr.initialize_fit(KMM, y_dim=x_dim)

    def fit_inverse_transform_batch(self, KTM, X):
        """
            Fit a batch for the inverse KPCA transform

            ---Arguments---
            KTM: centered kernel between the KPCA transformed training data
                and the transformed representative points
            X: the centered original input data
        """

        self.iskrr.fit_batch(KTM, X)

    def finalize_inverse_transform(self):
        """
            Finalize the fitting of the inverse KPCA transform
        """

        self.iskrr.finalize_fit()

    def inverse_transform(self, KXM):
        """
            Computes the reconstruction of X

            ---Arguments---
            KXM: centered kernel between the transformed data and the 
                representative transformed data

            ---Returns---
            Xr: reconstructed centered input data
        """

        # Compute the reconstruction
        W = self.iskrr.W
        Xr = np.matmul(KXM, W)

        return Xr
