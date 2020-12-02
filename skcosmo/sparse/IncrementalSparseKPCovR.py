#!/usr/bin/env python

import os
import sys
import numpy as np
from tools import sorted_eigh

class IncrementalSparseKPCovR(object):
    """
        Performs sparsified kernel principal covariates regression
        with iterative fitting

        ---Attributes---
        alpha: tuning parameter
        n_components: number of kernel principal components to retain
        regularization: regularization parameter
        regularization_type: type of regularization.
            Choices are 'scalar' (constant regularization),
            or 'max_eig' (regularization based on maximum eigenvalue)
        tiny: threshold for discarding small eigenvalues
        T_mean: auxiliary centering for the kernel matrix
        phi_mean: mean of RKHS features
        C: covariance (Phi.T x Phi)
            because the centering must be done based on the
            feature space, which is approximated
        Um: eigenvalues of KMM
        Vm: eigenvectors of KMM
        Uc: eigenvalues of Phi.T x Phi
        Vc: eigenvectors of Phi.T x Phi
        U: eigenvalues of S
        V: eigenvectors of S
        Pkt: projection matrix from the kernel matrix (K) to
            the latent space (T)
        Pty: projection matrix from the latent space (T) to
            the properties (Y)
        Ptk: projection matrix from the latent space (T) to
            the kernel matrix (K)
        P_scale: scaling for projection matrices
        Y_norm: scaling for the LR term of G
        iskrr: incremental sparse KRR object

        ---Methods---
        initialize_fit: initialize the fitting for the sparse KPCovR model
        fit_batch: fit a batch of data
        finalize_fit: finalize the fitting procedure
        transform: transform the data into KPCA space
        inverse_transform: computes the reconstructed kernel
            or the original X data (if provided during the fit)
        predict: computes predicted Y values
        regression_loss: computes the sparse kernel ridge regression loss
        projection_loss: computes the projection loss
        gram_loss: computes the Gram loss
    """

    def __init__(self, alpha=0.0, n_components=None, sigma=1.0, 
            regularization=1.0E-12, regularization_type='scalar', 
            tiny=1.0E-15, rcond=None):
        self.alpha = alpha
        self.n_components = n_components
        self.regularization = regularization
        self.regularization_type = regularization_type
        self.sigma = sigma
        self.tiny = tiny
        self.rcond = rcond
        self.T_mean = None
        self.C = None
        self.Um = None
        self.Vm = None
        self.Uc = None
        self.Vc = None
        self.U = None
        self.V = None
        self.Pkt = None
        self.Pty = None
        self.Ptk = None
        self.Ptx = None
        self.P_scale = None
        self.Y_norm = None
        self.iskrr = None

    def initialize_fit(self, KMM, y_dim=1):
        """
            Computes the eigendecomposition of the
            kernel matrix between the representative points

            ---Arguments---
            KMM: centered kernel between the representative points
            y_dim: number of properties
        """

        self.Um, self.Vm = sorted_eigh(KMM, tiny=self.tiny)

        # Set shape of T_mean and C according ot the
        # number of nonzero eigenvalues
        self.C = np.zeros((self.Um.size, self.Um.size))
        self.T_mean = np.zeros(self.Um.size)
        self.n_samples = 0
        self.Y_norm = 0

        # Initialize the iterative sparse KRR
        self.iskrr = IncrementalSparseKRR(sigma=self.sigma, 
                regularization=self.regularization, 
                regularization_type=self.regularization_type, 
                rcond=self.rcond)

        self.iskrr.initialize_fit(KMM, y_dim=y_dim)

    def fit_batch(self, KNM, Y):
        """
            Fits a batch of the sparse KPCovR model

            ---Arguments---
            KNM: centered kernel between all points and the subsampled points
            KMM: centered kernel between the subsampled points
            Y: centered dependent (response) variable
        """

        self.iskrr.fit_batch(KNM, Y)

        self.Y_norm += np.sum(Y**2)

        # Compute the feature space data
        phi = np.matmul(KNM, self.Vm)
        phi = np.matmul(phi, np.diagflat(1.0/np.sqrt(self.Um)))

        # Auxiliary centering of phi
        # since we are working with an approximate feature space
        old_mean = self.T_mean
        self.n_samples += phi.shape[0]
        self.T_mean = old_mean + np.sum(phi-old_mean, axis=0)/self.n_samples

        self.C += np.matmul((phi-self.T_mean).T, phi-old_mean)

        # TODO: incremental fit on X (if provided) for inverse transformation
        # Do with separate functions like IncrementalSparseKPCA

    def finalize_fit(self, X=None):

        self.P_scale = np.sqrt(np.trace(self.C))
        self.Y_norm = np.sqrt(self.Y_norm)

        self.iskrr.finalize_fit()
        W = self.iskrr.W

        # Change from kernel-based W to phi-based W
        W = np.matmul(self.Vm.T, W)
        W = np.matmul(np.diagflat(np.sqrt(self.Um)), W)

        self.Uc, self.Vc = sorted_eigh(self.C, tiny=self.tiny)

        C_inv_sqrt = np.matmul(self.Vc, np.diagflat(1.0/np.sqrt(self.Uc)))
        C_inv_sqrt = np.matmul(C_inv_sqrt, self.Vc.T)

        C_sqrt = np.matmul(self.Vc, np.diagflat(np.sqrt(self.Uc)))
        C_sqrt = np.matmul(C_sqrt, self.Vc.T)

        S_kpca = C/self.P_scale**2
        S_lr = np.matmul(C_sqrt, W)
        S_lr = np.matmul(S_lr, S_lr.T)/self.Y_norm**2

        S = self.alpha*S_kpca + (1.0 - self.alpha)*S_lr

        self.U, self.V = sorted_eigh(S, tiny=self.tiny)

        self.U = self.U[0:self.n_components]
        self.V = self.V[:, 0:self.n_components]

        P = np.matmul(np.diagflat(1.0/np.sqrt(self.U)), self.V.T)
        PP = np.matmul(C_inv_sqrt, self.V)
        PP = np.matmul(PP, np.diagflat(np.sqrt(self.U)))

        self.T_mean = np.matmul(self.T_mean, PP)
        self.T_mean *= self.P_scale 

        self.Pkt = np.matmul(self.Vm, np.diagflat(1.0/np.sqrt(self.Um)))
        self.Pkt = np.matmul(self.Pkt, PP)
        self.Pkt *= self.P_scale

        self.Pty = np.matmul(P, C_inv_sqrt)
        self.Pty = np.matmul(self.Pty, phi.T)
        self.Pty = np.matmul(self.Pty, Y)
        self.Pty /= self.P_scale

        self.Ptk = np.matmul(P, C_sqrt)
        self.Ptk = np.matmul(self.Ptk, np.diagflat(np.sqrt(self.Um)))
        self.Ptk = np.matmul(self.Ptk, self.Vm.T)
        self.Ptk /= self.P_scale

        if X is not None:
            self.Ptx = np.matmul(P, C_inv_sqrt)
            self.Ptx = np.matmul(self.Ptx, phi.T)
            self.Ptx = np.matmul(self.Ptx, X)
            self.Ptx /= self.P_scale

    def transform(self, KNM):
        """
            Transform the data into KPCA space

            ---Arguments---
            KNM: centered kernel between all points and the representative points

            ---Returns---
            T: centered transformed data
        """

        if self.Pkt is None:
            print("Error: must fit the KPCovR model before transforming")
        else:

            # Compute the KPCA-like projections
            T = np.matmul(KNM, self.Pkt) - self.T_mean

            return T

    def inverse_transform(self, KNM, mode='K'):
        """
            Compute the reconstruction of the kernel matrix

            ---Arguments---
            KNM: centered kernel between all points and the representative points
            mode: whether to do a reconstruction of the kernel ('K')
                or a reconstruction of X ('X')

            ---Returns---
            R: centered reconstructed quantity
        """

        if mode == 'K':
            if self.Ptk is None:
                print("Error: must fit the KPCovR model before transforming")
                return
            else:

                # Compute the KPCA-like projections
                T = self.transform(KNM)
                R = np.matmul(T + self.T_mean, self.Ptk)

        elif mode == 'X':
            if self.Ptx is None:
                print("Error: must provide X data during the KPCovR fit before transforming")
                return
            else:

                # Compute the KPCA-like projections
                T = self.transform(KNM)

                # NOTE: why does adding T_mean not work?
                # perhaps because we need X centered in feature space,
                # not the RKHS feature space
                R = np.matmul(T, self.Ptx)
        else:
            print("Error: invalid reconstruction mode; "
                    "use 'K' or 'X'")
            return

        return R

    def predict(self, KNM):
        """
            Compute the predictions of Y

            ---Arguments---
            KNM: centered kernel between all points and the representative points

            ---Returns---
            Yp: centered predicted Y values
        """

        if self.Pty is None:
            print("Error: must fit the KPCovR model before transforming")
        else:

            # Compute predicted Y values
            T = self.transform(KNM)
            Yp = np.matmul(T + self.T_mean, self.Pty)

            return Yp
