#!/usr/bin/env python

import os
import sys
import numpy as np
from tools import sorted_eigh

class IncrementalSparseKRR(object):
    """
        Performs sparsified kernel ridge regression. Example usage:

        KMM = build_kernel(Xm, Xm)
        iskrr = IncrementalSparseKRR()
        iskrr.initialize_fit(KMM)
        for i in batches:
            KNMi = build_kernel(Xi, Xm)
            iskrr.fit_batch(KNMi, Yi)
        iskrr.finalize_fit()
        for i in batches:
            KNMi = build_kernel(Xi, Xm)
            iskrr.predict(KNMi)

        ---Attributes---
        sigma: regularization parameter
        regularization: additional regularization
        regularization_type: type of regularization.
            Choices are 'scalar' (constant regularization),
            or 'max_eig' (regularization based on maximum eigenvalue)
        W: regression weights
        KMM: centered kernel between representative points
        KY: product of the KNM kernel and the properties Y
        
        ---Methods---
        initialize_fit: initialize the fit of the sparse KRR
            (i.e., store KMM)
        fit_batch: fit a batch of training data
        finalize_fit: finalize the KRR fitting
            (i.e., compute regression weights)
        predict: compute predicted Y values
        
        ---References---
        1.  M. Ceriotti, M. J. Willatt, G. Csanyi,
            'Machine Learning of Atomic-Scale Properties
            Based on Physical Principles', Handbook of Materials Modeling,
            Springer, 2018
        2.  A. J. Smola, B. Scholkopf, 'Sparse Greedy Matrix Approximation 
            for Machine Learning', Proceedings of the 17th International
            Conference on Machine Learning, 911-918, 2000
    """
    
    def __init__(self, sigma=1.0, regularization=1.0E-12, 
            regularization_type='scalar', rcond=None):
        self.sigma = sigma
        self.regularization = regularization
        self.regularization_type = regularization_type
        self.rcond = rcond
        self.W = None
        self.KX = None
        self.KY = None
        self.y_dim = None

    def initialize_fit(self, KMM, y_dim=1):
        """
            Initialize the KRR fitting by computing the
            eigendecomposition of KMM

            ---Arguments---
            KMM: centered kernel between the representative points
            y_dim: number of properties
        """

        # Check for valid Y dimension
        if y_dim < 1:
            print("Y dimension must be at least 1")
            return

        self.y_dim = y_dim
        self.KX = self.sigma*KMM
        self.KY = np.zeros((KMM.shape[0], self.y_dim)) 
        
    def fit_batch(self, KNM, Y):
        """
            Fits the KRR model by computing the regression weights

            ---Arguments---
            KNM: centered kernel between the whole dataset "
                "and the representative points
            Y: centered property values
        """

        if self.KX is None:
            print("Error: must initialize the fit before fitting the batch")
            return

        # Turn scalar into 2D array
        if not isinstance(Y, np.ndarray):
            Y = np.array([[Y]])

        # Reshape 1D kernel
        if KNM.ndim < 2:
            KNM = np.reshape(KNM, (1, -1))

        # Reshape 1D properties
        if Y.ndim < 2:
            Y = np.reshape(Y, (-1, self.y_dim))

        self.KX += np.matmul(KNM.T, KNM)
        self.KY += np.matmul(KNM.T, Y)
    
    def finalize_fit(self):
        """
            Finalize the iterative fitting of the sparse KRR model
            by computing regression weights
        """

        # Regularize the model
        if self.regularization_type == 'max_eig':
            scale = np.amax(np.linalg.eigvalsh(KX))
        elif self.regularization_type == 'scalar':
            scale = 1.0
        else:
            print("Error: invalid regularization_type. Use 'scalar' or 'max_eig'")
            return

        self.KX += np.eye(self.KX.shape[0])*scale*self.regularization

        # Solve KRR model
        self.W = np.linalg.lstsq(self.KX, self.KY, rcond=self.rcond)[0]
        
    def predict(self, KNM):
        """
            Computes predicted Y values

            ---Arguments---
            KNM: centered kernel matrix between training and testing data

            ---Returns---
            Yp: centered predicted Y values

        """

        if self.W is None:
            print("Error: must fit the KRR model before transforming")
        else:
            Yp = np.matmul(KNM, self.W)
            
            return Yp
