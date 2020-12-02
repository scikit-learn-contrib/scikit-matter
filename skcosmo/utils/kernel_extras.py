#!/usr/bin/env python

import os
import sys
import functools
import numpy as np
from tqdm import tqdm

def kernel_decorator(kernel_func):
    """
        Decorator for kernel functions.

        ---Arguments---
        kernel_func: kernel function to wrap

        ---Returns---
        kernel_wrapper: wrapped kernel function
    """

    @functools.wraps(kernel_func)
    def kernel_wrapper(XA, XB, **kwargs):
        """
            Wrapper for kernel functions

            ---Arguments---
            XA, XB: datasets with which to build the kernel.
                If a dataset is provided as a list,
                the kernel is averaged over the corresponding
                axis, blocked according to the list elements
            kwargs: keyword arguments passed to the kernel functions

            ---Returns---
            K: the kernel matrix
        """

        # XA structures, XB structures
        if isinstance(XA, list) and isinstance(XB, list):
            K = np.zeros((len(XA), len(XB)))
            for adx, a in enumerate(tqdm(XA)):
                for bdx, b in enumerate(XB):
                    K[adx, bdx] = np.mean(kernel_func(a, b, **kwargs))

        # XA structures, XB environments
        elif isinstance(XA, list):
            K = np.zeros((len(XA), XB.shape[0]))
            for adx, a in enumerate(tqdm(XA)):
                K[adx, :] = np.mean(kernel_func(a, XB, **kwargs), axis=0)

        # XA environments, XB structures
        elif isinstance(XB, list):
            K = np.zeros((XA.shape[0], len(XB)))
            for bdx, b in enumerate(tqdm(XB)):
                K[:, bdx] = np.mean(kernel_func(XA, b, **kwargs), axis=1)

        # XA environments, XB environments
        else:
            K = kernel_func(XA, XB, **kwargs)

        return K

    return kernel_wrapper 

def partial_kernel_decorator(partial_kernel_func):
    """
        Decorator for partial kernel functions.

        ---Arguments---
        partial_kernel_func: partial kernel function to wrap

        ---Returns---
        partial_kernel_wrapper: wrapped partial kernel function
    """

    @functools.wraps(partial_kernel_func)
    def partial_kernel_wrapper(XA, XB, **kwargs):
        """
            Wrapper for partial kernel functions

            ---Arguments---
            XA, XB: datasets with which to build the partial kernel.
                If a dataset is provided as a list,
                the partial kernel is averaged over the corresponding
                axis, blocked according to the list elements
            kwargs: keyword arguments passed to the partial kernel functions

            ---Returns---
            K: the partial kernel vector 
        """

        # XA structures, XB structures
        if isinstance(XA, list) and isinstance(XB, list):

            # XA and XB should be the same length
            K = np.zeros(len(XA)) 

            # If both XA and XB are lists (structures),
            # we have to get the full distance matrix.
            # Since a, b will be simple numpy arrays,
            # we can just override the partial kernel function
            # with the 'standard' kernel functions.
            # We have to access by name since the
            # function object appears to change by passing
            # through the decorator
            if partial_kernel_func.__name__ == 'linear_partial_kernel':
                partial_kernel_func_override = linear_kernel
            elif partial_kernel_func.__name__ == 'gaussian_partial_kernel':
                partial_kernel_func_override = gaussian_kernel
            else:
                print("Error: unrecognized kernel function")
                return
            for idx, (a, b) in enumerate(zip(tqdm(XA), XB)):
                K[idx] = np.mean(partial_kernel_func_override(a, b, **kwargs))

        # XA structures, XB environments
        elif isinstance(XA, list):
            K = np.zeros(len(XA))
            for idx, (a, b) in enumerate(zip(tqdm(XA), XB)):
                K[idx] = np.mean(partial_kernel_func(a, b, **kwargs), axis=0)

        # XA environments, XB structures
        elif isinstance(XB, list):
            K = np.zeros(len(XB))
            for idx, (a, b) in enumerate(tqdm(zip(XA, XB))):
                K[idx] = np.mean(partial_kernel_func(a, b, **kwargs), axis=1)

        # XA environments, XB environments
        else:
            K = partial_kernel_func(XA, XB, **kwargs)

        return K

    return partial_kernel_wrapper 

def build_kernel(XA, XB, XR=None, kernel='linear', gamma=1.0, zeta=1.0): 
    """
        Build a kernel
        
        ---Arguments---
        XA: XA data; if XA is a list, with element i being an array of environments
            in structure i, then row i of the kernel matrix will be an
            an average over environments in structure i
        XB: XB data; if XB is a list, with element j being an array of environments
            in structure j, then column j of the kernel matrix will be an
            average over environments in structure j
        XR: XR data (Nystrom mode); if XR is provided, a Nystrom approximation
            of the kernel will be computed. If XR is a list, with element k being
            an array of environments in structure k, then the column k
            in the kernel KAR, the row k in the kernel KRB, and the columns
            and rows in kernel KRR will be an average over environments in structure k
        kernel: kernel type (linear or gaussian)
        gamma: gamma (width) parameter for gaussian kernels
        zeta: zeta (exponent) parameter for linear kernels
        
        ---Returns---
        K: kernel matrix
    """
    
    # Initialize kernel functions and special arguments
    if kernel == 'gaussian':
        kernel_func = gaussian_kernel
        kw = {'gamma': gamma}
    else:
        kernel_func = linear_kernel
        kw = {'zeta': zeta}

        # If we have a linear kernel of structures,
        # take the mean over the environments to speed things up,
        # since we can avoid the looping decorator over XA/XB/XR
        if isinstance(XA, list):
            XA = np.vstack([np.mean(xa, axis=0) for xa in XA])

        if isinstance(XB, list):
            XB = np.vstack([np.mean(xb, axis=0) for xb in XB])

        if isinstance(XR, list):
            XR = np.vstack([np.mean(xr, axis=0) for xr in XR])

    # Initialize kernel matrices
    KRR = None
    KAR = None
    KRB = None
    K = None

    # Compute the kernels, where we sum over the axes
    # corresponding to the data that are provided in lists, 
    # where each element of a list represents a structure
    # as an array with the feature vectors of the environments
    # present in that structure as rows

    # Nystrom mode
    if XR is not None:
        
        # Build kernels between XA/XB/XR and XR
        KRR = kernel_func(XR, XR, **kw)
        KAR = kernel_func(XA, XR, **kw)
        KRB = kernel_func(XA, XR, **kw)
        
        # Build approximate kernel
        KRR_inv = np.linalg.inv(KRR)
        K = np.matmul(KAR, KRR_inv)
        K = np.matmul(K, KRB)
              
    # Normal mode
    else:

        # Build kernel between XA and XB
        K = kernel_func(XA, XB, **kw)

    return K

def build_partial_kernel(XA, XB, kernel='linear', gamma=1.0, zeta=1.0,
        section='diag', k=0):
    """
        Build a partial kernel
        
        ---Arguments---
        XA: XA data; if XA is a list, with element i being an array of environments
            in structure i, then row i of the kernel matrix will be an
            an average over environments in structure i
        XB: XB data; if XB is a list, with element j being an array of environments
            in structure j, then column j of the kernel matrix will be an
            average over environments in structure j
        XR: XR data (Nystrom mode); if XR is provided, a Nystrom approximation
            of the kernel will be computed. If XR is a list, with element k being
            an array of environments in structure k, then the column k
            in the kernel KAR, the row k in the kernel KRB, and the columns
            and rows in kernel KRR will be an average over environments in structure k
        kernel: kernel type (linear or gaussian)
        gamma: gamma (width) parameter for gaussian kernels
        zeta: zeta (exponent) parameter for linear kernels
        section: portion of the kernel to compute. Options are 
            'diag', 'upper', or 'lower' for computing the kernel diagonal,
            upper triangle, or lower triangle
        k: kth diagonal (0 for the main diagonal,
            k < 0 for below main diagonal, k > 0 for above main diagonal)
        
        ---Returns---
        K: vector of values from the kernel matrix in row major order
    """
    # TODO: Nystrom mode
    
    # Initialize kernel functions and special arguments
    if kernel == 'gaussian':
        kernel_func = gaussian_partial_kernel
        kw = {'gamma': gamma}
    else:
        kernel_func = linear_partial_kernel
        kw = {'zeta': zeta}

    if section == 'diag':
        XA_idxs, XB_idxs = diag_indices((len(XA), len(XB)), k=k)
    elif section == 'upper' or section == 'lower':
        XA_idxs, XB_idxs = tri_indices((len(XA), len(XB)), k=k,
                tri=section)
    else:
        print("Error: invalid selection. Valid options are "
                "'diag', 'upper', and 'lower'")
        return

    if isinstance(XA, list):
        XA = [XA[i] for i in XA_idxs]
    else:
        XA = XA[XA_idxs, :]

    if isinstance(XB, list):
        XB = [XB[i] for i in XB_idxs]
    else:
        XB = XB[XB_idxs, :]

    K = kernel_func(XA, XB, **kw)

    return K

def sqeuclidean_distances(XA, XB):
    """
        Evaluation of a distance matrix
        of squared euclidean distances

        ---Arugments---
        XA, XB: matrices of data with which to build the distance matrix,
            where each row is a sample and each column a feature

        ---Returns---
        D: distance matrix of shape A x B
    """

    # Reshape so arrays can be broadcast together into shape A x B
    XA2 = np.sum(XA**2, axis=1).reshape((-1, 1))
    XB2 = np.sum(XB**2, axis=1).reshape((1, -1))

    # Compute distance matrix
    D = XA2 + XB2 - 2*np.matmul(XA, XB.T)

    return D

def sqeuclidean_distances_vector(XA, XB): 
    """
        Evaluation of a vector
        of squared euclidean distances

        ---Arugments---
        XA, XB: matrices of data with which to build the distance matrix,
            where each row is a sample and each column a feature.
            The distance vector is computed between
            corresponding elements of XA and XB

        ---Returns---
        D: distance matrix of shape A x B
    """

    if XA.shape != XB.shape:
        print("Error: XA and XB must have same shape")
        return

    XA2 = np.sum(XA**2, axis=1)
    XB2 = np.sum(XB**2, axis=1)
    XAXB = np.sum(XA*XB, axis=1)

    # Compute distance matrix
    D = XA2 + XB2 - 2*XAXB

    return D

@kernel_decorator
def linear_kernel(XA, XB, zeta=1):
    """
        Builds a dot product kernel

        ---Arguments---
        XA, XB: matrices of data with which to build the kernel,
            where each row is a sample and each column a feature

        ---Returns---
        K: dot product kernel between XA and XB
    """

    K = np.matmul(XA, XB.T)**zeta
    return K

@kernel_decorator
def gaussian_kernel(XA, XB, gamma=1):
    """
        Builds a Gaussian kernel

        ---Arguments---
        XA, XB: matrices of data with which to build the kernel,
            where each row is a sample and each column a feature
        gamma: scaling parameter for the Gaussian

        ---Returns---
        K: Gaussian kernel between XA and XB
    """

    D = sqeuclidean_distances(XA, XB)
    K = np.exp(-gamma*D)
    return K

@partial_kernel_decorator
def gaussian_partial_kernel(XA, XB, gamma=1):
    """
        Computes a vector of Gaussian kernel values
        between corresponding samples

        ---Arguments---
        XA, XB: matrices of data with which to build the partial kernel,
            where each row is a sample and each column a feature
        gamma: scaling parameter for the Gaussian

        ---Returns---
        K: Gaussian partial kernel between corresponding samples
            in XA and XB
    """

    D = sqeuclidean_distances_vector(XA, XB)
    K = np.exp(-gamma*D)
    return K

@partial_kernel_decorator
def linear_partial_kernel(XA, XB, zeta=1):
    """
        Computes a vector of linear kernel values
        between corresponding samples

        ---Arguments---
        XA, XB: matrices of data with which to build the partial kernel,
            where each row is a sample and each column a feature
        gamma: scaling parameter for the Gaussian

        ---Returns---
        K: Gaussian partial kernel between corresponding samples
            in XA and XB
    """

    K = np.sum(XA*XB, axis=1)**zeta
    return K

def diag_indices(shape, k=0):
    """
        Computes the indices of the kth diagonal
        of a 2D matrix

        ---Arguments---
        shape: 2D tuple in the form (n_rows, n_columns)
        k: kth diagonal (0 for the main diagonal,
            k < 0 for below main diagonal, k > 0 for above main diagonal)

        ---Returns---
        idxs: tuple of array indices in the from (row_idxs, col_idxs)
    """

    row_start = np.abs(np.minimum(k, 0))
    row_end = np.minimum(np.abs(k - shape[1] + 1), shape[0] - 1)
    col_start = np.maximum(k, 0)
    col_end = np.minimum(k + shape[0] - 1, shape[1] - 1)

    row_idxs = np.arange(row_start, row_end + 1, dtype=int)
    col_idxs = np.arange(col_start, col_end + 1, dtype=int)
    idxs = (row_idxs, col_idxs)

    return idxs

def tri_indices(shape, k=0, tri='upper'):
    """
        Computes the indices of the upper or lower
        triangular matrix based on the diagonal

        ---Arguments---
        shape: 2D tuple in the form (n_rows, n_columns)
        k: kth diagonal (0 for the main diagonal,
            k < 0 for below main diagonal, k > 0 for above main diagonal)
        tri: 'upper' for upper triangular, 'lower' for lower triangular

        ---Returns---
        idxs: tuple of array indices in the form (row_idxs, col_idxs)
    """

    if tri == 'upper':
        start = k
        end = shape[1]

    elif tri == 'lower':
        start = -shape[0] + 1
        end = k + 1

    else:
        print("Error: 'tri' must be 'upper' or 'lower'")
        return

    row_idxs = []
    col_idxs = []
    for kk in np.arange(start, end):
        diag_idxs = diag_indices(shape, k=kk)
        row_idxs.append(diag_idxs[0])
        col_idxs.append(diag_idxs[1])

    row_idxs = np.concatenate(row_idxs)
    col_idxs = np.concatenate(col_idxs)
    row_idxs = np.sort(row_idxs)
    idxs = (row_idxs, col_idxs)

    return idxs
