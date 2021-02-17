# -*- coding: utf-8 -*-
"""

This module contains the necessary orthogonalizers for the CUR decomposition
subselection method.

Authors: Rose K. Cersonsky
         Michele Ceriotti

"""

import numpy as np
import warnings


def X_orthogonalizer(x1, c=None, x2=None, tol=1e-12):
    """
    Orthogonalizes a feature matrix by the given columns. Can be used to
    orthogonalize by samples by calling `X = X_orthogonalizer(X.T, row_index).T`.

    :param x1: feature matrix to orthogonalize
    :type x1: matrix of shape (n x m)

    :param c: index of the column to orthogonalize by
    :type c: int, less than m

    :param x2: a separate set of columns to orthogonalize by
    :type x2: matrix of shape (n x a)
    """

    if x2 is None and c is not None:
        x2 = x1[:, [c]]

    orthogonalizer = np.eye(x1.shape[0])

    for i in range(x2.shape[-1]):
        col = orthogonalizer @ x2[:, [i]]

        if np.linalg.norm(col) < tol:
            warnings.warn("Column vector contains only zeros.")
        else:
            col /= np.linalg.norm(col)

        orthogonalizer = (np.eye(x1.shape[0]) - col @ col.T) @ orthogonalizer

    return orthogonalizer @ x1


def Y_feature_orthogonalizer(y, X, tol=1e-12, copy=True):
    r"""
    Orthogonalizes a property matrix given the selected features in :math:`\mathbf{X}`

    .. math::
        \mathbf{Y} \leftarrow \mathbf{Y} -
        \mathbf{X} \left(\mathbf{X}^T\mathbf{X}\right)^{-1}\mathbf{X}^T \mathbf{Y}

    :param y: property matrix
    :type y: array of shape (n_samples x n_properties)

    :param X: feature matrix
    :type X: array of shape (n_samples x n_features)

    :param tol: cutoff for small eigenvalues to send to np.linalg.pinv
    :type tol: float

    :param copy: whether to return a copy of y or edit in-place, default=True
    :type copy: boolean
    """

    v = np.linalg.pinv(np.matmul(X.T, X), rcond=tol)
    v = np.matmul(X, v)
    v = np.matmul(v, X.T)

    if copy:
        return y.copy() - np.matmul(v, y)
    else:
        y -= np.matmul(v, y)
        return y


def sample_orthogonalizer(idx, X_proxy, Y_proxy, tol=1e-12):
    """
    Orthogonalizes two matrices, meant to represent a feature matrix
    :math:`{\\mathbf{X}}` and a property matrix :math:`{\\mathbf{Y}}`, given
    the selected samples :math:`{r}`

    Update to :math:`{\\mathbf{X}}`, where :math:`{\\mathbf{x}_{r}}` is a row
    vector containing the most recently-chosen sample:

    .. math::

        \\mathbf{X} \\leftarrow \\mathbf{X} -
        \\mathbf{X} \\left(\\frac{\\mathbf{x}_{r}^T \\mathbf{x}_{r}}
        {\\lVert \\mathbf{x}_{r}\\rVert^2}\\right)

    Update to :math:`{\\mathbf{Y}}`, where :math:`{\\mathbf{X}_{\\mathbf{r}}}`
    and :math:`{\\mathbf{Y}_{\\mathbf{r}}}`
    contain the features and properties of the previously-chosen samples:

    .. math::

        \\mathbf{Y} \\leftarrow \\mathbf{Y} -
        \\mathbf{X} \\left(\\mathbf{X}_{\\mathbf{r}}^T
        \\mathbf{X}_{\\mathbf{r}}\\right)^{-1}\\mathbf{X}_{\\mathbf{r}}^T
        \\mathbf{Y}_{\\mathbf{r}}

    :param idx: indices of selected samples
    :type idx: list of int

    :param X_proxy: feature matrix
    :type X_proxy: array of shape (n_samples x n_features)

    :param Y_proxy: property matrix
    :type Y_proxy: array of shape (n_samples x n_properties)

    :param tol: cutoff for small eigenvalues to send to np.linalg.pinv
    :type tol: float

    """
    if X_proxy is not None:
        if Y_proxy is not None:
            Y_proxy -= (
                X_proxy
                @ (
                    np.linalg.pinv(X_proxy[idx].T @ X_proxy[idx], rcond=tol)
                    @ X_proxy[idx].T
                )
                @ Y_proxy[idx]
            )

        X_proxy = X_orthogonalizer(X_proxy.T, idx[-1]).T

    return X_proxy, Y_proxy
