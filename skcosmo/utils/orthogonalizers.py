# -*- coding: utf-8 -*-
"""

This module contains the necessary orthogonalizers for the CUR decomposition
subselection method.

Authors: Rose K. Cersonsky
         Michele Ceriotti

"""

import numpy as np
import warnings


def X_orthogonalizer(x1, c=None, x2=None, tol=1e-12, copy=False):
    """
    Orthogonalizes a feature matrix by the given columns. Can be used to
    orthogonalize by samples by calling `X = X_orthogonalizer(X.T, row_index).T`.

    :param x1: feature matrix to orthogonalize
    :type x1: matrix of shape (n x m)

    :param c: index of the column to orthogonalize by
    :type c: int, less than m

    :param x2: a separate set of columns to orthogonalize with column-by-column
    :type x2: matrix of shape (n x a)
    """

    if x2 is None and c is not None:
        cols = x1[:, [c]]
    elif x2.shape[0] == x1.shape[0]:
        cols = np.reshape(x2, (x1.shape[0], -1))
    else:
        raise ValueError(
            f"Orthogonalization only with a matrix containing {x1.shape[0]} entries."
        )

    if copy:
        xnew = x1.copy()
    else:
        xnew = x1

    for i in range(cols.shape[-1]):

        col = cols[:, [i]]

        if np.linalg.norm(col) < tol:
            warnings.warn("Column vector contains only zeros.")
        else:
            col /= np.linalg.norm(col, axis=0)

        xnew -= col @ (col.T @ xnew)

    return xnew


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


def Y_sample_orthogonalizer(y, X, y_ref, X_ref, tol=1e-12, copy=True):
    """
    Orthogonalizes a matrix of targets :math:`{\\mathbf{Y}}`given a reference feature matrix
    :math:`{\\mathbf{X}_r}` and reference target matrix :math:`{\\mathbf{Y}_r}`:

    .. math::

        \\mathbf{Y} \\leftarrow \\mathbf{Y} -
        \\mathbf{X} \\left(\\mathbf{X}_{\\mathbf{r}}^T
        \\mathbf{X}_{\\mathbf{r}}\\right)^{-1}\\mathbf{X}_{\\mathbf{r}}^T
        \\mathbf{Y}_{\\mathbf{r}}


    :param y: property matrix
    :type y: array of shape (n_samples x n_properties)

    :param X: corresponding feature matrix
    :type X: array of shape (n_samples x n_features)

    :param y_ref: reference property matrix
    :type y_ref: array of shape (n_ref x n_properties)

    :param X_ref: reference feature matrix
    :type X_ref: array of shape (n_ref x n_features)

    :param tol: cutoff for small eigenvalues to send to np.linalg.pinv
    :type tol: float

    :param copy: whether to return a copy of y or edit in-place, default=True
    :type copy: boolean
    """

    y_frag = (
        X @ (np.linalg.pinv(X_ref.T @ X_ref, rcond=tol) @ X_ref.T) @ y_ref
    ).reshape(y.shape)

    if copy:
        return y.copy() - y_frag
    else:
        y -= y_frag
        return y
