# -*- coding: utf-8 -*-
"""Necessary orthogonalizers for the CUR decomposition subselection method.

Authors: Rose K. Cersonsky
         Michele Ceriotti
"""

import warnings

import numpy as np


def X_orthogonalizer(x1, c=None, x2=None, tol=1e-12, copy=False):
    """Orthogonalizes a feature matrix by the given columns.

    Can be used to orthogonalize by samples by calling `X = X_orthogonalizer(X.T,
    row_index).T`. After orthogonalization, each column of X will contain only what is
    orthogonal to X[:, c] or x2.

    Parameters
    ----------
    x1: numpy.ndarray of shape (n x m)
        feature matrix to orthogonalize
    c: int, less than m, default=None
       index of the column to orthogonalize by
    x2: numpy.ndarray of shape (n x a), default=x1[:, c]
        a separate set of columns to orthogonalize with respect to
        Note: the orthogonalizer will work column-by-column in column-index order
    """
    if x2 is None and c is not None:
        cols = x1[:, [c]]
    elif x2.shape[0] == x1.shape[0]:
        cols = np.reshape(x2, (x1.shape[0], -1))
    else:
        raise ValueError(
            "You can only orthogonalize a matrix using a vector with the same number "
            f"of rows. Matrix X has {x1.shape[0]} rows, whereas the orthogonalizing "
            f"matrix has {x2.shape[0]} rows."
        )

    if copy:
        xnew = x1.copy()
    else:
        xnew = x1

    for i in range(cols.shape[-1]):
        col = cols[:, [i]]

        if np.linalg.norm(col) < tol:
            warnings.warn("Column vector contains only zeros.", stacklevel=1)
        else:
            col = np.divide(col, np.linalg.norm(col, axis=0))

        xnew -= (col @ (col.T @ xnew)).astype(xnew.dtype)

    return xnew


def Y_feature_orthogonalizer(y, X, tol=1e-12, copy=True):
    r"""Orthogonalizes a property matrix given the selected features in
    :math:`\mathbf{X}`.

    .. math::
        \mathbf{Y} \leftarrow \mathbf{Y} -
        \mathbf{X} \left(\mathbf{X}^T\mathbf{X}\right)^{-1}\mathbf{X}^T \mathbf{Y}

    Parameters
    ----------
    y : numpy.ndarray of shape (n_samples x n_properties)
       property matrix
    X : numpy.ndarray of shape (n_samples x n_features)
       feature matrix
    tol: float
        cutoff for small eigenvalues to send to np.linalg.pinv
    copy: bool
        whether to return a copy of y or edit in-place, default=True
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
    r"""Orthogonalizes a matrix of targets :math:`{\mathbf{Y}}` given a reference
    feature matrix :math:`{\mathbf{X}_r}` and reference target matrix
    :math:`{\mathbf{Y}_r}`:

    .. math::
        \mathbf{Y} \leftarrow \mathbf{Y} -
        \mathbf{X} \left(\mathbf{X}_{\mathbf{r}}^T
        \mathbf{X}_{\mathbf{r}}\right)^{-1}\mathbf{X}_{\mathbf{r}}^T
        \mathbf{Y}_{\mathbf{r}}

    Parameters
    ----------
    y : numpy.ndarray of shape (n_samples x n_properties)
       property matrix
    X : numpy.ndarray of shape (n_samples x n_features)
       feature matrix
    y_ref : numpy.ndarray of shape (n_ref x n_properties)
        reference property matrix
    X_ref : numpy.ndarray of shape (n_ref x n_features)
        reference feature matrix
    tol: float
        cutoff for small eigenvalues to send to np.linalg.pinv
    copy: bool
        whether to return a copy of y or edit in-place, default=True
    """
    y_frag = (X @ (np.linalg.lstsq(X_ref, y_ref, rcond=tol)[0])).reshape(y.shape)

    if copy:
        return y.copy() - y_frag
    else:
        y -= y_frag
        return y
