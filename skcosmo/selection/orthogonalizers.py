import numpy as np


def feature_orthogonalizer(idx, X_proxy, Y_proxy, tol=1e-12):
    """
    Orthogonalizes two matrices, meant to represent a feature matrix
    :math:`{\\mathbf{X}}` and a property matrix :math:`{\\mathbf{Y}}`, given
    the selected features :math:`{c}`

    Update to :math:`{\\mathbf{X}}`, where :math:`{\\mathbf{X}_{c}}` is a column
    vector containing the most recently-chosen feature:

    .. math::

        \\mathbf{X} \\leftarrow \\mathbf{X} -
        \\left(\\frac{\\mathbf{X}_{c}\\mathbf{X}{c}^T}
        {\\lVert\\mathbf{X}_{c}\\rVert^2}\\right)\\mathbf{X}

    Update to :math:`{\\mathbf{Y}}`, where :math:`{\\mathbf{X}_{\\mathbf{c}}}`
    is a matrix containing all previously-chosen features:

    .. math::
        \\mathbf{Y} \\leftarrow \\mathbf{Y} -
        \\mathbf{X}_{\\mathbf{c}} \\left(\\mathbf{X}_{\\mathbf{c}}^T
        \\mathbf{X}_{\\mathbf{c}}\\right)^{-1}
        \\mathbf{X}_{\\mathbf{c}}^T \\mathbf{Y}

    :param idx: indices of selected features
    :type idx: list of int

    :param X_proxy: feature matrix
    :type X_proxy: array of shape (n_samples x n_features)

    :param Y_proxy: property matrix
    :type Y_proxy: array of shape (n_samples x n_properties)

    :param tol: cutoff for small eigenvalues to send to np.linalg.pinv
    :type tol: float

    """
    if X_proxy is not None:
        Aci = X_proxy[:, idx]

        if Y_proxy is not None:
            v = np.linalg.pinv(np.matmul(Aci.T, Aci), rcond=tol)
            v = np.matmul(Aci, v)
            v = np.matmul(v, Aci.T)

            Y_proxy -= np.matmul(v, Y_proxy)

        v = X_proxy[:, idx[-1]] / np.sqrt(
            np.matmul(X_proxy[:, idx[-1]], X_proxy[:, idx[-1]])
        )

        for i in range(X_proxy.shape[1]):
            X_proxy[:, i] -= v * np.dot(v, X_proxy[:, i])
    return X_proxy, Y_proxy


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

        Ajnorm = np.dot(X_proxy[idx[-1]], X_proxy[idx[-1]])
        for i in range(X_proxy.shape[0]):
            X_proxy[i] -= (np.dot(X_proxy[i], X_proxy[idx[-1]]) / Ajnorm) * X_proxy[
                idx[-1]
            ]
    return X_proxy, Y_proxy
