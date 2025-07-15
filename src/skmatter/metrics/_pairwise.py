from typing import Union

import numpy as np
from sklearn.metrics.pairwise import _euclidean_distances, check_pairwise_arrays


def periodic_pairwise_euclidean_distances(
    X,
    Y=None,
    *,
    squared=False,
    cell_length=None,
):
    r"""
    Compute the pairwise distance matrix between each pair from a vector array X and Y.

    .. math::
        d_{i, j} = \\sqrt{\\sum_{k=1}^n (x_{i, k} - y_{j, k})^2}

    For efficiency reasons, the euclidean distance between a pair of row
    vector x and y is computed as::

        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

    This formulation has two advantages over other ways of computing distances. First,
    it is computationally efficient when dealing with sparse data. Second, if one
    argument varies but the other remains unchanged, then `dot(x, x)` and/or `dot(y, y)`
    can be pre-computed.

    However, this is not the most precise way of doing this computation, because this
    equation potentially suffers from "catastrophic cancellation". Also, the distance
    matrix returned by this function may not be exactly symmetric as required by, e.g.,
    ``scipy.spatial.distance`` functions.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_components)
        An array where each row is a sample and each column is a component.
    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_components), \
            default=None
        An array where each row is a sample and each column is a component.
        If `None`, method uses `Y=X`.
    cell_length : array-like of shape (n_components,), default=None
        The side length of rectangular cell used for periodic boundary conditions.
        `None` for non-periodic boundary conditions.

        .. note::
            Only side lengths of rectangular cells are supported.
            Cell format: `[side_length_1, ..., side_length_n]`

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        Returns the distances between the row vectors of `X`
        and the row vectors of `Y`.

    Examples
    --------
    >>> import numpy as np
    >>> from skmatter.metrics import periodic_pairwise_euclidean_distances
    >>> X = np.array([[0, 1], [1, 1]])
    >>> origin = np.array([[0, 0]])
    >>> # distance between rows of X
    >>> periodic_pairwise_euclidean_distances(X, X)
    array([[0., 1.],
           [1., 0.]])
    >>> # get distance to origin
    >>> periodic_pairwise_euclidean_distances(X, origin, cell_length=[0.5, 0.7])
    array([[0.3],
           [0.3]])
    """
    _check_dimension(X, cell_length)
    X, Y = check_pairwise_arrays(X, Y)

    if cell_length is None:
        return _euclidean_distances(X, Y, squared=squared)
    else:
        return _periodic_euclidean_distances(X, Y, squared=squared, cell=cell_length)


def _periodic_euclidean_distances(X, Y=None, *, squared=False, cell=None):
    X, Y = np.array(X).astype(float), np.array(Y).astype(float)
    XY = np.concatenate([x - Y for x in X])
    XY -= np.round(XY / cell) * cell
    distance = np.linalg.norm(XY, axis=1).reshape(X.shape[0], Y.shape[0])
    if squared:
        distance **= 2
    return distance


def pairwise_mahalanobis_distances(
    X: np.ndarray,
    Y: np.ndarray,
    cov_inv: np.ndarray,
    cell_length: Union[np.ndarray, None] = None,
    squared: bool = False,
):
    r"""
    Calculate the pairwise Mahalanobis distance between two arrays.

    This metric is used for calculating the distances between observations from Gaussian
    distributions. It is defined as:

    .. math::
        d_{\Sigma}(x, y)^2 = (x - y)^T \Sigma^{-1} (x - y)

    where :math:`\Sigma` is the covariance matrix, :math:`x` and :math:`y` are
    observations from the same distribution.

    Parameters
    ----------
        X : numpy.ndarray of shape (n_samples_X, n_components)
            An array where each row is a sample and each column is a component.
        Y : np.ndarray of shape (n_samples_Y, n_components)
            An array where each row is a sample and each column is a component.
        cov_inv : np.ndarray
            The inverse covariance matrix of shape (n_components, n_components).
        cell_length : np.ndarray, optinal, default=None
            The cell size for periodic boundary conditions.
            None for non-periodic boundary conditions.

            .. note::
                Only cubic cells are supported.
                Cell format: `[side_length_1, ..., side_length_n]`

        squared : bool, default=False
            Whether to return the squared distance.

    Returns
    -------
    np.ndarray
        The pairwise Mahalanobis distance between the two input arrays,
        of shape `(cov_inv.shape[0], x.shape[0], y.shape[0])`.

    Examples
    --------
    >>> import numpy as np
    >>> from skmatter.metrics import pairwise_mahalanobis_distances
    >>> iv = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    >>> X = np.array([[1, 0, 0], [0, 2, 0], [2, 0, 0]])
    >>> Y = np.array([[0, 1, 0]])
    >>> pairwise_mahalanobis_distances(X, Y, iv)
    array([[[1.        ],
            [1.        ],
            [1.73205081]]])
    """

    def _mahalanobis(
        cell: np.ndarray, X: np.ndarray, Y: np.ndarray, cov_inv: np.ndarray
    ):
        XY = np.concatenate([x - Y for x in X])
        if cell is not None:
            XY -= np.round(XY / cell) * cell

        return np.sum(XY * np.transpose(cov_inv @ XY.T, (0, 2, 1)), axis=-1).reshape(
            (cov_inv.shape[0], X.shape[0], Y.shape[0])
        )

    _check_dimension(X, cell_length)
    X, Y = check_pairwise_arrays(X, Y)
    if len(cov_inv.shape) == 2:
        cov_inv = cov_inv[np.newaxis, :, :]
    dists = _mahalanobis(cell_length, X, Y, cov_inv)
    if not squared:
        dists **= 0.5
    return dists


def _check_dimension(X, cell_length):
    if (cell_length is not None) and (X.shape[1] != len(cell_length)):
        raise ValueError("Cell dimension does not match the data dimension.")
