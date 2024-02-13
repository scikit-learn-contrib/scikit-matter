from typing import Union
import numpy as np

from sklearn.metrics.pairwise import (
    check_pairwise_arrays,
    check_array,
    _euclidean_distances,
)


def pairwise_euclidean_distances(
    X, Y=None, *, Y_norm_squared=None, squared=False, X_norm_squared=None, cell=None
):
    """
    Compute the pairwise distance matrix between each pair from a vector array X and Y.

    For efficiency reasons, the euclidean distance between a pair of row
    vector x and y is computed as::

        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

    This formulation has two advantages over other ways of computing distances.
    First, it is computationally efficient when dealing with sparse data.
    Second, if one argument varies but the other remains unchanged, then
    `dot(x, x)` and/or `dot(y, y)` can be pre-computed.

    However, this is not the most precise way of doing this computation,
    because this equation potentially suffers from "catastrophic cancellation".
    Also, the distance matrix returned by this function may not be exactly
    symmetric as required by, e.g., ``scipy.spatial.distance`` functions.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        An array where each row is a sample and each column is a feature.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), \
            default=None
        An array where each row is a sample and each column is a feature.
        If `None`, method uses `Y=X`.

    Y_norm_squared : array-like of shape (n_samples_Y,) or (n_samples_Y, 1) \
            or (1, n_samples_Y), default=None
        Pre-computed dot-products of vectors in Y (e.g.,
        ``(Y**2).sum(axis=1)``)
        May be ignored in some cases, see the note below.

    squared : bool, default=False
        Return squared Euclidean distances.

    X_norm_squared : array-like of shape (n_samples_X,) or (n_samples_X, 1) \
            or (1, n_samples_X), default=None
        Pre-computed dot-products of vectors in X (e.g.,
        ``(X**2).sum(axis=1)``)
        May be ignored in some cases, see the note below.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        Returns the distances between the row vectors of `X`
        and the row vectors of `Y`.

    See Also
    --------
    paired_distances : Distances between pairs of elements of X and Y.

    Notes
    -----
    To achieve a better accuracy, `X_norm_squared` and `Y_norm_squared` may be
    unused if they are passed as `np.float32`.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> X = [[0, 1], [1, 1]]
    >>> # distance between rows of X
    >>> euclidean_distances(X, X)
    array([[0., 1.],
           [1., 0.]])
    >>> # get distance to origin
    >>> euclidean_distances(X, [[0, 0]])
    array([[1.        ],
           [1.41421356]])
    """

    X, Y = check_pairwise_arrays(X, Y)

    if X_norm_squared is not None:
        X_norm_squared = check_array(X_norm_squared, ensure_2d=False)
        original_shape = X_norm_squared.shape
        if X_norm_squared.shape == (X.shape[0],):
            X_norm_squared = X_norm_squared.reshape(-1, 1)
        if X_norm_squared.shape == (1, X.shape[0]):
            X_norm_squared = X_norm_squared.T
        if X_norm_squared.shape != (X.shape[0], 1):
            raise ValueError(
                f"Incompatible dimensions for X of shape {X.shape} and "
                f"X_norm_squared of shape {original_shape}."
            )

    if Y_norm_squared is not None:
        Y_norm_squared = check_array(Y_norm_squared, ensure_2d=False)
        original_shape = Y_norm_squared.shape
        if Y_norm_squared.shape == (Y.shape[0],):
            Y_norm_squared = Y_norm_squared.reshape(1, -1)
        if Y_norm_squared.shape == (Y.shape[0], 1):
            Y_norm_squared = Y_norm_squared.T
        if Y_norm_squared.shape != (1, Y.shape[0]):
            raise ValueError(
                f"Incompatible dimensions for Y of shape {Y.shape} and "
                f"Y_norm_squared of shape {original_shape}."
            )

    if cell is None:
        return _euclidean_distances(X, Y, X_norm_squared, Y_norm_squared, squared)
    else:
        return _periodic_euclidean_distances(X, Y, squared=squared, cell=cell)


def _periodic_euclidean_distances(X, Y=None, *, squared=False, cell=None):
    X, Y = np.array(X).astype(float), np.array(Y).astype(float)
    XY = np.concatenate([X - y for y in Y])
    XY -= np.round(XY / cell) * cell
    distance = np.linalg.norm(XY, axis=1).reshape(X.shape[0], Y.shape[0], order="F")
    if squared:
        distance **= 2
    return distance


def pairwise_mahalanobis_distance(
    X: np.ndarray,
    Y: np.ndarray,
    cov_inv: np.ndarray,
    cell: Union[np.ndarray, None] = None,
    squared: bool = False,
):
    """
    Calculate the pairwise Mahalanobis distance between two arrays.

    Args:
        x (np.ndarray): The first input array.
        y (np.ndarray): The second input array.
        cov_inv (np.ndarray): The inverse covariance matrix.
        cell (Union[np.ndarray, None]): The cell size for periodic boundary conditions.
        squared (bool): Whether to return the squared distance.

    Returns:
        np.ndarray: The pairwise Mahalanobis distance between the two input arrays,
            of shape (cov_inv.shape[0], x.shape[0], y.shape[0]).
    """

    def _mahalanobis_preprocess(cov_inv: np.ndarray):

        if len(cov_inv.shape) == 2:
            cov_inv = cov_inv[np.newaxis, :, :]

        return cov_inv

    def _mahalanobis(
        cell: np.ndarray, X: np.ndarray, Y: np.ndarray, cov_inv: np.ndarray
    ):

        XY = np.concatenate([X - y for y in Y])
        if cell is not None:
            XY -= np.round(XY / cell) * cell

        return np.sum(XY * np.transpose(cov_inv @ XY.T, (0, 2, 1)), axis=-1).reshape(
            (cov_inv.shape[0], X.shape[0], Y.shape[0]), order="F"
        )

    X, Y = check_pairwise_arrays(X, Y)
    cov_inv = _mahalanobis_preprocess(cov_inv)
    dists = _mahalanobis(cell, X, Y, cov_inv)
    if not squared:
        dists **= 0.5
    return dists
