"""This file holds utility functions and classes for the sparse KDE."""

import numpy as np


def effdim(cov):
    """
    Calculate the effective dimension of a covariance matrix based on Shannon entropy.

    Parameters
    ----------
    cov : numpy.ndarray
        The covariance matrix.

    Returns
    -------
    float
        The effective dimension of the covariance matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from skmatter.utils import effdim
    >>> cov = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    >>> round(effdim(cov), 3)
    1.89

    References
    ----------
    https://ieeexplore.ieee.org/document/7098875
    """

    eigval = np.linalg.eigvals(cov)
    eigval /= sum(eigval)
    eigval *= np.log(eigval)
    eigval[np.isnan(eigval)] = 0.0

    return np.exp(-sum(eigval))


def oas(cov: np.ndarray, n: float, D: int) -> np.ndarray:
    """
    Oracle approximating shrinkage (OAS) estimator

    Parameters
    ----------
    cov : numpy.ndarray
        A covariance matrix
    n : float
        The local population
    D : int
        Dimension

    Examples
    --------
    >>> import numpy as np
    >>> from skmatter.utils import oas
    >>> cov = np.array([[0.5, 1.0], [0.7, 0.4]])
    >>> oas(cov, 10, 2)
    array([[0.48903924, 0.78078484],
           [0.54654939, 0.41096076]])

    Returns
    -------
    np.ndarray
        Covariance matrix
    """

    tr = np.trace(cov)
    tr2 = tr**2
    tr_cov2 = np.trace(cov**2)
    phi = ((1 - 2 / D) * tr_cov2 + tr2) / ((n + 1 - 2 / D) * tr_cov2 - tr2 / D)

    return (1 - phi) * cov + phi * np.eye(D) * tr / D
