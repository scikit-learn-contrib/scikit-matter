"""This file holds utility functions and classes for the sparse KDE."""

from typing import Union

import numpy as np
from tqdm import tqdm

from ..metrics import DIST_METRICS


class NearestGridAssigner:
    """Assign descriptor to its nearest grid. This is an auxilirary class.

    Parameters
    ----------
    metric :
        The metric to use.
        Currently only `sklearn.metrics.pairwise.pairwise_euclidean_distances`.
    cell : np.ndarray
        An array of periods for each dimension of the grid.


    Attributes
    ----------
    grid_pos : np.ndarray
        An array of grid positions.
    grid_npoints : np.ndarray
        An array of number of points in each grid.
    grid_weight : np.ndarray
        An array of weights in each grid.
    grid_neighbour : dict
        A dictionary of neighbor lists for each grid.
    labels_ : np.ndarray
        An array of labels for each descriptor.
    """

    def __init__(
        self,
        metric,
        metric_params: Union[dict, None] = None,
    ) -> None:

        self.labels_ = None
        self.metric = metric
        self.metric_params = metric_params
        if isinstance(self.metric_params, dict):
            self.cell = self.metric_params["cell"]
        else:
            self.cell = None
        self.grid_pos = None
        self.grid_npoints = None
        self.grid_weight = None
        self.grid_neighbour = None

    def fit(self, X: np.ndarray, y: Union[np.ndarray, None] = None) -> None:
        """Fit the data.

        Parameters
        ----------
            X : np.ndarray
                An array of grid positions.
            y : np.ndarray, optional, default=None
                Igonred.
        """

        ngrid = len(X)
        self.grid_pos = X
        self.grid_npoints = np.zeros(ngrid, dtype=int)
        self.grid_weight = np.zeros(ngrid, dtype=float)
        self.grid_neighbour = {i: [] for i in range(ngrid)}

    def predict(
        self,
        X: np.ndarray,
        y: Union[np.ndarray, None] = None,
        sample_weight: Union[np.ndarray, None] = None,
    ) -> np.ndarray:
        """
        Predicts labels for input data and returns an array of labels.

        Parameters
        ----------
        X : np.ndarray
            Input data to predict labels for.
        y : np.ndarray, optional, default=None
            Igonred.
        sample_weight : np.ndarray, optional
            Sample weights for each data point.

        Returns
        -------
        np.ndarray
            Array of predicted labels.
        """
        if sample_weight is None:
            sample_weight = np.ones(len(X)) / len(X)
        self.labels_ = []
        for i, point in tqdm(
            enumerate(X), desc="Assigning samples to grids...", total=len(X)
        ):
            descriptor2grid = DIST_METRICS[self.metric](
                X=point.reshape(1, -1), Y=self.grid_pos, cell=self.cell
            )
            self.labels_.append(np.argmin(descriptor2grid))
            self.grid_npoints[self.labels_[-1]] += 1
            self.grid_weight[self.labels_[-1]] += sample_weight[i]
            self.grid_neighbour[self.labels_[-1]].append(i)

        for key in self.grid_neighbour:
            self.grid_neighbour[key] = np.array(self.grid_neighbour[key])

        return self.labels_


def covariance(X: np.ndarray, sample_weights: np.ndarray, cell: np.ndarray):
    """
    Calculate the covariance matrix for a given set of grid positions and weights.

    Parameters
    ----------
    X : np.ndarray
        An array of shape (nsample, dimension) representing the grid positions.
    sample_weights : np.ndarray
        An array of shape (nsample,) representing the weights of the grid positions.
    cell : np.ndarray
        An array of shape (dimension,) representing the periodicity of each dimension.
    Returns
    -------
    np.ndarray
        The covariance matrix of shape (dimension, dimension).
    Notes
    -----
    The function assumes that the grid positions, weights,
    and total weight are provided correctly.
    The function handles periodic and non-periodic dimensions differently to
    calculate the covariance matrix.
    """

    totw = np.sum(sample_weights)

    if cell is None:
        xm = np.average(X, axis=0, weights=sample_weights / totw)
    else:
        sumsin = np.average(
            np.sin(X) * (2 * np.pi) / cell,
            axis=0,
            weights=sample_weights / totw,
        )
        sumcos = np.average(
            np.cos(X) * (2 * np.pi) / cell,
            axis=0,
            weights=sample_weights / totw,
        )
        xm = np.arctan2(sumsin, sumcos)

    xxm = X - xm
    if cell is not None:
        xxm -= np.round(xxm / cell) * cell
    xxmw = xxm * sample_weights.reshape(-1, 1) / totw
    cov = xxmw.T.dot(xxm)
    cov /= 1 - sum((sample_weights / totw) ** 2)

    return cov


def local_population(
    cell: np.ndarray,
    grid_pos: np.ndarray,
    target_grid_pos: np.ndarray,
    grid_weight: np.ndarray,
    s2: float,
):
    """
    Calculates the local population of a set of vectors in a grid.

    Parameters
    ----------
    cell : np.ndarray
        An array of periods for each dimension of the grid.
    grid_pos : np.ndarray
        An array of vectors to be localized.
    target_grid_pos : np.ndarray
        An array of target vectors representing the grid.
    grid_weight : np.ndarray
        An array of weights for each target vector.
    s2 : float
        The scaling factor for the squared distance.


    Returns
    -------
    tuple
        A tuple containing two numpy arrays:
        wl : np.ndarray
            An array of localized weights for each vector.
        num : np.ndarray
            The sum of the localized weights.

    """

    xy = grid_pos - target_grid_pos
    if cell is not None:
        xy -= np.round(xy / cell) * cell

    wl = np.exp(-0.5 / s2 * np.sum(xy**2, axis=1)) * grid_weight
    num = np.sum(wl)

    return wl, num


def effdim(cov):
    """
    Calculate the effective dimension of a covariance matrix based on Shannon entropy.

    Parameters
    ----------
    cov : ndarray
        The covariance matrix.

    Returns
    -------
    float
        The effective dimension of the covariance matrix.

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
    cov : np.ndarray
        A covariance matrix
    n : float
        The local population
    D : int
        Dimension

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
