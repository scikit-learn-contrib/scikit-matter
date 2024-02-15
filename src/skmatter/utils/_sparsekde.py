"""This file holds utility functions and classes for the sparse KDE."""

from dataclasses import dataclass
from typing import Union

import numpy as np
from tqdm import tqdm


class NearestGridAssigner:
    """NearestGridAssigner Class
    Assign descriptor to its nearest grid. This is an auxilirary class.


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
        cell: Union[np.ndarray, None] = None,
    ) -> None:

        self.labels_ = None
        self.metric = metric
        self.cell = cell
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
            descriptor2grid = self.metric(
                X=point.reshape(1, -1), Y=self.grid_pos, cell=self.cell
            )
            self.labels_.append(np.argmin(descriptor2grid))
            self.grid_npoints[self.labels_[-1]] += 1
            self.grid_weight[self.labels_[-1]] += sample_weight[i]
            self.grid_neighbour[self.labels_[-1]].append(i)

        for key in self.grid_neighbour:
            self.grid_neighbour[key] = np.array(self.grid_neighbour[key])

        return self.labels_


@dataclass
class GaussianMixtureModel:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    cell: Union[np.ndarray, None] = None
    """
    A Gaussian Mixture Model for clustering and density estimation.

    Parameters
    ----------
    weights : np.ndarray
        The weights of the Gaussian components.
    means : np.ndarray
        The means of the Gaussian components.
    covariances : np.ndarray
        The covariance matrices of the Gaussian components.
    cell : Optional[np.ndarray], optional
        The cell size for periodic boundary conditions. Default is None.

    Attributes
    ----------
    dimension : int
        The dimension of the Gaussian Mixture Model.
    cov_invs : np.ndarray
        The inverses of the covariance matrices.
    cov_dets : np.ndarray
        The determinants of the covariance matrices.
    norms : np.ndarray
        The normalization constants.

    """

    def __post_init__(self):
        self.dimension = self.means.shape[1]
        self.cov_invs = np.linalg.inv(self.covariances)
        self.cov_dets = np.linalg.det(self.covariances)
        self.norms = 1 / np.sqrt((2 * np.pi) ** self.dimension * self.cov_dets)

    def __call__(self, x: np.ndarray, i: Union[int, list[int], None] = None):
        """
        Calculate the probability density function (PDF) value for a given input array.

        Parameters
        ----------
        x : np.ndarray
            The input array for which the PDF is calculated. Once a point.
        i : int, list[int], optional, default=None
            The index of the element inthe PDF array to return.
            If None, the sum of all elements is returned.

        Returns
        -------
        float or np.ndarray
            The PDF value(s) for the given input(s).
            If i is None, the sum of all PDF values is returned.
            If i is specified, the normalized value of the corresponding
            gaussian is returned.
        """

        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        if self.cell is not None:
            xij = np.zeros(self.means.shape)
            xij = rij(self.cell, x, self.means)
        else:
            xij = x - self.means
        p = (
            self.weights
            * self.norms
            * np.exp(
                -0.5 * (xij[:, np.newaxis, :] @ self.cov_invs @ xij[:, :, np.newaxis])
            ).reshape(-1)
        )
        sum_p = np.sum(p)
        if i is None:
            return sum_p

        return np.sum(p[i]) / sum_p


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


def quick_shift(
    probs: np.ndarray,
    dist_matrix: np.ndarray,
    cutoff2: np.ndarray,
    gs: float,
):
    """
    Perform quick shift clustering on the given probability array and distance matrix.

    Parameters
    ----------
    probs : np.ndarray
        The log-likelihood of each sample.
    dist_matrix : np.ndarray
        The squared distance matrix.
    cutoff2 : np.ndarray
        The squared cutoff array.
    gs : float
        The value of gs.

    Returns
    -------
    tuple
        A tuple containing the cluster centers and the root indices.
    """

    def gs_next(
        idx: int,
        probs: np.ndarray,
        n_shells: int,
        distmm: np.ndarray,
        gabriel: np.ndarray,
    ):
        """Find next cluster in Gabriel graph."""

        ngrid = len(probs)
        neighs = np.copy(gabriel[idx])
        for _ in range(1, n_shells):
            nneighs = np.full(ngrid, False)
            for j in range(ngrid):
                if neighs[j]:
                    # j can be accessed from idx
                    # j's neighbors can also be accessed from idx
                    nneighs |= gabriel[j]
            neighs |= nneighs

        next_idx = idx
        dmin = np.inf
        for j in range(ngrid):
            if probs[j] > probs[idx] and distmm[idx, j] < dmin and neighs[j]:
                # find the closest neighbor
                next_idx = j
                dmin = distmm[idx, j]

        return next_idx

    def qs_next(
        idx: int, idxn: int, probs: np.ndarray, distmm: np.ndarray, cutoff: float
    ):
        """Find next cluster with respect to cutoff."""

        ngrid = len(probs)
        dmin = np.inf
        next_idx = idx
        if probs[idxn] > probs[idx]:
            next_idx = idxn
        for j in range(ngrid):
            if probs[j] > probs[idx] and distmm[idx, j] < min(dmin, cutoff):
                next_idx = j
                dmin = distmm[idx, j]

        return next_idx

    gabrial = get_gabriel_graph(dist_matrix)
    idmindist = np.argmin(dist_matrix, axis=1)
    idxroot = np.full(dist_matrix.shape[0], -1, dtype=int)
    for i in tqdm(range(dist_matrix.shape[0]), desc="Quick-Shift"):
        if idxroot[i] != -1:
            continue
        qspath = []
        qspath.append(i)
        while qspath[-1] != idxroot[qspath[-1]]:
            if gs > 0:
                idxroot[qspath[-1]] = gs_next(
                    qspath[-1], probs, gs, dist_matrix, gabrial
                )
            else:
                idxroot[qspath[-1]] = qs_next(
                    qspath[-1],
                    idmindist[qspath[-1]],
                    probs,
                    dist_matrix,
                    cutoff2[qspath[-1]],
                )
            if idxroot[idxroot[qspath[-1]]] != -1:
                break
            qspath.append(idxroot[qspath[-1]])
        idxroot[qspath] = idxroot[idxroot[qspath[-1]]]
    cluster_centers = np.concatenate(
        np.argwhere(idxroot == np.arange(dist_matrix.shape[0]))
    )

    return cluster_centers, idxroot


def get_gabriel_graph(dist_matrix2: np.ndarray):
    """
    Generate the Gabriel graph based on the given squared distance matrix.

    Parameters
    ----------
    dist_matrix2 : np.ndarray
        The squared distance matrix of shape (n_points, n_points).

    Returns
    -------
    np.ndarray
        The Gabriel graph matrix of shape (n_points, n_points).
    """

    n_points = dist_matrix2.shape[0]
    gabriel = np.full((n_points, n_points), True)
    for i in tqdm(range(n_points), desc="Calculating Gabriel graph"):
        gabriel[i, i] = False
        for j in range(i, n_points):
            if np.sum(dist_matrix2[i] + dist_matrix2[j] < dist_matrix2[i, j]):
                gabriel[i, j] = False
                gabriel[j, i] = False

    return gabriel


def rij(period: Union[np.ndarray, None], xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
    """
    Calculate the position vector considering the periodic boundary conditions.

    Parameters
    ----------
    period : np.ndarray, Optional
        An array of periods for each dimension of the points.
        None stands for not periodic.
    xi : np.ndarray
        An array of point coordinates. It can also contain many points.
        Shape: (n_points, n_dimensions)
    xj : np.ndarray
        An array of point coordinates. It can only contain one point.

    Returns
    -------
    np.ndarray
        An array of position vectors. Shape: (n_points, n_dimensions)
    """

    xij = xi - xj
    if period is not None:
        xij -= np.round(xij / period) * period

    return xij
