import warnings
from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp as LSE
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_random_state
from tqdm import tqdm

from ..metrics._pairwise import (
    pairwise_mahalanobis_distances,
    periodic_pairwise_euclidean_distances,
)
from ..utils._sparsekde import effdim, oas


class SparseKDE(BaseEstimator):
    """A sparse implementation of the Kernel Density Estimation.
    This class is used to build a sparse kernel density estimator.
    It takes a set of descriptors and a set of weights as input,
    and fit the KDE model on the sampled point (e.g. the grid point
    selected by FPS). First, the probability density is estimated for
    each sampled point. Then, quick shift clustering is applied to the
    grid points. Finally, a kernel density estimator is built based on
    the clustering results.

    .. note::
        Currently only the Gaussian kernel is supported.

    Parameters
    ----------
    descriptors: numpy.ndarray
        Descriptors of the system where you want to build a sparse KDE.
        It should be an array of shape `(n_descriptors, n_features)`.
    weights: numpy.ndarray, default=None
        Weights of the descriptors.
        If None, all weights are set to `1/n_descriptors`.
    metric : Callable[[ArrayLike, ArrayLike, bool, dict], ArrayLike],
        default=:func:`skmatter.metrics.pairwise_euclidean_distances()`
        The metric to use. Your metric should be able to take at least three arguments
        in secquence: `X`, `Y`, and `squared=True`. Here, `X` and `Y` are two array-like
        of shape (n_samples, n_components). The return of the metric is an array-like of
        shape (n_samples, n_samples). If you want to use periodic boundary
        conditions, be sure to provide the cell size in the metric_params and
        provide a metric that can take the cell argument.
    metric_params : dict, default=None
        Additional parameters to be passed to the use of
        metric.  i.e. the cell dimension for
        :func:`skmatter.metrics.pairwise_euclidean_distances()`
        `{'cell_length': [side_length_1, ..., side_length_n]}`
    fspread : float, default=-1.0
        The fractional "space" occupied by the voronoi cell of each grid. Use this when
        each cell is of a similar size.
    fpoints : float, default=0.15
        The fractional number of points in the voronoi cell of each grid points. Use
        this when each cell has a similar number of points.
    verbose : bool, default=False
        Whether to print progress.


    Attributes
    ----------
    n_samples : int
        The number of descriptors.
    kdecut_squared : float
        The cut-off value for the KDE. If the mahalanobis distance between two grid
        points is larger than kdecut2, they are considered to be far away.
    cell : numpy.ndarray
        The cell dimension for the metric.
    bandwidth_: numpy.ndarray
        The bandwidth of the KDE.


    Examples
    --------
    >>> import numpy as np
    >>> from skmatter.neighbors import SparseKDE
    >>> from skmatter.feature_selection import FPS
    >>> np.random.seed(0)
    >>> n_samples = 10_000

    To create two gaussians with different means and covariance and sample from them

    >>> cov1 = [[1, 0.5], [0.5, 1]]
    >>> cov2 = [[1, 0.5], [0.5, 0.5]]
    >>> sample1 = np.random.multivariate_normal([0, 0], cov1, n_samples)
    >>> sample2 = np.random.multivariate_normal([4, 4], cov2, n_samples)
    >>> samples = np.concatenate([sample1, sample2])

    To select grid points using FPS

    >>> selector = FPS(n_to_select=int(np.sqrt(2 * n_samples)))
    >>> result = selector.fit_transform(samples.T).T

    Conduct sparse KDE based on the grid points

    >>> estimator = SparseKDE(samples, None, fpoints=0.5)
    >>> _ = estimator.fit(result)

    The total log-likelihood under the model

    >>> print(round(estimator.score(result), 3))
    -759.831
    """

    def __init__(
        self,
        descriptors: np.ndarray,
        weights: Union[np.ndarray, None] = None,
        metric: Callable[
            [ArrayLike, ArrayLike, bool, dict], ArrayLike
        ] = periodic_pairwise_euclidean_distances,
        metric_params: Union[dict, None] = None,
        fspread: float = -1.0,
        fpoints: float = 0.15,
        kernel: str = "gaussian",
        verbose: bool = False,
    ):
        self.metric_params = (
            metric_params if metric_params is not None else {"cell_length": None}
        )
        self.metric = lambda X, Y: metric(X, Y, squared=True, **self.metric_params)
        self.cell = metric_params["cell_length"] if metric_params is not None else None
        self._check_dimension(descriptors)
        self.descriptors = descriptors
        self.weights = weights if weights is not None else np.ones(len(descriptors))
        self.weights /= np.sum(self.weights)
        self.fspread = fspread
        self.fpoints = fpoints
        self.kernel = kernel
        if self.kernel != "gaussian":
            raise NotImplementedError
        self.verbose = verbose

        if self.fspread > 0:
            self.fpoints = -1.0

        self.bandwidth_ = None
        self._covariance = None

    @property
    def nsamples(self):
        if not hasattr(self, "__nsamples"):
            self.__nsamples = len(self.descriptors)
        return self.__nsamples

    @property
    def ndimension(self):
        return self.descriptors.shape[1]

    @property
    def kdecut_squared(self):
        if not hasattr(self, "__kdecut_squared"):
            self.__kdecut_squared = (3 * (np.sqrt(self.descriptors.shape[1]) + 1)) ** 2
        return self.__kdecut_squared

    @property
    def _bandwidth_inv(self):
        if self.fitted_:
            if not hasattr(self, "__bandwidth_inv"):
                self.__bandwidth_inv = np.array(
                    [np.linalg.inv(h) for h in self.bandwidth_]
                )
        else:
            raise ValueError("The model is not fitted yet.")
        return self.__bandwidth_inv

    @property
    def _normkernels(self):
        if not hasattr(self, "__normkernels"):
            self.__normkernels = np.array(
                [
                    self.ndimension * np.log(2 * np.pi) + np.linalg.slogdet(h)[1]
                    for h in self.bandwidth_
                ]
            )
        return self.__normkernels

    def fit(self, X, y=None, sample_weight=None):
        """Fit the Kernel Density model on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        sample_weight : array-like of shape (n_samples,), default=None
            List of sample weights attached to the data X. This parameter
            is ignored. Instead of reading sample_weight from the input,
            it is calculated internally.


        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._check_dimension(X)
        self._grids = X
        grid_dist_mat = self.metric(X, X)
        np.fill_diagonal(grid_dist_mat, np.inf)
        min_grid_dist = np.min(grid_dist_mat, axis=1)
        _, self._grid_neighbour, self._sample_labels_, self._sample_weights = (
            self._assign_descriptors_to_grids(X)
        )
        self._computes_localized_bandwidth(X, self._sample_weights, min_grid_dist)

        self.fitted_ = True

        return self

    def score_samples(self, X):
        """Compute the log-likelihood of each sample under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).

        Returns
        -------
        density : ndarray of shape (n_samples,)
            Log-likelihood of each sample in `X`. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        return self._computes_kernel_density_estimation(X)

    def score(self, X, y=None):
        """Compute the total log-likelihood under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        logprob : float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        """
        return np.sum(self.score_samples(X))

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        random_state : int, RandomState instance or None, default=None
            Determines random number generation used to generate
            random samples. Pass an int for reproducible results
            across multiple function calls.
            See :term:`Glossary <random_state>`.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            List of samples.
        """
        check_is_fitted(self)
        rng = check_random_state(random_state)
        u = rng.uniform(0, 1, size=n_samples)
        cumsum_weight = np.cumsum(np.asarray(self._sample_weights))
        sum_weight = cumsum_weight[-1]
        idxs = np.searchsorted(cumsum_weight, u * sum_weight)

        return np.concatenate(
            [
                np.atleast_2d(
                    rng.multivariate_normal(self._grids[i], self.bandwidth_[i])
                )
                for i in idxs
            ]
        )

    def _check_dimension(self, X):
        if (self.cell is not None) and (X.shape[1] != len(self.cell)):
            raise ValueError("Cell dimension does not match the data dimension.")

    def _assign_descriptors_to_grids(self, X):

        assigner = _NearestGridAssigner(self.metric, self.metric_params, self.verbose)
        assigner.fit(X)
        labels = assigner.predict(self.descriptors, sample_weight=self.weights)
        grid_npoints = assigner.grid_npoints
        grid_neighbour = assigner.grid_neighbour

        return grid_npoints, grid_neighbour, labels, assigner.grid_weight

    def _computes_localized_bandwidth(
        self, X, sample_weights: np.ndarray, mindist: np.ndarray
    ):
        """Compute the localized bandwidth of the kernel density estimator
        on grid points.
        """
        # estimate the sigma
        cov = _covariance(X, sample_weights, self.cell)
        if self.cell is not None:
            tune = sum(self.cell**2)
        else:
            tune = np.trace(cov)
        sigma2 = np.full(len(X), tune, dtype=float)

        # initialize the localization based on fraction of data spread
        if self.fspread > 0:
            sigma2 *= self.fspread**2
        flocal = np.zeros(len(X))
        self.bandwidth_ = np.zeros((len(X), X.shape[1], X.shape[1]))
        self._covariance = np.zeros((len(X), X.shape[1], X.shape[1]))

        for i in tqdm(
            range(len(X)),
            desc="Estimating kernel density bandwidths",
            disable=not self.verbose,
        ):
            wlocal, flocal[i] = _local_population(
                self.cell, X, X[i], sample_weights, sigma2[i]
            )
            if self.fpoints > 0:
                sigma2, flocal, wlocal = (
                    self._tune_localization_factor_based_on_fraction_of_points(
                        X, sample_weights, sigma2, flocal, i, 1 / self.nsamples, tune
                    )
                )
            elif sigma2[i] < flocal[i]:
                sigma2, flocal, wlocal = (
                    self._tune_localization_factor_based_on_fraction_of_spread(
                        X, sample_weights, sigma2, flocal, i, mindist
                    )
                )
            self.bandwidth_[i], self._covariance[i] = (
                self._bandwidth_estimation_from_localization(X, wlocal, flocal, i)
            )

    def _tune_localization_factor_based_on_fraction_of_points(
        self, X, sample_weights, sigma2, flocal, idx, delta, tune
    ):
        """Used in cases where one expects clusters with very different spreads,
        but similar populations
        """
        lim = self.fpoints
        if lim <= sample_weights[idx]:
            lim = sample_weights[idx] + delta
            warnings.warn(
                " Warning: localization smaller than voronoi,"
                " increase grid size (meanwhile adjusted localization)!",
                stacklevel=2,
            )
        while flocal[idx] < lim:
            sigma2[idx] += tune
            wlocal, flocal[idx] = _local_population(
                self.cell, X, X[idx], sample_weights, sigma2[idx]
            )
        j = 1
        while True:
            if flocal[idx] > lim:
                sigma2[idx] -= tune / 2**j
            else:
                sigma2[idx] += tune / 2**j
            wlocal, flocal[idx] = _local_population(
                self.cell, X, X[idx], sample_weights, sigma2[idx]
            )
            if abs(flocal[idx] - lim) < delta:
                break
            j += 1

        return sigma2, flocal, wlocal

    def _tune_localization_factor_based_on_fraction_of_spread(
        self, X, sample_weights, sigma2, flocal, idx, mindist
    ):
        """Used in cases where one expects the spatial extent of clusters to be
        relatively homogeneous
        """
        sigma2[idx] = mindist[idx]
        wlocal, flocal[idx] = _local_population(
            self.cell, self.descriptors, X, sample_weights, sigma2[idx]
        )

        return sigma2, flocal, wlocal

    def _bandwidth_estimation_from_localization(self, X, wlocal, flocal, idx):
        """Compute the bandwidth based on localized version of Silverman's rule"""
        cov = _covariance(X, wlocal, self.cell)
        nlocal = flocal[idx] * self.nsamples
        local_dimension = effdim(cov)
        cov = oas(cov, nlocal, X.shape[1])
        # localized version of Silverman's rule
        h = (4.0 / nlocal / (local_dimension + 2.0)) ** (
            2.0 / (local_dimension + 4.0)
        ) * cov

        return h, cov

    def _computes_kernel_density_estimation(self, X: np.ndarray):

        prob = np.full(len(X), -np.inf)
        dummd1s_mat = pairwise_mahalanobis_distances(
            X, self._grids, self._bandwidth_inv, self.cell, squared=True
        )
        for i in tqdm(
            range(len(X)),
            desc="Computing kernel density on reference points",
            disable=not self.verbose,
        ):
            for j, dummd1 in enumerate(np.diagonal(dummd1s_mat[:, i, :])):
                # The second point is the mean corresponding to the cov
                if dummd1 > self.kdecut_squared:
                    lnk = -0.5 * (self._normkernels[j] + dummd1) + np.log(
                        self._sample_weights[j]
                    )
                    prob[i] = LSE([prob[i], lnk])
                else:
                    neighbours = self._grid_neighbour[j][
                        np.any(
                            self.descriptors[self._grid_neighbour[j]] != X[i], axis=1
                        )
                    ]
                    if neighbours.size == 0:
                        continue
                    dummd1s = pairwise_mahalanobis_distances(
                        self.descriptors[neighbours],
                        X[i][np.newaxis, ...],
                        self._bandwidth_inv[j],
                        self.cell,
                        squared=True,
                    ).reshape(-1)
                    lnks = -0.5 * (self._normkernels[j] + dummd1s) + np.log(
                        self.weights[neighbours]
                    )
                    prob[i] = LSE(np.concatenate([[prob[i]], lnks]))

        prob -= np.log(np.sum(self._sample_weights))

        return prob


class _NearestGridAssigner:
    """Assign descriptor to its nearest grid. This is an auxilirary class.

    Parameters
    ----------
    metric :
        The metric to use.
        Currently only `sklearn.metrics.pairwise.pairwise_euclidean_distances`.
    metric_params : dict, default=None
        Additional parameters to be passed to the use of
        metric.  i.e. the cell dimension for ``periodic_euclidean``
        {'cell_length': [2, 2]}
    verbose : bool, default=False
        Whether to print progress.

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
        verbose: bool = False,
    ) -> None:

        self.labels_ = None
        self.metric = metric
        self.metric_params = metric_params
        self.verbose = verbose
        if isinstance(self.metric_params, dict):
            self.cell = self.metric_params["cell_length"]
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
            enumerate(X),
            desc="Assigning samples to grids...",
            total=len(X),
            disable=not self.verbose,
        ):
            descriptor2grid = self.metric(point.reshape(1, -1), self.grid_pos)
            self.labels_.append(np.argmin(descriptor2grid))
            self.grid_npoints[self.labels_[-1]] += 1
            self.grid_weight[self.labels_[-1]] += sample_weight[i]
            self.grid_neighbour[self.labels_[-1]].append(i)

        for key in self.grid_neighbour:
            self.grid_neighbour[key] = np.array(self.grid_neighbour[key])

        return self.labels_


def _covariance(X: np.ndarray, sample_weights: np.ndarray, cell: np.ndarray):
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


def _local_population(
    cell: np.ndarray,
    grid_j: np.ndarray,
    grid_i: np.ndarray,
    grid_j_weight: np.ndarray,
    sigma_squared: float,
):
    r"""
    Calculates the local population of a selected grid. The local population is defined
    as a sum of the weighting factors for each other grid arond it.

    .. math::
        N_i = \\sum_j u_{ij}

    where :math:`u_{ij}` is the weighting factor. The weighting factor is calculated
    from an spherical Gaussian

    .. math::
        u_{ij} = \\exp\\left[-\\frac{(x_i - x_j)^2}{2\\sigma^2}  \\right] N w_j /
        \\sum_j w_j

    where :math:`w_j` is the weighting factor for each other grid, :math:`N` is the
    number of grid points, and :math:`\\sigma` is the localization factor.

    Parameters
    ----------
    cell : np.ndarray
        An array of periods for each dimension of the grid.
    grid_j : np.ndarray
        An array of vectors of the grid around the selected grid.
    grid_i : np.ndarray
        An array of the vector of the selected grid.
    grid_j_weight : np.ndarray
        An array of weights for each target vector.
    sigma_squared : float
        The localization factor for the spherical Gaussian.


    Returns
    -------
    tuple
        A tuple containing two numpy arrays:
        wl : np.ndarray
            An array of localized weights for each vector.
        num : np.ndarray
            The sum of the localized weights.

    """
    xy = grid_j - grid_i
    if cell is not None:
        xy -= np.round(xy / cell) * cell

    wl = np.exp(-0.5 / sigma_squared * np.sum(xy**2, axis=1)) * grid_j_weight
    num = np.sum(wl)

    return wl, num
