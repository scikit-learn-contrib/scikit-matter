import warnings
from typing import Union, Optional
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _check_sample_weight
import numpy as np
from rich.progress import track

from ..metrics.pairwise import pairwise_euclidean_distances


class SparseKDE(BaseEstimator):
    """A sparse implementation of the Kernel Density Estimation.

    The bandwidth will be optimized per sample.

    - We only support Gaussian kernels. (Check
    howe hard others are and make it paramater later)
    - Implement a sklean like metric: named periodic euclidian. and make metric parameter
    distance.

    Parameters
    ----------
    kernel : {'gaussian'}, default='gaussian'
        The kernel to use. Currentlty only one. Check how sklearn kernels are defined. Try to reuse
    metric : str, default='periodic_euclidean'
    metric_params : dict, default=None
        Additional parameters to be passed to the use of
        metric.  i.e. the cell dimension for `periodic_euclidean`
    """

    def __init__(
        self,
        kernel,
        metric,
        metric_params,
        descriptors: np.ndarray,
        weights: np.ndarray,
        qs: float = 1.0,
        gs: int = -1,
        thrpcl: float = 0.0,
        fspread: float = -1.0,
        fpoints: float = 0.15,
        nmsopt: int = 0,
    ):
        self.kernel = kernel
        self.metric = metric
        self.metric_params = metric_params
        self.cell = metric_params["cell"] if "cell" in metric_params else None
        self.descriptors = descriptors
        self.weight = (
            weights
            if weights is not None
            else np.ones(len(descriptors)) / len(descriptors)
        )
        self.nsamples = len(descriptors)
        self.qs = qs
        self.gs = gs
        self.thrpcl = thrpcl
        self.fspread = fspread
        self.fpoints = fpoints
        self.nmsopt = nmsopt
        self.kdecut2 = None

        if self.fspread > 0:
            self.fpoints = -1.0

    def fit(self, X, y=None, sample_weight=None, igrid=None):
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
            List of sample weights attached to the data X.

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        # if sample_weight is not None:
        #     sample_weight = _check_sample_weight(
        #         sample_weight, X, dtype=np.float64, only_non_negative=True
        #     )
        # else:
        #     sample_weight = np.ones(X.shape[0], dtype=np.float64) / X.shape[0]
        self.kdecut2 = 9 * (np.sqrt(X.shape[1]) + 1) ** 2
        grid_dist_mat = pairwise_euclidean_distances(X, X, squared=True, cell=self.cell)
        np.fill_diagonal(grid_dist_mat, np.inf)
        min_grid_dist = np.min(grid_dist_mat, axis=1)
        grid_npoints, grid_neighbour, sample_labels_, sample_weight = (
            self._assign_descriptors_to_grids(X)
        )
        h_invs, normkernels, qscut2 = self._computes_localization(
            X, sample_weight, min_grid_dist
        )
        probs = self._computes_kernel_density_estimation(
            X, sample_weight, h_invs, normkernels, igrid, grid_neighbour
        )
        normpks = logsumexp(np.ones(grid_dist_mat.shape[0]), probs, 1)
        cluster_centers, idxroot = quick_shift(
            X, probs, grid_dist_mat, qscut2, normpks, self.gs, self.cell, self.thrpcl
        )

        return self.generate_probability_model(X, sample_labels_, cluster_centers, h_invs, normkernels, probs, idxroot, normpks)

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
        ...

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

        Currently, this is implemented only for gaussian and tophat kernels.

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
        ...

    def _assign_descriptors_to_grids(self, X):

        assigner = NearestNeighborClustering(self.cell)
        assigner.fit(X)
        labels = assigner.predict(self.descriptors, sample_weight=self.weight)
        grid_npoints = assigner.grid_npoints
        grid_neighbour = assigner.grid_neighbour

        return grid_npoints, grid_neighbour, labels, assigner.grid_weight

    def _computes_localization(
        self, X, sample_weights: np.ndarray, mindist: np.ndarray
    ):

        cov = covariance(X, sample_weights, self.cell)

        if self.cell is not None:
            tune = sum(self.cell**2)
        else:
            tune = np.trace(cov)

        sigma2 = np.full(len(X), tune, dtype=float)
        # initialize the localization based on fraction of data spread
        if self.fspread > 0:
            sigma2 *= self.fspread**2
        flocal, normkernels, qscut2, h_tr_normed = (
            np.zeros(len(X)),
            np.zeros(len(X)),
            np.zeros(len(X)),
            np.zeros(len(X)),
        )
        h_invs = np.zeros((len(X), X.shape[1], X.shape[1]))

        for i in track(
            range(len(X)), description="Estimating kernel density bandwidths"
        ):
            wlocal, flocal[i] = local_population(
                self.cell, X, X[i], sample_weights, sigma2[i]
            )
            if self.fpoints > 0:
                sigma2, flocal, wlocal = self._localization_based_on_fraction_of_points(
                    X, sample_weights, sigma2, flocal, i, 1 / self.nsamples, tune
                )
            elif sigma2[i] < flocal[i]:
                sigma2, flocal, wlocal = self._localization_based_on_fraction_of_spread(
                    X, sample_weights, sigma2, flocal, i, mindist
                )
            h_invs[i], normkernels[i], qscut2[i], h_tr_normed[i] = (
                self._bandwidth_estimation_from_localization(
                    X, sample_weights, wlocal, flocal, i
                )
            )

        qscut2 *= self.qs**2

        return h_invs, normkernels, qscut2

    def _localization_based_on_fraction_of_points(
        self, X, sample_weights, sigma2, flocal, idx, delta, tune
    ):
        """Used in cases where one expects clusterswith very different spreads,
        but similar populations"""

        lim = self.fpoints
        if lim <= sample_weights[idx]:
            lim = sample_weights[idx] + delta
            warnings.warn(
                " Warning: localization smaller than voronoi,"
                " increase grid size (meanwhile adjusted localization)!"
            )
        while flocal[idx] < lim:
            sigma2[idx] += tune
            wlocal, flocal[idx] = local_population(
                self.cell, X, X[idx], sample_weights, sigma2[idx]
            )
        j = 1
        while True:
            if flocal[idx] > lim:
                sigma2[idx] -= tune / 2**j
            else:
                sigma2[idx] += tune / 2**j
            wlocal, flocal[idx] = local_population(
                self.cell, X, X[idx], sample_weights, sigma2[idx]
            )
            if abs(flocal[idx] - lim) < delta:
                break
            j += 1

        return sigma2, flocal, wlocal

    def _localization_based_on_fraction_of_spread(
        self, X, sample_weights, sigma2, flocal, idx, mindist
    ):

        sigma2[idx] = mindist[idx]
        wlocal, flocal[idx] = local_population(
            self.cell, self.descriptors, X, sample_weights, sigma2[idx]
        )

        return sigma2, flocal, wlocal

    def _bandwidth_estimation_from_localization(
        self, X, sample_weights, wlocal, flocal, idx
    ):

        cov_i = covariance(X, wlocal, self.cell)
        nlocal = flocal[idx] * self.nsamples
        local_dimension = effdim(cov_i)
        cov_i = oas(cov_i, nlocal, X.shape[1])
        # localized version of Silverman's rule
        h = (4.0 / nlocal / (local_dimension + 2.0)) ** (
            2.0 / (local_dimension + 4.0)
        ) * cov_i
        h_tr_normed = np.trace(h) / h.shape[0]
        h_inv = np.linalg.inv(h)
        _, logdet_h = np.linalg.slogdet(h)
        normkernel = X.shape[1] * np.log(2 * np.pi) + logdet_h
        qscut2 = np.trace(cov_i)

        return h_inv, normkernel, qscut2, h_tr_normed

    def _computes_kernel_density_estimation(
        self,
        X: np.ndarray,
        sample_weights: np.ndarray,
        h_inv: np.ndarray,
        normkernel: np.ndarray,
        igrid: np.ndarray,
        neighbour: dict,
    ):

        prob = np.full(len(X), -np.inf)
        for i in track(
            range(len(X)), description="Computing kernel density on reference points"
        ):
            dummd1s = mahalanobis(self.cell, X, X[i], h_inv)
            for j, dummd1 in enumerate(dummd1s):
                if dummd1 > self.kdecut2:
                    lnk = -0.5 * (normkernel[j] + dummd1) + np.log(sample_weights[j])
                    prob[i] = _update_prob(prob[i], lnk)
                else:
                    neighbours = neighbour[j][neighbour[j] != igrid[i]]
                    dummd1s = mahalanobis(
                        self.cell, self.descriptors[neighbours], X[i], h_inv[j]
                    )
                    lnks = -0.5 * (normkernel[j] + dummd1s) + np.log(
                        self.weight[neighbours]
                    )
                    prob[i] = _update_probs(prob[i], lnks)

        prob -= np.log(np.sum(sample_weights))

        return prob

    def generate_probability_model(
        self,
        X: np.ndarray,
        sample_labels: np.ndarray,
        cluster_centers: np.ndarray,
        h_invs: np.ndarray,
        normkernels: np.ndarray,
        probs: np.ndarray,
        idxroot: np.ndarray,
        normpks: float,
    ):
        """
        Generates a probability model based on the given inputs.

        Parameters:
            None

        Returns:
            None
        """

        dimension = X.shape[1]
        cluster_mean = np.zeros((len(cluster_centers), dimension), dtype=float)
        cluster_cov = np.zeros(
            (len(cluster_centers), dimension, dimension), dtype=float
        )
        cluster_weight = np.zeros(len(cluster_centers), dtype=float)
        center_idx = np.unique(idxroot)

        for k in range(len(cluster_centers)):
            cluster_mean[k] = X[center_idx[k]]
            cluster_weight[k] = np.exp(
                logsumexp(idxroot, probs, center_idx[k]) - normpks
            )
            for _ in range(self.nmsopt):
                msmu = np.zeros(dimension, dtype=float)
                tmppks = -np.inf
                for i, x in enumerate(X):
                    dummd1 = mahalanobis(
                        self.cell, x, X[center_idx[k]], h_invs[center_idx[k]]
                    )
                    msw = -0.5 * (normkernels[center_idx[k]] + dummd1) + probs[i]
                    tmpmsmu = rij(self.cell, x, X[center_idx[k]])
                    msmu += np.exp(msw) * tmpmsmu
                tmppks = _update_prob(tmppks, msw)
                cluster_mean[k] += msmu / np.exp(tmppks)
            cluster_cov[k] = self._update_cluster_cov(X, k, sample_labels, probs, idxroot, center_idx)

        return cluster_weight, cluster_mean, cluster_cov

    def _update_cluster_cov(
        self,
        X: np.ndarray,
        k: int,
        sample_labels: np.ndarray,
        probs: np.ndarray,
        idxroot: np.ndarray,
        center_idx: np.ndarray,
    ):

        if self.cell is not None:
            cov = self._get_lcov_clusterp(
                len(X), self.nsamples, X, idxroot, center_idx[k], probs, self.cell
            )
            if np.sum(idxroot == center_idx[k]) == 1:
                cov = self._get_lcov_clusterp(
                    self.nsamples,
                    self.nsamples,
                    self.descriptors,
                    sample_labels,
                    center_idx[k],
                    self.weight,
                    self.cell,
                )
                print("Warning: single point cluster!")
        else:
            cov = self._get_lcov_cluster(len(X), X, idxroot, center_idx[k], probs, self.cell)
            if np.sum(idxroot == center_idx[k]) == 1:
                cov = self._get_lcov_cluster(
                    self.nsamples,
                    self.descriptors,
                    sample_labels,
                    center_idx[k],
                    self.weight,
                    self.cell,
                )
                print("Warning: single point cluster!")
            cov = oas(
                cov,
                logsumexp(idxroot, probs, center_idx[k]) * self.nsamples,
                X.shape[1],
            )

        return cov

    def _get_lcov_cluster(
        self,
        N: int,
        x: np.ndarray,
        clroots: np.ndarray,
        idcl: int,
        probs: np.ndarray,
        cell: np.ndarray,
    ):

        ww = np.zeros(N)
        normww = logsumexp(clroots, probs, idcl)
        ww[clroots == idcl] = np.exp(probs[clroots == idcl] - normww)
        cov = covariance(x, ww, cell)

        return cov

    def _get_lcov_clusterp(
        self,
        N: int,
        Ntot: int,
        x: np.ndarray,
        clroots: np.ndarray,
        idcl: int,
        probs: np.ndarray,
        cell: np.ndarray,
    ):

        ww = np.zeros(N)
        totnormp = logsumexp(np.zeros(N), probs, 0)
        cov = np.zeros((x.shape[1], x.shape[1]), dtype=float)
        xx = np.zeros(x.shape, dtype=float)
        ww[clroots == idcl] = np.exp(probs[clroots == idcl] - totnormp)
        ww *= Ntot
        nlk = np.sum(ww)
        for i in range(x.shape[1]):
            xx[:, i] = x[:, i] - np.round(x[:, i] / cell[i]) * cell[i]
            r2 = (np.sum(ww * np.cos(xx[:, i])) / nlk) ** 2 + (
                np.sum(ww * np.sin(xx[:, i])) / nlk
            ) ** 2
            re2 = (nlk / (nlk - 1)) * (r2 - (1 / nlk))
            cov[i, i] = 1 / (np.sqrt(re2) * (2 - re2) / (1 - re2))

        return cov


def covariance(X: np.ndarray, sample_weights: np.ndarray, cell: np.ndarray):
    """
    Calculate the covariance matrix for a given set of grid positions and weights.

    Parameters:
        grid_pos (np.ndarray): An array of shape (nsample, dimension)
        representing the grid positions.
        period (np.ndarray): An array of shape (dimension,)
        representing the periodicity of each dimension.
        grid_weight (np.ndarray): An array of shape (nsample,)
        representing the weights of the grid positions.

    Returns:
        cov (np.ndarray): The covariance matrix of shape (dimension, dimension).

    Note:
        The function assumes that the grid positions, weights, and total weight are provided correctly.
        The function handles periodic and non-periodic dimensions differently to calculate the covariance matrix.
    """

    nsample = X.shape[0]
    dimension = X.shape[1]
    xm = np.zeros(dimension)
    xxm = np.zeros((nsample, dimension))
    xxmw = np.zeros((nsample, dimension))
    totw = np.sum(sample_weights)

    if cell is None:
        xm = np.average(X, axis=0, weights=sample_weights / totw)
    else:
        for i in range(dimension):
            sumsin = (
                np.sum(sample_weights * np.sin(X[:, i]) * (2 * np.pi) / cell[i]) / totw
            )
            sumcos = (
                np.sum(sample_weights * np.cos(X[:, i]) * (2 * np.pi) / cell[i]) / totw
            )
            xm[i] = np.arctan2(sumsin, sumcos)

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

    Args:
        cell (np.ndarray): An array of periods for each dimension of the grid.
        x (np.ndarray): An array of vectors to be localized.
        y (np.ndarray): An array of target vectors representing the grid.
        grid_weight (np.ndarray): An array of weights for each target vector.
        s2 (float): The scaling factor for the squared distance.

    Returns:
        tuple: A tuple containing two numpy arrays:
            wl (np.ndarray): An array of localized weights for each vector.
            num (np.ndarray): The sum of the localized weights.

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

    Parameters:
        cov (ndarray): The covariance matrix.

    Returns:
        float: The effective dimension of the covariance matrix.

    Ref:
        https://ieeexplore.ieee.org/document/7098875
    """

    eigval = np.linalg.eigvals(cov)
    eigval /= sum(eigval)
    eigval *= np.log(eigval)
    eigval[np.isnan(eigval)] = 0.0

    return np.exp(-sum(eigval))


def oas(cov: np.ndarray, n: float, D: int):
    """Oracle approximating shrinkage (OAS) estimator

    Args:
        cov: A covariance matrix
        n: The local population
        D: Dimension

    Returns
        Covariance matrix
    """

    tr = np.trace(cov)
    tr2 = tr**2
    tr_cov2 = np.trace(cov**2)
    phi = ((1 - 2 / D) * tr_cov2 + tr2) / ((n + 1 - 2 / D) * tr_cov2 - tr2 / D)

    return (1 - phi) * cov + phi * np.eye(D) * tr / D


def mahalanobis(period: np.ndarray, x: np.ndarray, y: np.ndarray, cov_inv: np.ndarray):
    """
    Calculates the Mahalanobis distance between two vectors.

    Args:
        period (np.ndarray): An array of periods for each dimension of vectors.
        x (np.ndarray): An array of vectors to be localized.
        y (np.ndarray): An array of target vectors.
        cov_inv (np.ndarray): The inverse of the covariance matrix.

    Returns:
        float: The Mahalanobis distance.

    """

    x, cov_inv = _mahalanobis_preprocess(x, cov_inv)
    return _mahalanobis(period, x, y, cov_inv)


def _mahalanobis_preprocess(x: np.ndarray, cov_inv: np.ndarray):

    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    if len(cov_inv.shape) == 2:
        cov_inv = cov_inv[np.newaxis, :, :]

    return x, cov_inv


def _mahalanobis(period: np.ndarray, x: np.ndarray, y: np.ndarray, cov_inv: np.ndarray):

    tmpv = np.zeros(x.shape, dtype=float)
    xcx = np.zeros(x.shape[0], dtype=float)
    xy = x - y
    if period is not None:
        xy -= np.round(xy / period) * period
    if cov_inv.shape[0] == 1:
        # many samples and one cov
        tmpv = xy.dot(cov_inv[0])
    else:
        # many samples and many cov
        for i in range(x.shape[0]):
            tmpv[i] = np.dot(xy[i], cov_inv[i])
    for i in range(x.shape[0]):
        xcx[i] = np.dot(xy[i], tmpv[i].T)

    return xcx


def _update_probs(prob_i: float, lnks: np.ndarray):

    for lnk in lnks:
        prob_i = _update_prob(prob_i, lnk)

    return prob_i


def _update_prob(prob_i: float, lnk: float):

    if prob_i > lnk:
        return prob_i + np.log(1 + np.exp(lnk - prob_i))
    else:
        return lnk + np.log(1 + np.exp(prob_i - lnk))


class NearestNeighborClustering:
    """NearestNeighborClustering Class
    Assign descriptor to its nearest grid."""

    def __init__(self, period: Optional[np.ndarray] = None) -> None:

        self.labels_ = None
        self.period = period
        self._distance = pairwise_euclidean_distances
        self.grid_pos = None
        self.grid_npoints = None
        self.grid_weight = None
        self.grid_neighbour = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit the data. Generate the cluster center by FPS algorithm."""

        ngrid = len(X)
        self.grid_pos = X
        self.grid_npoints = np.zeros(ngrid, dtype=int)
        self.grid_weight = np.zeros(ngrid, dtype=float)
        self.grid_neighbour = {i: [] for i in range(ngrid)}

    def predict(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transform the data."""
        if sample_weight is None:
            sample_weight = np.ones(len(X)) / len(X)
        self.labels_ = []
        for i, point in track(
            enumerate(X), description="Assigning samples to grids...", total=len(X)
        ):
            descriptor2grid = self._distance(
                X=point.reshape(1, -1), Y=self.grid_pos, cell=self.period
            )
            self.labels_.append(np.argmin(descriptor2grid))
            self.grid_npoints[self.labels_[-1]] += 1
            self.grid_weight[self.labels_[-1]] += sample_weight[i]
            self.grid_neighbour[self.labels_[-1]].append(i)

        for key in self.grid_neighbour:
            self.grid_neighbour[key] = np.array(self.grid_neighbour[key])

        return self.labels_


def quick_shift(
    X: np.ndarray,
    probs: np.ndarray,
    dist_matrix: np.ndarray,
    cutoff2: np.ndarray,
    normpks: float,
    gs: float,
    cell: np.ndarray,
    thrpcl: float,
):
    """
    Perform quick shift clustering on the given probability array and distance matrix.

    Args:
        probs (np.ndarray): The log-likelihood of each sample.
        dist_matrix (np.ndarray): The squared distance matrix.
        cutoff2 (np.ndarray): The squared cutoff array.
        gs (float): The value of gs.

    Returns:
        tuple: A tuple containing the cluster centers and the root indices.
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
                    nneighs |= gabriel[j]
            neighs |= nneighs

        next_idx = idx
        dmin = np.inf
        for j in range(ngrid):
            if probs[j] > probs[idx] and distmm[idx, j] < dmin and neighs[j]:
                next_idx = j
                dmin = distmm[idx, j]

        return next_idx

    def qs_next(
        idx: int, idxn: int, probs: np.ndarray, distmm: np.ndarray, lambda_: float
    ):
        """Find next cluster with respect to qscut(lambda_)."""

        ngrid = len(probs)
        dmin = np.inf
        next_idx = idx
        if probs[idxn] > probs[idx]:
            next_idx = idxn
        for j in range(ngrid):
            if (
                probs[j] > probs[idx]
                and distmm[idx, j] < dmin
                and distmm[idx, j] < lambda_
            ):
                next_idx = j
                dmin = distmm[idx, j]

        return next_idx

    def getidmax(v1: np.ndarray, probs: np.ndarray, clusterid: int):

        tmpv = np.copy(probs)
        tmpv[v1 != clusterid] = -np.inf
        return np.argmax(tmpv)

    def post_process(
        normpks: float,
        cluster_centers: np.ndarray,
        grid_pos: np.ndarray,
        idxroot: np.ndarray,
        probs: np.ndarray,
        cell: np.ndarray,
        thrpcl: float,
    ):

        nk = len(cluster_centers)
        to_merge = np.full(nk, False)
        for k in range(nk):
            dummd1 = np.exp(logsumexp(idxroot, probs, cluster_centers[k]) - normpks)
            to_merge[k] = dummd1 > thrpcl
        # merge the outliers
        for i in range(nk):
            if not to_merge[k]:
                continue
            dummd1yi1 = cluster_centers[i]
            dummd1 = np.inf
            for j in range(nk):
                if to_merge[k]:
                    continue
                dummd2 = pairwise_euclidean_distances(
                    grid_pos[idxroot[dummd1yi1]], grid_pos[idxroot[j]], cell=cell
                )
                if dummd2 < dummd1:
                    dummd1 = dummd2
                    cluster_centers[i] = j
            idxroot[idxroot == dummd1yi1] = cluster_centers[i]
        if sum(to_merge) > 0:
            cluster_centers = np.concatenate(
                np.argwhere(idxroot == np.arange(len(idxroot)))
            )
            nk = len(cluster_centers)
            for i in range(nk):
                dummd1yi1 = cluster_centers[i]
                cluster_centers[i] = getidmax(idxroot, probs, cluster_centers[i])
                idxroot[idxroot == dummd1yi1] = cluster_centers[i]

        return cluster_centers, idxroot

    gabrial = get_gabriel_graph(dist_matrix)
    idmindist = np.argmin(dist_matrix, axis=1)
    idxroot = np.full(dist_matrix.shape[0], -1, dtype=int)
    for i in track(range(dist_matrix.shape[0]), description="Quick-Shift"):
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

    return post_process(normpks, cluster_centers, X, idxroot, probs, cell, thrpcl)


def get_gabriel_graph(dist_matrix2: np.ndarray):
    """
    Generate the Gabriel graph based on the given squared distance matrix.

    Parameters:
        dist_matrix2 (np.ndarray): The squared distance matrix of shape (n_points, n_points).
        outputname (Optional[str]): The name of the output file. Default is None.

    Returns:
        np.ndarray: The Gabriel graph matrix of shape (n_points, n_points).

    """

    n_points = dist_matrix2.shape[0]
    gabriel = np.full((n_points, n_points), True)
    for i in track(range(n_points), description="Calculating Gabriel graph"):
        gabriel[i, i] = False
        for j in range(i, n_points):
            if np.sum(dist_matrix2[i] + dist_matrix2[j] < dist_matrix2[i, j]):
                gabriel[i, j] = False
                gabriel[j, i] = False

    return gabriel


from scipy.special import logsumexp as LSE


def logsumexp(v1: np.ndarray, probs: np.ndarray, clusterid: int):

    mask = v1 == clusterid
    probs = np.copy(probs)
    probs[~mask] = -np.inf

    return LSE(probs)


def rij(period: np.ndarray, xi: np.ndarray, xj: np.ndarray):
    """
    Calculates the period-concerned position vector.
    Args:
        period (np.ndarray): An array of periods for each dimension of the points.
        -1 stands for not periodic.
        xij (np.ndarray): An array for storing the result.
        xi (np.ndarray): An array of point coordinates. It can also contain many points.
        Shape: (n_points, n_dimensions)
        xj (np.ndarray): An array of point coordinates. It can only contain one point.

    Returns:
        xij (np.ndarray): An array of position vectors. Shape: (n_points, n_dimensions)
    """

    xij = xi - xj
    if period is not None:
        xij -= np.round(xij / period) * period

    return xij
