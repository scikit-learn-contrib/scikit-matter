import warnings
from typing import Union

import numpy as np
from scipy.special import logsumexp as LSE
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_random_state
from tqdm import tqdm

from ..metrics._pairwise import (
    pairwise_euclidean_distances,
    pairwise_mahalanobis_distances,
)
from ..utils._sparsekde import (
    GaussianMixtureModel,
    NearestGridAssigner,
    covariance,
    effdim,
    local_population,
    oas,
    quick_shift,
    rij,
)


DIST_METRICS = {
    "periodic_euclidean": pairwise_euclidean_distances,
}


class SparseKDE(BaseEstimator):
    """A sparse implementation of the Kernel Density Estimation.
    This class is used to build a sparse kernel density estimator.
    It takes a set of descriptors and a set of weights as input,
    and fit the KDE model on the sampled point (e.g. the grid point
    selected by FPS). First, the probability density is estimated for
    each sampled point. Then, quick shift clustering is applied to the
    grid points. Finally, a kernel density estimator is built based on
    the clustering results.

    Parameters
    ----------
    descriptors: np.ndarray
        Descriptors of the system where you want to build a sparse KDE.
        It should be an array of shape `(n_descriptors, n_features)`.
    weights: np.ndarray, default=None
        Weights of the descriptors.
        If None, all weights are set to `1/n_descriptors`.
    kernel : {'gaussian'}, default='gaussian'
        The kernel to use. Currentlty only one.
    metric : str, default='periodic_euclidean'
        The metric to use. Currently only one.
    metric_params : dict, default=None
        Additional parameters to be passed to the use of
        metric.  i.e. the cell dimension for `periodic_euclidean`
        {'cell': [2, 2]}
    qs : float, default=1.0
        Scaling factor used during the QS clustering.
    gs : int, default=-1
        The neighbor shell for gabriel shift.
    thrpcl : float, default=0.0
        Clusters with a pk lower than this value are merged with the NN.
    fspread : float, default=-1.0
        The fractional variance for bandwidth estimation.
    fpoints : float, default=0.15
        The fractional number of grid points.
    nmsopt : int, default=0
        The number of mean-shift refinement steps.


    Attributes
    ----------
    kdecut2 : float
        The cut-off value for the KDE.
    cell : numpy.ndarray
        The cell dimension for the metric.
    model : :class:`skmatter.utils.GaussianMixtureModel`
        The model of the KDE.
    cluster_mean : numpy.ndarray of shape (n_clusters, n_features)
        The mean of each gaussian.
    cluster_cov : numpy.ndarray of shape (n_clusters, n_features)
        The covariance matrix of each gaussian.
    cluster_weight : numpy.ndarray of shape (n_clusters, n_features)
        The weight of each gaussian.


    Examples
    --------
    >>> import numpy as np
    >>> from skmatter.neighbors import SparseKDE
    >>> from skmatter.feature_selection import FPS
    >>> np.random.seed(0)
    >>> n_samples = 10000
    >>> cov1 = [[1, 0.5], [0.5, 1]]
    >>> cov2 = [[1, 0.5], [0.5, 0.5]]
    >>> sample1 = np.random.multivariate_normal([0, 0], cov1, n_samples)
    >>> sample2 = np.random.multivariate_normal([4, 4], cov2, n_samples)
    >>> samples = np.concatenate([sample1, sample2])
    >>> selector = FPS(n_to_select=int(np.sqrt(2 * n_samples)))
    >>> result = selector.fit_transform(samples.T).T
    >>> estimator = SparseKDE(samples, None, fpoints=0.5, qs=0.85)
    >>> estimator.fit(result)
    SparseKDE(descriptors=array([[-1.72779275, -1.32763554],
           [-1.96805856,  0.27283464],
           [-1.12871372, -2.1059916 ],
           ...,
           [ 3.75859454,  3.10217702],
           [ 1.6544348 ,  3.41851374],
           [ 4.08667637,  3.42457743]]),
              fpoints=0.5, qs=0.85,
              weights=array([5.e-05, 5.e-05, 5.e-05, ..., 5.e-05, 5.e-05, 5.e-05]))
    >>> round(estimator.score(result), 3)
    2.767
    >>> estimator.sample()
    array([[3.32383366, 3.51779084]])
    """

    def __init__(
        self,
        descriptors: np.ndarray,
        weights: Union[np.ndarray, None] = None,
        kernel: str = "gaussian",
        metric: str = "periodic_euclidean",
        metric_params: Union[dict, None] = None,
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
        self.cell = metric_params["cell"] if metric_params is not None else None
        self.descriptors = descriptors
        self.weights = (
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
        self.model = None
        self.cluster_mean = None
        self.cluster_cov = None
        self.cluster_weight = None

        if self.fspread > 0:
            self.fpoints = -1.0

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

        self.kdecut2 = 9 * (np.sqrt(X.shape[1]) + 1) ** 2
        grid_dist_mat = DIST_METRICS[self.metric](X, X, squared=True, cell=self.cell)
        np.fill_diagonal(grid_dist_mat, np.inf)
        min_grid_dist = np.min(grid_dist_mat, axis=1)
        _, grid_neighbour, sample_labels_, sample_weight = (
            self._assign_descriptors_to_grids(X)
        )
        h_invs, normkernels, qscut2 = self._computes_localization(
            X, sample_weight, min_grid_dist
        )
        probs = self._computes_kernel_density_estimation(
            X, sample_weight, h_invs, normkernels, grid_neighbour
        )
        normpks = LSE(probs)
        cluster_centers, idxroot = quick_shift(probs, grid_dist_mat, qscut2, self.gs)
        cluster_centers, idxroot = self._post_process(
            X, cluster_centers, idxroot, probs, normpks
        )
        self.cluster_weight, self.cluster_mean, self.cluster_cov = (
            self._generate_probability_model(
                X,
                sample_labels_,
                cluster_centers,
                h_invs,
                normkernels,
                probs,
                idxroot,
                normpks,
            )
        )
        self.model = GaussianMixtureModel(
            self.cluster_weight, self.cluster_mean, self.cluster_cov, cell=self.cell
        )
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
        return np.array([self.model(x) for x in X])

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
        check_is_fitted(self)
        rng = check_random_state(random_state)
        u = rng.uniform(0, 1, size=n_samples)
        cumsum_weight = np.cumsum(np.asarray(self.cluster_weight))
        sum_weight = cumsum_weight[-1]
        idxs = np.searchsorted(cumsum_weight, u * sum_weight)

        return np.concatenate(
            [
                np.atleast_2d(
                    rng.multivariate_normal(self.cluster_mean[i], self.cluster_cov[i])
                )
                for i in idxs
            ]
        )

    def _assign_descriptors_to_grids(self, X):

        assigner = NearestGridAssigner(DIST_METRICS[self.metric], self.cell)
        assigner.fit(X)
        labels = assigner.predict(self.descriptors, sample_weight=self.weights)
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

        for i in tqdm(range(len(X)), desc="Estimating kernel density bandwidths"):
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
                self._bandwidth_estimation_from_localization(X, wlocal, flocal, i)
            )

        qscut2 *= self.qs**2

        return h_invs, normkernels, qscut2

    def _localization_based_on_fraction_of_points(
        self, X, sample_weights, sigma2, flocal, idx, delta, tune
    ):
        """Used in cases where one expects clusters with very different spreads,
        but similar populations"""

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
        """Used in cases where one expects the spatial extentof clusters to be
        relatively homogeneous"""
        sigma2[idx] = mindist[idx]
        wlocal, flocal[idx] = local_population(
            self.cell, self.descriptors, X, sample_weights, sigma2[idx]
        )

        return sigma2, flocal, wlocal

    def _bandwidth_estimation_from_localization(self, X, wlocal, flocal, idx):

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
        h_invs: np.ndarray,
        normkernel: np.ndarray,
        neighbour: dict,
    ):

        prob = np.full(len(X), -np.inf)
        dummd1s_mat = pairwise_mahalanobis_distances(
            X, X, h_invs, self.cell, squared=True
        )
        for i in tqdm(
            range(len(X)), desc="Computing kernel density on reference points"
        ):
            for j, dummd1 in enumerate(dummd1s_mat[:, i, i]):
                if dummd1 > self.kdecut2:
                    lnk = -0.5 * (normkernel[j] + dummd1) + np.log(sample_weights[j])
                    prob[i] = LSE([prob[i], lnk])
                else:
                    neighbours = neighbour[j][
                        np.any(self.descriptors[neighbour[j]] != X[i], axis=1)
                    ]
                    if neighbours.size == 0:
                        continue
                    dummd1s = pairwise_mahalanobis_distances(
                        self.descriptors[neighbours],
                        X[i][np.newaxis, ...],
                        h_invs[j],
                        self.cell,
                        squared=True,
                    ).reshape(-1)
                    lnks = -0.5 * (normkernel[j] + dummd1s) + np.log(
                        self.weights[neighbours]
                    )
                    prob[i] = LSE(np.concatenate([[prob[i]], lnks]))

        prob -= np.log(np.sum(sample_weights))

        return prob

    def _post_process(
        self,
        X: np.ndarray,
        cluster_centers: np.ndarray,
        idxroot: np.ndarray,
        probs: np.ndarray,
        normpks: float,
    ):

        nk = len(cluster_centers)
        to_merge = np.full(nk, False)
        for k in range(nk):
            dummd1 = np.exp(LSE(probs[idxroot == cluster_centers[k]]) - normpks)
            to_merge[k] = dummd1 > self.thrpcl
        # merge the outliers
        for i in range(nk):
            if not to_merge[k]:
                continue
            dummd1yi1 = cluster_centers[i]
            dummd1 = np.inf
            for j in range(nk):
                if to_merge[k]:
                    continue
                dummd2 = self.metric(
                    X[idxroot[dummd1yi1]], X[idxroot[j]], cell=self.cell
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
                cluster_centers[i] = np.argmax(
                    np.ma.array(probs, mask=idxroot != cluster_centers[i])
                )
                idxroot[idxroot == dummd1yi1] = cluster_centers[i]

        return cluster_centers, idxroot

    def _generate_probability_model(
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
            cluster_weight[k] = np.exp(LSE(probs[idxroot == center_idx[k]]) - normpks)
            cluster_mean[k] = self._mean_shift_optimizaton(
                X[center_idx[k]],
                X,
                h_invs[center_idx[k]],
                normkernels[center_idx[k]],
                probs,
            )
            cluster_cov[k] = self._update_cluster_cov(
                X, k, sample_labels, probs, idxroot, center_idx
            )

        return cluster_weight, cluster_mean, cluster_cov

    def _mean_shift_optimizaton(
        self,
        mean: np.ndarray,
        X: np.array,
        h_inv: np.ndarray,
        normkernel: float,
        probs: np.ndarray,
    ):
        # Never tested and not used in any available example cases
        grid = np.copy(mean)
        for _ in range(self.nmsopt):
            # Mean shift optimization
            msmu = np.zeros(X.shape[1], dtype=float)
            tmppks = -np.inf
            dummd1s = pairwise_mahalanobis_distances(
                X,
                grid[np.newaxis, ...],
                h_inv,
                self.cell,
                squared=True,
            )[0]
            msws = -0.5 * (normkernel + dummd1s) + probs
            tmpmsmu = rij(self.cell, X, grid)
            msmu += np.sum(np.exp(msws) * tmpmsmu, axis=1)
            tmppks = LSE(np.concatenate([tmppks, msws]))
            mean += msmu / np.exp(tmppks)

        return mean

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
                    self.weights,
                    self.cell,
                )
                print("Warning: single point cluster!")
        else:
            cov = self._get_lcov_cluster(
                len(X), X, idxroot, center_idx[k], probs, self.cell
            )
            if np.sum(idxroot == center_idx[k]) == 1:
                cov = self._get_lcov_cluster(
                    self.nsamples,
                    self.descriptors,
                    sample_labels,
                    center_idx[k],
                    self.weights,
                    self.cell,
                )
                print("Warning: single point cluster!")
            cov = oas(
                cov,
                LSE(probs[idxroot == center_idx[k]]) * self.nsamples,
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
        normww = LSE(probs[clroots == idcl])
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
        totnormp = LSE(probs)
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
