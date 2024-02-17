#!/usr/bin/env python
# coding: utf-8

"""
Sparse KDE examples
================================================================

Example for the usage of the :class:`skmatter.neighbors.SparseKDE` class. Here we show
how to use the sparse KDE model to fit the probability distribution based on sampled
data and how to conduct pobabilistic analysis of molecular motifs.

We start from a simple case. Here we first sample from two gaussians.
"""


# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from skmatter.clustering import QuickShift
from skmatter.datasets import load_hbond_dataset
from skmatter.feature_selection import FPS
from skmatter.metrics import DIST_METRICS, pairwise_mahalanobis_distances
from skmatter.neighbors import SparseKDE
from skmatter.utils import covariance, oas


np.random.seed(0)
means = np.array([[0, 0], [4, 4], [6, -2]])
covariances = np.array(
    [[[1, 0.5], [0.5, 1]], [[1, 0.5], [0.5, 0.5]], [[1, -0.5], [-0.5, 1]]]
)
N_SAMPLES = 100000
samples = np.concatenate(
    [
        np.random.multivariate_normal(means[0], covariances[0], N_SAMPLES),
        np.random.multivariate_normal(means[1], covariances[1], N_SAMPLES),
        np.random.multivariate_normal(means[2], covariances[2], N_SAMPLES),
    ]
)

# %% [markdown]
# We can visualize our sample result:
#
#

# %%
fig, ax = plt.subplots()
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.05, s=1)
ax.scatter(means[:, 0], means[:, 1], marker="+", color="red", s=100)

# %% [markdown]
# The perquisite of conducting sparse KDE is to partition the sample set. Here, we use
# the FPS method to generate grid points in the sample space:
#
#

# %%
selector = FPS(n_to_select=int(np.sqrt(3 * N_SAMPLES)))
grids = selector.fit_transform(samples.T).T
fig, ax = plt.subplots()
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.05, s=1)
ax.scatter(means[:, 0], means[:, 1], marker="+", color="red", s=100)
ax.scatter(grids[:, 0], grids[:, 1], color="orange", s=1)

# %% [markdown]
# Now we can do sparse KDE
#
#

# %%
estimator = SparseKDE(samples, None, fpoints=0.5, qs=0.85)
estimator.fit(grids)

# %% [markdown]
# We can have a comparasion with the original sampling result by plotting them:
#
#


# %%
class GaussianMixtureModel:

    def __init__(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        period: np.ndarray = None,
    ):
        self.weights = weights
        self.means = means
        self.covariances = covariances
        self.period = period
        self.dimension = self.means.shape[1]
        self.cov_inv = np.linalg.inv(self.covariances)
        self.cov_det = np.linalg.det(self.covariances)
        self.norm = 1 / np.sqrt((2 * np.pi) ** self.dimension * self.cov_det)

    def __call__(self, x: np.ndarray, i: int = None):

        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        if self.period is not None:
            xij = np.zeros(self.means.shape)
            xij = rij(self.period, xij, x, self.means)
        else:
            xij = x - self.means
        p = (
            self.weights
            * self.norm
            * np.exp(
                -0.5 * (xij[:, np.newaxis, :] @ self.cov_inv @ xij[:, :, np.newaxis])
            ).reshape(-1)
        )
        sum_p = np.sum(p)
        if i is None:
            return sum_p

        return np.sum(p[i]) / sum_p


def rij(period: np.ndarray, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:

    xij = xi - xj
    if period is not None:
        xij -= np.round(xij / period) * period

    return xij


# %%
original_model = GaussianMixtureModel(np.full(3, 1 / 3), means, covariances)
fitted_model = GaussianMixtureModel(
    estimator._sample_weights, estimator._grids, estimator._h
)
x, y = np.meshgrid(np.linspace(-6, 12, 100), np.linspace(-8, 8))
points = np.concatenate(np.stack([x, y], axis=-1))

probs = np.array([original_model(point) for point in points])
fitted_probs = np.array([fitted_model(point) for point in points])

fig, ax = plt.subplots()
ct1 = ax.contour(x, y, probs.reshape(x.shape), colors="blue")
ct2 = ax.contour(x, y, fitted_probs.reshape(x.shape), colors="orange")
h1, _ = ct1.legend_elements()
h2, _ = ct2.legend_elements()
ax.legend(
    [h1[0], h2[0]],
    ["original", "fitted"],
)

# %% [markdown]
# # Probabilistic Analysis of Molecular Motifs (PAMM)
#
#

# %% [markdown]
# Probabilistic analysis of molecular motifs is a method identifying molecular patterns
# based on an analysis of the probability distribution of fragments observed in an
# atomistic simulation. With the help of sparse KDE, it can be easily conducted. Here
# we define some functions to help us.
#
#


# %%
def quick_shift_refinement(
    X: np.ndarray,
    cluster_centers: np.ndarray,
    idxroot: np.ndarray,
    probs: np.ndarray,
    normpks: float,
    metric: str,
    cell: np.ndarray = None,
    thrpcl: float = 0.0,
):

    nk = len(cluster_centers)
    to_merge = np.full(nk, False)
    for k in range(nk):
        dummd1 = np.exp(logsumexp(probs[idxroot == cluster_centers[k]]) - normpks)
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
            dummd2 = DIST_METRICS[metric](
                X[idxroot[dummd1yi1]], X[idxroot[j]], cell=cell
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


def generate_probability_model(
    cluster_center_idx: np.ndarray,
    labels: np.ndarray,
    X: np.ndarray,
    descriptors: np.ndarray,
    descriptor_labels: np.ndarray,
    descriptor_weights: np.ndarray,
    normkernels: np.ndarray,
    h_invs: np.ndarray,
    probs: np.ndarray,
    normpks: float,
    cell: np.ndarray = None,
    nmsopt: int = 0,
):
    """
    Generates a probability model based on the given inputs.

    Parameters:
        None

    Returns:
        None
    """

    def _mean_shift_optimizaton(
        mean: np.ndarray,
        X: np.array,
        h_inv: np.ndarray,
        normkernel: float,
        probs: np.ndarray,
    ):
        # Never tested and not used in any available example cases
        grid = np.copy(mean)
        for _ in range(nmsopt):
            # Mean shift optimization
            msmu = np.zeros(X.shape[1], dtype=float)
            tmppks = -np.inf
            dummd1s = pairwise_mahalanobis_distances(
                X,
                grid[np.newaxis, ...],
                h_inv,
                cell,
                squared=True,
            )[0]
            msws = -0.5 * (normkernel + dummd1s) + probs
            tmpmsmu = rij(cell, X, grid)
            msmu += np.sum(np.exp(msws) * tmpmsmu, axis=1)
            tmppks = logsumexp(np.concatenate([tmppks, msws]))
            mean += msmu / np.exp(tmppks)

        return mean

    def _update_cluster_cov(
        X: np.ndarray,
        k: int,
        sample_labels: np.ndarray,
        probs: np.ndarray,
        idxroot: np.ndarray,
        center_idx: np.ndarray,
    ):

        if cell is not None:
            cov = _get_lcov_clusterp(
                len(X), nsamples, X, idxroot, center_idx[k], probs, cell
            )
            if np.sum(idxroot == center_idx[k]) == 1:
                cov = _get_lcov_clusterp(
                    nsamples,
                    nsamples,
                    descriptors,
                    sample_labels,
                    center_idx[k],
                    descriptor_weights,
                    cell,
                )
                print("Warning: single point cluster!")
        else:
            cov = _get_lcov_cluster(len(X), X, idxroot, center_idx[k], probs, cell)
            if np.sum(idxroot == center_idx[k]) == 1:
                cov = _get_lcov_cluster(
                    nsamples,
                    descriptors,
                    sample_labels,
                    center_idx[k],
                    descriptor_weights,
                    cell,
                )
                print("Warning: single point cluster!")
            cov = oas(
                cov,
                logsumexp(probs[idxroot == center_idx[k]]) * nsamples,
                X.shape[1],
            )

        return cov

    def _get_lcov_cluster(
        N: int,
        x: np.ndarray,
        clroots: np.ndarray,
        idcl: int,
        probs: np.ndarray,
        cell: np.ndarray,
    ):

        ww = np.zeros(N)
        normww = logsumexp(probs[clroots == idcl])
        ww[clroots == idcl] = np.exp(probs[clroots == idcl] - normww)
        cov = covariance(x, ww, cell)

        return cov

    def _get_lcov_clusterp(
        N: int,
        Ntot: int,
        x: np.ndarray,
        clroots: np.ndarray,
        idcl: int,
        probs: np.ndarray,
        cell: np.ndarray,
    ):

        ww = np.zeros(N)
        totnormp = logsumexp(probs)
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

    nclusters = len(cluster_center_idx)
    nsamples = len(descriptors)
    dimension = X.shape[1]
    cluster_mean = np.zeros((nclusters, dimension), dtype=float)
    cluster_cov = np.zeros((nclusters, dimension, dimension), dtype=float)
    cluster_weight = np.zeros(nclusters, dtype=float)
    center_idx = np.unique(labels)

    for k in range(nclusters):
        cluster_weight[k] = np.exp(logsumexp(probs[labels == center_idx[k]]) - normpks)
        cluster_mean[k] = _mean_shift_optimizaton(
            X[center_idx[k]],
            X,
            h_invs[center_idx[k]],
            normkernels[center_idx[k]],
            probs,
        )
        cluster_cov[k] = _update_cluster_cov(
            X, k, descriptor_labels, probs, labels, center_idx
        )
    for k in range(nclusters):
        labels[labels == center_idx[k]] = k + 1

    return cluster_weight, cluster_mean, cluster_cov, labels


def cluster_distribution_3D(
    grids: np.ndarray,
    grid_weights: np.ndarray,
    grid_label_: np.ndarray = None,
    use_index: list[int] = None,
    label_text: list[str] = None,
    size_scale: float = 1e4,
    fig_size: tuple[int, int] = (12, 12),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generate a 3D scatter plot of the cluster distribution.

    Parameters:
        grids (numpy.ndarray): The array containing the grid data.
        use_index (Optional[list[int]]): The indices of the features to use for the
            scatter plot.
            If None, the first three features will be used.
        label_text (Optional[list[str]]): The labels for the x, y, and z axes.
            If None, the labels will be set to
            'Feature 0', 'Feature 1', and 'Feature 2'.
        size_scale (float): The scale factor for the size of the scatter points.
            Default is 1e4.
        fig_size (tuple[int, int]): The size of the figure. Default is (12, 12)

    Returns:
        tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib
            Figure and Axes objects.
    """
    if use_index is None:
        use_index = [0, 1, 2]
    if label_text is None:
        label_text = [f"Feature {i}" for i in range(3)]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=fig_size, dpi=100)
    scatter = ax.scatter(
        grids[:, use_index[0]],
        grids[:, use_index[1]],
        grids[:, use_index[2]],
        c=grid_label_,
        s=grid_weights * size_scale,
    )
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Gaussian")
    ax.add_artist(legend1)
    ax.set_xlabel(label_text[0])
    ax.set_ylabel(label_text[1])
    ax.set_zlabel(label_text[2])

    return fig, ax


# %% [markdown]
# We first load our dataset:
#
#

# %%
hbond_data = load_hbond_dataset()
descriptors = hbond_data["descriptors"]
weights = hbond_data["weights"]

# %%
ngrid = int(len(descriptors) ** 0.5)
selector = FPS(initialize=26310, n_to_select=ngrid)
selector.fit(descriptors.T)
selector.selected_idx_
grids = descriptors[selector.selected_idx_]

# %%
estimator = SparseKDE(descriptors, weights)
estimator.fit(grids)

# %% [markdown]
# Now we visualize the distribution and the weight of clusters.
# %%
#
#

# %%
cluster_distribution_3D(
    grids, estimator._sample_weights, label_text=[r"$\nu$", r"$\mu$", r"r"]
)

# %% [markdown]
# We need to estimate the probability at each grid point to do quick shift, which can
# further partition the set of grid points in to several clusters. The resulting
# clusters can be interpreted as (meta-)stable states of the system.
#
#

# %%
probs = estimator.score_samples(grids)
clustering = QuickShift(
    estimator._qscut2,
    estimator.gs,
    metric=estimator.metric,
    metric_params=estimator.metric_params,
)
clustering.fit(grids, samples_weight=probs)
cluster_centers_idx = clustering.cluster_centers_idx_
labels = clustering.labels_
normpks = logsumexp(probs)

cluster_centers, labels = quick_shift_refinement(
    grids,
    cluster_centers_idx,
    labels,
    probs,
    normpks,
    estimator.metric,
    estimator.cell,
)

# %% [markdown]
# Based on the results, the gaussian mixture model of the system can be generated:
#
#

# %%
cluster_weights, cluster_means, cluster_covs, labels = generate_probability_model(
    cluster_centers_idx,
    labels,
    grids,
    estimator.descriptors,
    estimator._sample_labels_,
    estimator.weights,
    estimator._normkernels,
    estimator._h_invs,
    probs,
    normpks,
    estimator.cell,
)

# %% [markdown]
# The final result shows seven (meta-)stable states of hydrogen bond. Here we also show
# the reference hydrogen bond descriptor. The gaussian with the largest weight locates
# closest to the reference point.
#
#

# %%
REF_HB = np.array([0.82, 2.82, 2.74])

fig, ax = cluster_distribution_3D(
    grids, estimator._sample_weights, labels, label_text=[r"$\nu$", r"$\mu$", r"r"]
)
ax.scatter(REF_HB[0], REF_HB[1], REF_HB[2], marker="+", color="red", s=1000)

# %%
f"The gaussian with the highest probability is {np.argmax(cluster_weights) + 1}"
