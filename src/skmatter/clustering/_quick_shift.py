from typing import Callable, Optional

import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm

from ..metrics._pairwise import periodic_pairwise_euclidean_distances


class QuickShift(BaseEstimator):
    """Conducts quick shift clustering.

    This class is used to implement the quick shift clustering algorithm,
    which is used in probabilistic analysis of molecular motifs (PAMM). There
    are two ways of searching the next point: (1) search for the point within the given
    distance cutoff and (2) search for the point within the given number of neighbor
    shell of the Gabriel graph. If both of them are set, the distance cutoff
    is used.

    Parameters
    ----------
    dist_cutoff_sq : float, default=None
        The squared distance cutoff for searching for the next point. Two points are
        considered as neighbors if they are within this distance. If :obj:`None`,
        the scheme of Gabriel graph is used.
    gabriel_shell : int, default=None
        The number of neighbor shell of Gabriel graph for searching for the next point.
        For example, if the number is 1, two points will be considered as neighbors if
        they have at least one common neighbor, like for the case "A-B-C", we will
        consider "A-C" as neighbors. If the number is 2, for the case "A-B-C-D",
        we will consider "A-D" as neighbors. If :obj:`None`, the scheme of distance
        cutoff is used.
    scale : float, default=1.0
        Distance cutoff scaling factor used during the QS clustering. It will be squared
        since the squared distance is used in this class.
    metric : Callable, default=None
        The metric to use. Your metric should be able to take at least three arguments
        in secquence: `X`, `Y`, and `squared=True`. Here, `X` and `Y` are two array-like
        of shape (n_samples, n_components). The return of the metric is an array-like of
        shape (n_samples, n_samples). If you want to use periodic boundary
        conditions, be sure to provide the cell length in the ``metric_params`` and
        provide a metric that can take the cell argument. If :obj:`None`, the
        :func:`skmatter.metrics.periodic_pairwise_euclidean_distances()` is used.
    metric_params : dict, default=None
        Additional parameters to be passed to the use of
        metric.  i.e. the dimension of a rectangular cell of side length :math:`a_i`
        for :func:`skmatter.metrics.periodic_pairwise_euclidean_distances()`
        ``{'cell_length': [a_1, a_2, ..., a_n]}``

    Attributes
    ----------
    labels_ : numpy.ndarray
        An array of labels for each input data.
    cluster_centers_idx_ : numpy.ndarray
        An array of indices of cluster centers.
    cluster_centers_ : numpy.ndarray
        An array of cluster centers.

    Examples
    --------
    >>> import numpy as np
    >>> from skmatter.clustering import QuickShift

    Create some points and their weights for quick shift clustering

    >>> feature1 = np.array([-1.72, -4.44, 0.54, 3.19, -1.13, 0.55])
    >>> feature2 = np.array([-1.32, -2.13, -2.43, -0.49, 2.33, 0.18])
    >>> points = np.vstack((feature1, feature2)).T
    >>> weights = np.array([-3.94, -12.68, -7.07, -9.03, -8.26, -2.61])

    Set cutoffs for seraching

    >>> cuts = np.array([6.99, 8.80, 7.68, 9.51, 8.07, 6.22])

    Do the clustering

    >>> model = QuickShift(cuts).fit(points, samples_weight=weights)
    >>> print(model.labels_)
    [0 0 0 5 5 5]
    >>> print(model.cluster_centers_idx_)
    [0 5]

    We can also apply a periodic boundary condition

    >>> model = QuickShift(cuts, metric_params={"cell_length": [3, 3]})
    >>> model = model.fit(points, samples_weight=weights)
    >>> print(model.labels_)
    [5 5 5 5 5 5]
    >>> print(model.cluster_centers_idx_)
    [5]

    Since the searching cuts are all larger than the maximum distance in the PBC box,
    it can be expected that all points are assigned to the same cluster, of the center
    that has the largest weight.
    """

    def __init__(
        self,
        dist_cutoff_sq: Optional[float] = None,
        gabriel_shell: Optional[int] = None,
        scale: float = 1.0,
        metric: Optional[Callable] = None,
        metric_params: Optional[dict] = None,
    ):
        if (dist_cutoff_sq is None) and (gabriel_shell is None):
            raise ValueError("Either dist_cutoff or gabriel_depth must be set.")
        self.dist_cutoff_sq = dist_cutoff_sq
        self.gabriel_shell = gabriel_shell
        self.scale = scale
        if self.dist_cutoff_sq is not None:
            self.dist_cutoff_sq *= self.scale**2
        self.metric_params = (
            metric_params if metric_params is not None else {"cell_length": None}
        )

        if metric is None:
            metric = periodic_pairwise_euclidean_distances

        self.metric = lambda X, Y: metric(X, Y, squared=True, **self.metric_params)
        if isinstance(self.metric_params, dict):
            self.cell = self.metric_params["cell_length"]
        else:
            self.cell = None

    def fit(self, X, y=None, samples_weight=None):
        """Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        Y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        samples_weight : array-like of shape (n_samples,), default=None
            List of sample weights attached to the data X. This parameter
        must be given in order to do the quick shift clustering.
        """
        if (self.cell is not None) and (X.shape[1] != len(self.cell)):
            raise ValueError(
                "Dimension of the cell length does not match the data dimension."
            )
        dist_matrix = self.metric(X, X)
        np.fill_diagonal(dist_matrix, np.inf)
        if self.dist_cutoff_sq is None:
            gabrial = _get_gabriel_graph(dist_matrix)
        idmindist = np.argmin(dist_matrix, axis=1)
        idxroot = np.full(dist_matrix.shape[0], -1, dtype=int)
        for i in tqdm(range(dist_matrix.shape[0]), desc="Quick-Shift"):
            if idxroot[i] != -1:
                continue
            qspath = []
            qspath.append(i)
            current = qspath[-1]
            while current != idxroot[current]:
                if self.gabriel_shell is not None:
                    idxroot[current] = self._gs_next(
                        current, samples_weight, dist_matrix, gabrial
                    )
                else:
                    idxroot[current] = self._qs_next(
                        current,
                        idmindist[current],
                        samples_weight,
                        dist_matrix,
                        self.dist_cutoff_sq[current],
                    )
                if idxroot[idxroot[current]] != -1:
                    # Found a path to a root
                    break
                qspath.append(idxroot[current])
                current = qspath[-1]
            idxroot[qspath] = idxroot[idxroot[current]]

        self.labels_ = idxroot
        self.cluster_centers_idx_ = np.concatenate(
            np.argwhere(idxroot == np.arange(dist_matrix.shape[0]))
        )
        self.cluster_centers_ = X[self.cluster_centers_idx_]

        return self

    def _gs_next(
        self,
        idx: int,
        probs: np.ndarray,
        distmm: np.ndarray,
        gabriel: np.ndarray,
    ):
        """Find next cluster in Gabriel graph."""
        ngrid = len(probs)
        neighs = np.copy(gabriel[idx])
        for _ in range(1, self.gabriel_shell):
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

    def _qs_next(
        self, idx: int, idxn: int, probs: np.ndarray, distmm: np.ndarray, cutoff: float
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


def _get_gabriel_graph(dist_matrix_sq: np.ndarray):
    """
    Generate the Gabriel graph based on the given squared distance matrix.

    Parameters
    ----------
    dist_matrix_sq : np.ndarray
        The squared distance matrix of shape (n_points, n_points).

    Returns
    -------
    np.ndarray
        The Gabriel graph matrix of shape (n_points, n_points).
    """
    n_points = dist_matrix_sq.shape[0]
    gabriel = np.full((n_points, n_points), True)
    for i in tqdm(range(n_points), desc="Calculating Gabriel graph"):
        gabriel[i, i] = False
        for j in range(i, n_points):
            if np.sum(dist_matrix_sq[i] + dist_matrix_sq[j] < dist_matrix_sq[i, j]):
                gabriel[i, j] = False
                gabriel[j, i] = False

    return gabriel
