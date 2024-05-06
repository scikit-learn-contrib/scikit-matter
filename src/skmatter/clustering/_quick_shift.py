from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from tqdm import tqdm

from ..metrics._pairwise import pairwise_euclidean_distances


class QuickShift(BaseEstimator):
    """Conducts quick shift clustering.

    This class is used to implement the quick shift clustering algorithm,
    which is used in probabilistic analysis of molecular motifs (PAMM). There
    are two ways of searching the next point: (1) search for the point within the given
    distance cutoff and (2) search for the point within the given number of neighbor
    shell of the gabriel graph. If both of them are set, the distance cutoff
    is used.

    Parameters
    ----------
    dist_cutoff_sq : float, default=None
        The squared distance cutoff for searching for the next point. Two points are
        considered as neighbors if they are within this distance. If :obj:`None`,
        the scheme of Gabriel graph is used.
    gabriel_shell : int, default=None
        The number of neighbor shell of gabriel graph for searching for the next point.
        For example, if the number is 1, two points will be considered as neighbors if
        they have at least one common neighbor, like for the case "A-B-C", we will
        consider "A-C" as neighbors. If the number is 2, for the case "A-B-C-D",
        we will consider "A-D" as neighbors. If :obj:`None`, the scheme of distance
        cutoff is used.
    scale : float, default=1.0
        Distance cutoff scaling factor used during the QS clustering. It will be squared
        since the squared distance is used in this class.
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
        metric.  i.e. the cell dimension
        for :func:`skmatter.metrics.pairwise_euclidean_distances()`
        `{'cell': [side_length_1, ..., side_length_n]}`

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
    """

    def __init__(
        self,
        dist_cutoff_sq: Union[float, None] = None,
        gabriel_shell: Union[int, None] = None,
        scale: float = 1.0,
        metric: Callable[
            [ArrayLike, ArrayLike, bool, dict], ArrayLike
        ] = pairwise_euclidean_distances,
        metric_params: Union[dict, None] = None,
    ):
        if (dist_cutoff_sq is None) and (gabriel_shell is None):
            raise ValueError("Either dist_cutoff or gabriel_depth must be set.")
        self.dist_cutoff2 = dist_cutoff_sq
        self.gabriel_shell = gabriel_shell
        self.scale = scale
        if self.dist_cutoff2 is not None:
            self.dist_cutoff2 *= self.scale**2
        self.metric_params = (
            metric_params if metric_params is not None else {"cell": None}
        )
        self.metric = lambda X, Y: metric(X, Y, squared=True, **self.metric_params)
        if isinstance(self.metric_params, dict):
            self.cell = self.metric_params["cell"]
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
            must be given in order to do the quick shift clustering."""

        if (self.cell is not None) and (X.shape[1] != len(self.cell)):
            raise ValueError("Cell dimension does not match the data dimension.")
        dist_matrix = self.metric(X, X)
        if self.dist_cutoff2 is None:
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
                        self.dist_cutoff2[current],
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
