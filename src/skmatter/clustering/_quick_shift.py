from typing import Union

import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm

from ..metrics import DIST_METRICS


class QuickShift(BaseEstimator):
    """TODO"""

    def __init__(
        self,
        dist_cutoff2: Union[float, None] = None,
        gabriel_shell: Union[int, None] = None,
        metric: str = "periodic_euclidean",
        metric_params: Union[dict, None] = None,
    ):
        if (dist_cutoff2 is None) and (gabriel_shell is None):
            raise ValueError("Either dist_cutoff or gabriel_depth must be set.")
        self.dist_cutoff2 = dist_cutoff2
        self.gabriel_shell = gabriel_shell
        self.metric = metric
        self.metric_params = metric_params
        if isinstance(self.metric_params, dict):
            self.cell = self.metric_params["cell"]
        else:
            self.cell = None

    def fit(self, X, y=None, samples_weight=None):

        dist_matrix = DIST_METRICS[self.metric](X, X, squared=True, cell=self.cell)
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


def _get_gabriel_graph(dist_matrix2: np.ndarray):
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
