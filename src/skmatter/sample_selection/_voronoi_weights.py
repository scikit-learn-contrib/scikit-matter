"""Voronoi-based weighting for landmark points.

Provides :func:`voronoi_weights`, a small utility for computing the
"density weight" of each landmark by counting how many points of the full
dataset fall in its Voronoi cell. These weights are the standard input to
weighted Sketch-Map fits (see [Mazitov2025a]_) and to any other
landmark-based dimensionality reduction where dense regions should drive
the embedding more strongly than sparse outliers.
"""

import numpy as np
from sklearn.metrics import pairwise_distances

# Cap on the (block x n_landmarks) distance matrix
_CHUNK_ELEMENTS = 40_000_000


def voronoi_weights(X_full, X_landmarks, sample_weights=None, normalize=True):
    """Compute per-landmark Voronoi weights.

    Each row of ``X_full`` is assigned to its nearest landmark in
    ``X_landmarks``; the weight of a landmark is the (possibly weighted)
    count of points in its Voronoi cell. By default the resulting weights
    are normalized to sum to 1, which matches the weight column written by
    the C++ ``dimlandmark -mode minmax -w`` tool used in the MAD paper.

    Parameters
    ----------
    X_full : array-like of shape (n_samples, n_features)
        The full dataset from which the landmarks were drawn.
    X_landmarks : array-like of shape (n_landmarks, n_features)
        The selected landmark points (e.g. via
        :class:`skmatter.sample_selection.FPS`).
    sample_weights : array-like of shape (n_samples,) or None, default=None
        Optional per-row weight applied when summing ``X_full`` rows into
        their Voronoi cells. ``None`` is equivalent to all-ones (raw counts).
    normalize : bool, default=True
        If True, divide the resulting weights so they sum to 1. The C++
        ``-mode minmax -w`` writer normalizes; set to False to recover raw
        (weighted) counts instead.

    Returns
    -------
    weights : ndarray of shape (n_landmarks,)
        Voronoi weight of each landmark, in the same order as ``X_landmarks``.

    Examples
    --------
    >>> import numpy as np
    >>> from skmatter.sample_selection import FPS, voronoi_weights
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=500, n_features=4, centers=4, random_state=0)
    >>> fps = FPS(n_to_select=20, random_state=0).fit(X)
    >>> X_landmarks = X[fps.selected_idx_]
    >>> w = voronoi_weights(X, X_landmarks)
    >>> w.shape
    (20,)
    >>> bool(np.isclose(w.sum(), 1.0))
    True
    """
    X_full = np.asarray(X_full)
    X_landmarks = np.asarray(X_landmarks)
    if sample_weights is None:
        sample_weights = np.ones(X_full.shape[0])
    else:
        sample_weights = np.asarray(sample_weights, dtype=float)
        if sample_weights.shape[0] != X_full.shape[0]:
            raise ValueError(
                f"sample_weights has length {sample_weights.shape[0]} but "
                f"X_full has {X_full.shape[0]} rows."
            )

    n_landmarks = X_landmarks.shape[0]
    chunk = max(1, _CHUNK_ELEMENTS // max(n_landmarks, 1))

    # accumulate cell populations in row-blocks
    weights = np.zeros(n_landmarks, dtype=float)
    for start in range(0, X_full.shape[0], chunk):
        block = X_full[start : start + chunk]
        assignments = np.argmin(pairwise_distances(block, X_landmarks), axis=1)
        weights += np.bincount(
            assignments,
            weights=sample_weights[start : start + chunk],
            minlength=n_landmarks,
        )

    if normalize:
        total = weights.sum()
        if total > 0:
            weights /= total

    return weights
