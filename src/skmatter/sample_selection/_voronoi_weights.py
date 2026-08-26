import numpy as np
from sklearn.metrics import pairwise_distances_argmin


def voronoi_weights(X_full, X_landmarks, normalize=True, power=1.0):
    """Compute per-landmark Voronoi weights.

    Each row of ``X_full`` is assigned to its nearest landmark in ``X_landmarks``. The
    weight of a landmark is the count of points in its Voronoi cell, that is how many
    rows of ``X_full`` it stands for. By default the weights are normalized to sum to 1.

    Landmark selectors such as :class:`skmatter.sample_selection.FPS` choose points for
    coverage rather than for density, so the landmarks alone no longer say how the
    original dataset was distributed. These weights put that information back, letting a
    densely sampled region influence a subsequent fit in proportion to its population.

    Parameters
    ----------
    X_full : array-like of shape (n_samples, n_features)
        The full dataset from which the landmarks were drawn
    X_landmarks : array-like of shape (n_landmarks, n_features)
        The selected landmark points (e.g. via :class:`skmatter.sample_selection.FPS`)
    normalize : bool, default=True
        If True, divide the resulting weights so they sum to 1
    power : float, default=1.0
        Exponent applied to the cell populations before normalization, interpolating
        between uniform coverage of the landmarks (``power=0``) and the density of the
        full collection (``power=1``), as in [Ceriotti2013]_. Exponents between the two
        damp the dominance of very dense cells, useful when full density would drown out
        the sparse extremes of the dataset

    Returns
    -------
    weights : ndarray of shape (n_landmarks,)
        Voronoi weight of each landmark, in the same order as ``X_landmarks``

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

    With ``power=0`` every landmark gets the same weight. Otherwise the heaviest
    landmark exceeds the lightest by their population ratio raised to ``power``, so
    lowering ``power`` evens the weights out:

    >>> uniform = voronoi_weights(X, X_landmarks, power=0.0)
    >>> bool(np.allclose(uniform, 1 / 20))
    True
    >>> float(round(w.max() / w.min(), 4))
    11.0
    >>> damped = voronoi_weights(X, X_landmarks, power=0.5)
    >>> float(round(damped.max() / damped.min(), 4))
    3.3166
    """
    if power < 0:
        raise ValueError(f"power must be non-negative, got {power}")

    X_full = np.asarray(X_full)
    X_landmarks = np.asarray(X_landmarks)

    n_landmarks = X_landmarks.shape[0]

    assignments = pairwise_distances_argmin(X_full, X_landmarks)
    weights = np.bincount(assignments, minlength=n_landmarks).astype(float)

    if power != 1.0:
        weights = weights**power

    if normalize:
        weights /= weights.sum()

    return weights
