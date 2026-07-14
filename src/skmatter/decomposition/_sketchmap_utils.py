"""Sigmoid transformation and parameter-estimation utilities for Sketch-Map."""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import curve_fit


# Sigmoid transformation functions


def sigmoid_transform(distances, sigma, a, b):
    r"""Apply the Sketch-Map sigmoid transformation to distances.

    The sigmoid function maps pairwise distances to the range [0, 1], with
    the switching point at :math:`r = \sigma` (where :math:`s(\sigma) = 0.5`).

    The transformation effectively "compresses" both very small and very large
    distances, focusing the optimization on intermediate-range structure.

    Mathematical form:

    .. math::

        s(r) = 1 - \left(1 + A \cdot \left(\frac{r}{\sigma}\right)^a\right)^{-b/a}

    where :math:`A = 2^{a/b} - 1` ensures :math:`s(\sigma) = 0.5`.

    Parameters
    ----------
    distances : ndarray
        Pairwise distances to transform.
    sigma : float
        Switching distance where :math:`s(\sigma) = 0.5`. This is the
        "characteristic scale" of the transformation.
    a : float
        Short-range exponent controlling how quickly :math:`s(r) \to 0` as
        :math:`r \to 0`. Larger values make the sigmoid steeper for distances
        below sigma.
    b : float
        Long-range exponent controlling how quickly :math:`s(r) \to 1` as
        :math:`r \to \infty`. Larger values make the sigmoid steeper for
        distances above sigma.
    """
    A = 2.0 ** (a / b) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        transformed = 1.0 - (1.0 + A * (distances / sigma) ** a) ** (-b / a)
        transformed[distances <= 0.0] = 0.0
    return transformed


def sigmoid_inverse(y, sigma, a, b):
    r"""Compute the inverse of the sigmoid transformation.

    Maps a transformed distance value back to the original distance. This function is
    useful for interpreting transformed distances in terms of the original distance
    scale.

    Parameters
    ----------
    y : ndarray
        Transformed distances in [0, 1].
    sigma : float
        Switching distance where :math:`s(\sigma) = 0.5`.
    a : float
        Short-range steepness parameter.
    b : float
        Long-range steepness parameter.
    """
    A = 2.0 ** (a / b) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        one_minus_y = np.maximum(1.0 - y, 1e-12)
        inner = np.power(one_minus_y, -a / b) - 1.0
        distances = sigma * np.power(np.maximum(inner / A, 0.0), 1.0 / a)
        distances[y <= 0.0] = 0.0
        distances[y >= 1.0] = np.inf
    return distances


def sigmoid_derivative(distances, sigma, a, b):
    r"""Compute the derivative of the sigmoid with respect to distance.

    This derivative is required for gradient-based optimization of the Sketch-Map
    stress function.

    Parameters
    ----------
    distances : ndarray
        Pairwise distances.
    sigma : float
        Switching distance where :math:`s(\sigma) = 0.5`.
    a : float
        Short-range steepness parameter.
    b : float
        Long-range steepness parameter.

    Returns
    -------
    derivative : ndarray
        The value :math:`\mathrm{d}s/\mathrm{d}r` at each distance point.
    """
    return _sigmoid_fdf(np.asarray(distances, dtype=float), sigma, a, b)[1]


def _sigmoid_fdf(distances, sigma, a, b):
    r"""Evaluate the sigmoid and its derivative in one pass.

    Shares the expensive power evaluations between the value

    .. math:: s(r) = 1 - (1 + u)^{-b/a}, \qquad u = A (r/\sigma)^a

    and the derivative

    .. math:: s'(r) = \frac{b A}{\sigma} (r/\sigma)^{a-1}
              \frac{(1 + u)^{-b/a}}{1 + u}

    Both are defined to be 0 at :math:`r \le 0` (the C++ implementation never
    evaluates the derivative there; returning 0 keeps overlapping points from
    producing NaNs).

    Returns
    -------
    f, df : ndarray
        Sigmoid value and derivative, same shape as ``distances``.
    """
    A = 2.0 ** (a / b) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        r = distances / sigma
        u = A * r**a
        decay = (1.0 + u) ** (-b / a)  # this is 1 - s(r)
        f = 1.0 - decay
        df = (b * A / sigma) * r ** (a - 1.0) * decay / (1.0 + u)

    nonpositive = distances <= 0.0
    if np.any(nonpositive):
        f[nonpositive] = 0.0
        df[nonpositive] = 0.0
    return f, df


# Utility functions


def classical_mds(distances, n_components):
    """Compute a classical MDS embedding as initialization.

    Classical MDS finds coordinates that best reproduce the given distance matrix using
    eigendecomposition of the doubly-centered Gram matrix.

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        Symmetric pairwise distance matrix.
    n_components : int
        Number of dimensions for the embedding.
    """
    # Double-centering to obtain the Gram matrix (distances**2 is a copy, so
    # the caller's distance matrix is never modified)
    gram_matrix = distances**2
    gram_matrix *= -0.5
    gram_matrix -= gram_matrix.mean(axis=0, keepdims=True)
    gram_matrix -= gram_matrix.mean(axis=1, keepdims=True)

    n = gram_matrix.shape[0]
    k = min(n_components, n)
    eigenvalues, eigenvectors = eigh(gram_matrix, subset_by_index=[n - k, n - 1])

    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    coordinates = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))

    # Consistent sign convention: largest-magnitude entry of each column positive
    for i in range(coordinates.shape[1]):
        col = coordinates[:, i]
        if col[np.argmax(np.abs(col))] < 0:
            coordinates[:, i] *= -1

    return coordinates


def _gaussian(x, amplitude, center, std_dev):
    """Gaussian function for curve fitting in distance distribution analysis."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * std_dev**2))


def _maybe_tqdm(iterable, enabled, **tqdm_kwargs):
    """Wrap an iterable in a tqdm progress bar when ``enabled``.

    tqdm is an optional dependency, so it is imported lazily and a clear
    error is raised only when a progress bar was actually requested.
    """
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError as error:
        raise ImportError(
            "progress_bar=True requires the optional dependency 'tqdm'; "
            "install it with 'pip install tqdm'"
        ) from error
    return tqdm(iterable, **tqdm_kwargs)


# Distance distribution analysis


def analyze_distance_distribution(distances, n_bins=200):
    """Analyze the distribution of pairwise distances for parameter estimation.

    This function computes a histogram of the pairwise distances and identifies key
    features that inform the choice of sigmoid parameters. 1. Peak distance (where the
    bulk of pairwise distances lie, this represents the characteristic scale of the
    data.) 2. Gaussian range (the regime dominated by short-range fluctuations,
    distances below this should be compressed). 3. Uniform cutoff (where the
    high-dimensional "curse of dimensionality" causes distances to become
    uninformative).

    Parameters
    ----------
    distances : ndarray
        Pairwise distance matrix (square symmetric) or flattened upper triangle.
    n_bins : int, default=200
        Number of histogram bins for the analysis.
    """
    if distances.ndim == 2:
        d = distances[np.triu_indices_from(distances, k=1)]
    else:
        d = distances.copy()

    d = d[np.isfinite(d) & (d >= 0)]
    if d.size == 0:
        raise ValueError("Empty or invalid distances array")

    max_distance = np.percentile(d, 99.9)
    bin_edges = np.linspace(0, max_distance, n_bins + 1)
    prob_density, _ = np.histogram(d, bins=bin_edges, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    peak_idx = np.argmax(prob_density)
    peak_distance = bin_centers[peak_idx]

    analysis = {
        "peak_distance": peak_distance,
        "gaussian_std": None,
        "gaussian_range": None,
        "uniform_cutoff": None,
        "bin_centers": bin_centers,
        "prob_density": prob_density,
        "max_distance": max_distance,
    }

    # Estimate the Gaussian fluctuation range by fitting the left side of the
    # peak; this characterizes the short-range noise regime.
    left_mask = bin_centers <= peak_distance
    if np.sum(left_mask) > 3:
        try:
            initial_guess = [np.max(prob_density), peak_distance, 1.0]
            optimal_params, _ = curve_fit(
                _gaussian,
                bin_centers[left_mask],
                prob_density[left_mask],
                p0=initial_guess,
                maxfev=5000,
            )
            analysis["gaussian_std"] = abs(optimal_params[2])
            # Gaussian range extends ~3 sigma from the peak
            analysis["gaussian_range"] = peak_distance + 3 * analysis["gaussian_std"]
        except (RuntimeError, ValueError):
            analysis["gaussian_std"] = peak_distance / 3.0
            analysis["gaussian_range"] = peak_distance * 2.0

    # Estimate the high-dimensional cutoff as the right-side distance where the
    # density drops to 10% of the peak value.
    right_mask = bin_centers > peak_distance
    right_density = prob_density[right_mask]

    if len(right_density) > 3:
        threshold = 0.1 * prob_density[peak_idx]
        below_threshold = right_density < threshold

        if np.any(below_threshold):
            first_below = np.argmax(below_threshold)
            analysis["uniform_cutoff"] = bin_centers[right_mask][first_below]
        else:
            analysis["uniform_cutoff"] = np.percentile(d, 90)

    # Ensure gaussian_range < uniform_cutoff
    if (
        analysis["gaussian_range"] is not None
        and analysis["uniform_cutoff"] is not None
        and analysis["gaussian_range"] >= analysis["uniform_cutoff"]
    ):
        analysis["gaussian_range"] = peak_distance + 0.2 * max_distance
        analysis["uniform_cutoff"] = peak_distance + 0.6 * max_distance

    return analysis


def suggest_sigmoid_params(distances, n_components, n_bins=200):
    r"""Suggest sigmoid parameters based on distance distribution analysis.

    This function implements heuristics from the Sketch-Map guidelines for automatic
    parameter selection.

    Parameter selection strategy:

    - ``sigma``: placed just before the peak (90% of peak distance) to ensure the bulk
      of distances fall in the sigmoid's sensitive region.

    - ``a_high``, ``b_high`` (high-D sigmoid): ``a_high`` sets the short-range
      exponent and ``b_high`` the long-range one. Since :math:`1 - s(r) \propto
      r^{-b}` for :math:`r \gg \sigma`, a *small* ``b_high`` is what keeps the
      long-distance tail long, so ``b_high`` must stay well below ``a_high``
      (the published choices are :math:`A=8, B=2` for LJ38 and :math:`A=4, B=2`
      for MAD). Inverting the two makes the filter saturate too early, which
      flings outliers outward and inflates the map without bound.

    - ``a_low``, ``b_low`` (low-D sigmoid): set to the embedding dimensionality
      ``n_components`` (2 for a 2D map), the standard Sketch-Map choice. The
      volume-equalization rule :math:`a_{\text{low}} \cdot d \approx
      a_{\text{high}} \cdot D` motivates a small low-D exponent but diverges for
      :math:`D \gg d`, so the conventional value is used directly.

    Parameters
    ----------
    distances : ndarray
        Pairwise distance matrix.
    n_components : int
        Target embedding dimensionality.
    n_bins : int, default=200
        Number of histogram bins for distance analysis.

    Returns
    -------
    params : dict
        Suggested parameters with keys: ``sigma``, ``a_high``, ``b_high``,
        ``a_low``, ``b_low``.
    analysis : dict
        Distance distribution analysis results from
        :func:`analyze_distance_distribution`.
    """
    analysis = analyze_distance_distribution(distances, n_bins=n_bins)

    sigma = 0.9 * analysis["peak_distance"]

    if (
        analysis["gaussian_range"] is not None
        and analysis["uniform_cutoff"] is not None
    ):
        range_ratio = analysis["uniform_cutoff"] / max(
            analysis["gaussian_range"], 1e-10
        )
        b_high = np.clip(2.0 + np.log(range_ratio), 2.0, 6.0)
        a_high = np.clip(b_high * 2, 4.0, 12.0)
    else:
        a_high = 6.0
        b_high = 2.0

    a_low = b_low = float(max(n_components, 1))

    params = {
        "sigma": sigma,
        "a_high": a_high,
        "b_high": b_high,
        "a_low": a_low,
        "b_low": b_low,
    }

    return params, analysis
