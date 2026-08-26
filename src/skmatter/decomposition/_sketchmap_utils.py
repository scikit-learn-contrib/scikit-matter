import numpy as np
from scipy.linalg import eigh
from scipy.optimize import curve_fit


def _sigmoid_transform(distances, sigma, a, b):
    r"""Apply the sketch-map sigmoid transformation to distances.

    The sigmoid maps pairwise distances into [0, 1] crossing 0.5 at the switching
    distance :math:`\sigma`. Distances well below and well above :math:`\sigma` are
    pushed toward 0 and 1, so the fit is driven by the intermediate range.

    .. math::

        s(r) = 1 - \left(1 + A \cdot \left(\frac{r}{\sigma}\right)^a\right)^{-b/a}

    where :math:`A = 2^{a/b} - 1` ensures :math:`s(\sigma) = 0.5`.

    Parameters
    ----------
    distances : ndarray
        Pairwise distances to transform
    sigma : float
        Switching distance where :math:`s(\sigma) = 0.5`. This is the "characteristic
        scale" of the transformation
    a : float
        Short-range exponent controlling how quickly :math:`s(r) \to 0` as :math:`r \to
        0`. Larger values make the sigmoid steeper for distances below sigma
    b : float
        Long-range exponent controlling how quickly :math:`s(r) \to 1` as :math:`r \to
        \infty`. Larger values make the sigmoid steeper for distances above sigma

    Returns
    -------
    transformed : ndarray
        Sigmoid values in [0, 1], same shape as ``distances``
    """
    A = 2.0 ** (a / b) - 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        transformed = 1.0 - (1.0 + A * (distances / sigma) ** a) ** (-b / a)
        transformed[distances <= 0.0] = 0.0

    return transformed


def _sigmoid_and_derivative(distances, sigma, a, b):
    r"""Evaluate the sigmoid and its derivative in one pass.

    The stress and its gradient need both, and they share the expensive power
    evaluations, so computing them together is cheaper than calling
    :func:`_sigmoid_transform` and differentiating separately.

    .. math:: s(r) = 1 - (1 + u)^{-b/a}, \qquad u = A (r/\sigma)^a

    and the derivative

    .. math:: s'(r) = \frac{b A}{\sigma} (r/\sigma)^{a-1}
              \frac{(1 + u)^{-b/a}}{1 + u}

    Both are defined to be 0 at :math:`r \le 0`.

    Parameters
    ----------
    distances : ndarray
        Pairwise distances
    sigma : float
        Switching distance where :math:`s(\sigma) = 0.5`
    a : float
        Short-range steepness parameter
    b : float
        Long-range steepness parameter

    Returns
    -------
    value : ndarray
        Sigmoid values in [0, 1], same shape as ``distances``
    derivative : ndarray
        :math:`\mathrm{d}s/\mathrm{d}r` at each distance, same shape as ``distances``
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


def _classical_mds(distances, n_components):
    """Classical MDS embedding used to initialize the optimization

    Recovers coordinates from the distance matrix by eigendecomposing the
    doubly-centered Gram matrix.

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        Symmetric pairwise distance matrix
    n_components : int
        Number of dimensions for the embedding

    Returns
    -------
    coordinates : ndarray of shape (n_samples, n_components)
        Embedding coordinates, ordered by decreasing eigenvalue
    """
    # double-centering the squared distances gives the Gram matrix
    gram_matrix = -0.5 * distances**2
    gram_matrix -= gram_matrix.mean(axis=0, keepdims=True)
    gram_matrix -= gram_matrix.mean(axis=1, keepdims=True)

    n = gram_matrix.shape[0]
    k = min(n_components, n)
    eigenvalues, eigenvectors = eigh(gram_matrix, subset_by_index=[n - k, n - 1])

    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    coordinates = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))

    # sign convention so the largest-magnitude entry of each column is positive
    for i in range(coordinates.shape[1]):
        col = coordinates[:, i]
        if col[np.argmax(np.abs(col))] < 0:
            coordinates[:, i] *= -1

    return coordinates


def _gaussian(x, amplitude, center, std_dev):
    """Gaussian function for curve fitting in distance distribution analysis"""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * std_dev**2))


def _analyze_distance_distribution(distances, n_bins=200, sample_weight=None):
    """Do analysis of the pairwise-distance distribution which is necessary for
    automatic sketch-map parameter estimation

    Builds a histogram of the distances and extracts three features that guide where the
    sigmoid should switch:

    - ``peak_distance``: the most common distance, the characteristic scale of
      the data
    - ``gaussian_range``: the short-range regime dominated by fluctuations,
      which should be compressed
    - ``uniform_cutoff``: the long-range regime where, in high dimensions,
      distances stop being informative

    Parameters
    ----------
    distances : ndarray
        Pairwise distance matrix (square symmetric) or flattened upper triangle
    n_bins : int, default=200
        Number of histogram bins for the analysis
    sample_weight : ndarray of shape (n_samples,), optional
        Per-sample weights, for instance the Voronoi weights of a set of landmarks
        (:func:`skmatter.sample_selection.voronoi_weights`). When given, the histogram
        weights each pair by :math:`w_i w_j`, matching how the sketch-map stress counts
        pairs, so the analysis describes the weighted dataset the fit actually sees.
        Requires ``distances`` to be the square matrix.

    Returns
    -------
    analysis : dict
        Summary of the distribution:

        - ``peak_distance``: the distance at the histogram peak
        - ``gaussian_std``: width of the Gaussian fitted to the short-range side of the
          peak, ``None`` when fewer than four bins fall below the peak
        - ``gaussian_range``: end of the short-range regime, about three standard
          deviations above the peak, ``None`` whenever ``gaussian_std`` is
        - ``uniform_cutoff``: start of the long-range regime, where the density has
          dropped to a tenth of the peak, ``None`` when fewer than four bins fall above
          the peak
        - ``bin_centers``, ``prob_density``: the histogram itself
        - ``max_distance``: the 99.9th percentile of the distances, which is the upper
          edge of the histogram
    """
    if distances.ndim == 2:
        upper = np.triu_indices_from(distances, k=1)
        d = distances[upper]
    else:
        d = distances

    pair_weights = None

    if sample_weight is not None:
        if distances.ndim != 2:
            raise ValueError(
                "sample_weight requires the square distance matrix, not a "
                "flattened upper triangle"
            )

        per_sample = np.asarray(sample_weight, dtype=float)
        if per_sample.shape[0] != distances.shape[0]:
            raise ValueError(
                f"sample_weight length {per_sample.shape[0]} != "
                f"n_samples {distances.shape[0]}"
            )

        pair_weights = np.outer(per_sample, per_sample)[upper]

    valid = np.isfinite(d) & (d >= 0)
    d = d[valid]

    if pair_weights is not None:
        pair_weights = pair_weights[valid]

    if d.size == 0:
        raise ValueError("Empty or invalid distances array")

    max_distance = np.percentile(d, 99.9)
    bin_edges = np.linspace(0, max_distance, n_bins + 1)
    prob_density, _ = np.histogram(
        d, bins=bin_edges, density=True, weights=pair_weights
    )
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

    # the left side of the peak characterises the short-range noise regime
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

            # the gaussian range reaches about 3 sigma from the peak
            analysis["gaussian_range"] = peak_distance + 3 * analysis["gaussian_std"]
        except (RuntimeError, ValueError):
            analysis["gaussian_std"] = peak_distance / 3.0
            analysis["gaussian_range"] = peak_distance * 2.0

    # the cutoff is where the right-side density falls to 10% of the peak
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

    # keep gaussian_range below uniform_cutoff
    if (
        analysis["gaussian_range"] is not None
        and analysis["uniform_cutoff"] is not None
        and analysis["gaussian_range"] >= analysis["uniform_cutoff"]
    ):
        analysis["gaussian_range"] = peak_distance + 0.2 * max_distance
        analysis["uniform_cutoff"] = peak_distance + 0.6 * max_distance

    return analysis


def _suggest_sigmoid_params(distances, n_components, n_bins=200, sample_weight=None):
    r"""Estimate sigmoid parameters from the pairwise-distance distribution

    The heuristics follow the published sketch-map guidelines:

    - ``sigma``: placed just before the peak (90% of peak distance) to ensure the bulk
      of distances fall in the sigmoid's sensitive region

    - ``a_high``, ``b_high`` (high-dimensional sigmoid): ``a_high`` sets the short-range
      exponent and ``b_high`` the long-range one. Since :math:`1 - s(r) \propto r^{-b}`
      for :math:`r \gg \sigma`, a small ``b_high`` is what keeps the long-distance tail
      long, so ``b_high`` must stay well below ``a_high``, and ``a_high`` is taken as
      twice ``b_high``. ``b_high`` grows with the ratio between the long- and
      short-range regimes, and falls back to 3 when the distribution is too irregular to
      characterize them

    - ``a_low``, ``b_low`` (low-dimensional sigmoid): set to the embedding
      dimensionality ``n_components``, the standard sketch-map choice. The rule
      :math:`a_{\text{low}} \cdot d \approx a_{\text{high}}
      \cdot D` motivates a small low-dimensional exponent but diverges for :math:`D \gg
      d`

    Parameters
    ----------
    distances : ndarray
        Pairwise distance matrix
    n_components : int
        Target embedding dimensionality
    n_bins : int, default=200
        Number of histogram bins for distance analysis
    sample_weight : ndarray of shape (n_samples,), optional
        Per-sample weights, for instance the Voronoi weights of a set of landmarks.
        Forwarded to :func:`_analyze_distance_distribution` so the histogram counts each
        pair with weight :math:`w_i w_j`, the same statistic the sketch-map stress uses

    Returns
    -------
    params : dict
        Suggested parameters with keys: ``sigma``, ``a_high``, ``b_high``, ``a_low``,
        ``b_low``
    analysis : dict
        Distance distribution analysis results from
        :func:`_analyze_distance_distribution`
    """
    analysis = _analyze_distance_distribution(
        distances, n_bins=n_bins, sample_weight=sample_weight
    )

    sigma = 0.9 * analysis["peak_distance"]

    if (
        analysis["gaussian_range"] is not None
        and analysis["uniform_cutoff"] is not None
    ):
        range_ratio = analysis["uniform_cutoff"] / max(
            analysis["gaussian_range"], 1e-10
        )
        b_high = np.clip(2.0 + np.log(range_ratio), 2.0, 6.0)
    else:
        b_high = 3.0

    a_high = 2.0 * b_high

    a_low = b_low = float(max(n_components, 1))

    params = {
        "sigma": sigma,
        "a_high": a_high,
        "b_high": b_high,
        "a_low": a_low,
        "b_low": b_low,
    }

    return params, analysis
