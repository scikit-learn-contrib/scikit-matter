"""
Sketch-Map: Nonlinear dimensionality reduction

Sketch-Map is a nonlinear dimensionality reduction technique. Unlike linear methods
(PCA) or neighborhood-based methods (t-SNE, UMAP), Sketch-Map uses sigmoid functions to
selectively focus on "intermediate-range" pairwise distances while filtering out noise
from both extremes.

The key insight is that in high-dimensional data very short distances often represent
thermal fluctuations or noise that should not influence the global structure, very long
distances are unreliable due to the "curse of dimensionality" where all points appear
roughly equidistant in high dimensions. Intermediate distances contain the meaningful
structural information we want to preserve in the low-dimensional embedding.

The sigmoid transformation compresses both extremes toward 0 and 1, effectively creating
a "sketch" of the distance relationships that emphasizes the informative middle range.

References
----------
See Ceriotti et al. [Ceriotti2011]_, which introduces Sketch-Map, and the follow-up
[Ceriotti2013]_.
"""

import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import basinhopping, minimize, curve_fit
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


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
    with np.errstate(divide="ignore", invalid="ignore"):
        # Normalize distances by the switching distance
        r_normalized = distances / sigma

        # Sigmoid coefficient ensuring s(sigma) = 0.5
        A = 2 ** (a / b) - 1.0

        # Compute sigmoid transformation
        term = A * (r_normalized**a)
        transformed = 1.0 - (1.0 + term) ** (-b / a)

        # Handle edge case: zero distance should map to zero
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
    with np.errstate(divide="ignore", invalid="ignore"):
        A = 2 ** (a / b) - 1.0

        # Inverse formula derived by solving s(r) = y for r
        one_minus_y = np.maximum(1.0 - y, 1e-12)
        inner = np.power(one_minus_y, -a / b) - 1.0
        distances = sigma * np.power(np.maximum(inner / A, 0.0), 1.0 / a)

        # Handle boundary cases
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
    A = 2 ** (a / b) - 1.0
    derivative = np.zeros_like(distances, dtype=float)

    # Only compute for positive distances (derivative at r=0 is 0)
    positive_mask = distances > 0.0

    if np.any(positive_mask):
        r = distances[positive_mask]
        u = A * (r / sigma) ** a
        prefactor = b * A * (r ** (a - 1.0)) / (sigma**a)
        derivative[positive_mask] = prefactor * (1.0 + u) ** (-b / a - 1.0)

    return derivative


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
    # Double-centering to obtain the Gram matrix
    gram_matrix = -0.5 * (distances**2)
    gram_matrix -= gram_matrix.mean(axis=0, keepdims=True)  # Center columns
    gram_matrix -= gram_matrix.mean(axis=1, keepdims=True)  # Center rows

    # Eigendecomposition (eigh returns sorted in ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

    # Select top n_components (largest eigenvalues)
    top_indices = np.argsort(eigenvalues)[::-1][:n_components]
    top_eigenvalues = eigenvalues[top_indices]
    top_eigenvectors = eigenvectors[:, top_indices]

    # Compute coordinates: X = V @ sqrt(lambda)
    coordinates = top_eigenvectors * np.sqrt(np.maximum(top_eigenvalues, 0.0))

    # Apply consistent sign convention
    for i in range(n_components):
        col = coordinates[:, i]
        if col[np.argmax(np.abs(col))] < 0:
            coordinates[:, i] *= -1

    return coordinates


def _gaussian(x, amplitude, center, std_dev):
    """Gaussian function for curve fitting in distance distribution analysis."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * std_dev**2))


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
    # Extract upper triangle if given a square matrix
    if distances.ndim == 2:
        d = distances[np.triu_indices_from(distances, k=1)]

    else:
        d = distances.copy()

    # Filter invalid values
    d = d[np.isfinite(d) & (d >= 0)]
    if d.size == 0:
        raise ValueError("Empty or invalid distances array")

    # Build histogram
    max_distance = np.percentile(d, 99.9)
    bin_edges = np.linspace(0, max_distance, n_bins + 1)
    prob_density, _ = np.histogram(d, bins=bin_edges, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Locate the peak of the distribution
    peak_idx = np.argmax(prob_density)
    peak_distance = bin_centers[peak_idx]

    # Initialize results
    analysis = {
        "peak_distance": peak_distance,
        "gaussian_std": None,
        "gaussian_range": None,
        "uniform_cutoff": None,
        "bin_centers": bin_centers,
        "prob_density": prob_density,
        "max_distance": max_distance,
    }

    # Estimate Gaussian fluctuation range (left side of peak)
    # Fit a Gaussian to the left half of the histogram to characterize the short-range
    # noise regime

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
            # Fallback: use heuristic estimate
            analysis["gaussian_std"] = peak_distance / 3.0
            analysis["gaussian_range"] = peak_distance * 2.0

    #  Estimate high-dimensional cutoff (right side of peak)
    # Find where density drops to 10% of peak value

    right_mask = bin_centers > peak_distance
    right_density = prob_density[right_mask]

    if len(right_density) > 3:
        peak_density = prob_density[peak_idx]
        threshold = 0.1 * peak_density
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
    ):
        if analysis["gaussian_range"] >= analysis["uniform_cutoff"]:
            # fallback to default spacing
            analysis["gaussian_range"] = peak_distance + 0.2 * max_distance
            analysis["uniform_cutoff"] = peak_distance + 0.6 * max_distance

    return analysis


def suggest_sigmoid_params(distances, n_components, n_features, n_bins=200):
    r"""Suggest sigmoid parameters based on distance distribution analysis.

    This function implements heuristics from the Sketch-Map guidelines for automatic
    parameter selection.

    Parameter selection strategy:

    - ``sigma``: placed just before the peak (90% of peak distance) to ensure the bulk
      of distances fall in the sigmoid's sensitive region.

    - ``a_high``, ``b_high`` (high-D sigmoid): control how aggressively short and long
      distances are compressed in high-dimensional space.

    - ``a_low``, ``b_low`` (low-D sigmoid): typically smaller values to account for the
      "volume equalization" between high-D and low-D spaces. A rule of thumb is
      :math:`a_{\text{low}} \cdot d \approx a_{\text{high}} \cdot D` where :math:`d` is
      the target dimension and :math:`D` is the original dimension.

    Parameters
    ----------
    distances : ndarray
        Pairwise distance matrix.
    n_components : int
        Target embedding dimensionality.
    n_features : int
        Original feature dimensionality (used for dimension ratio heuristics).
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

    # Estimate sigma
    sigma = 0.9 * analysis["peak_distance"]

    # Estimate high-dimensional sigmoid parameters (a_high, b_high)
    if (
        analysis["gaussian_range"] is not None
        and analysis["uniform_cutoff"] is not None
    ):
        range_ratio = analysis["uniform_cutoff"] / max(
            analysis["gaussian_range"], 1e-10
        )
        a_high = np.clip(2.0 + np.log(range_ratio), 2.0, 6.0)
        b_high = np.clip(a_high * 2, 4.0, 12.0)

    else:
        a_high = 2.0
        b_high = 6.0

    # Estimate low-dimensional sigmoid parameters (a_low, b_low)
    if n_features > 0:
        a_low = np.clip(a_high * n_components / n_features, 1.0, 2.0)

    else:
        a_low = 2.0

    b_low = np.clip(a_low, 1.0, 2.0)

    params = {
        "sigma": sigma,
        "a_high": a_high,
        "b_high": b_high,
        "a_low": a_low,
        "b_low": b_low,
    }

    return params, analysis


# Main class


class SketchMap(TransformerMixin, BaseEstimator):
    r"""
    Sketch-Map is a nonlinear dimensionality reduction algorithm. Unlike methods that
    try to preserve all pairwise distances (MDS) or only local neighborhoods (t-SNE),
    Sketch-Map selectively focuses on intermediate-range distances using sigmoid
    transformations: short distances (:math:`r \ll \sigma`) compressed toward 0 (these
    often represent thermal noise or fluctuations that should not dominate the
    embedding), long distances (:math:`r \gg \sigma`) compressed toward 1 (in high
    dimensions, very long distances become unreliable due to the "curse of
    dimensionality"), intermediate distances (:math:`r \approx \sigma`) are preserved
    (these contain the meaningful structural information).

    The sigmoid function has the form:

    .. math::

        s(r) = 1 - \left(1 + A \cdot \left(\frac{r}{\sigma}\right)^a\right)^{-b/a}

    where:

      - :math:`\sigma` is the "switching distance" where :math:`s(\sigma) = 0.5`
      - :math:`a` controls the short-range exponent (how fast :math:`s \to 0`)
      - :math:`b` controls the long-range exponent (how fast :math:`s \to 1`)
      - :math:`A = 2^{a/b} - 1` ensures :math:`s(\sigma) = 0.5`

    Separate sigmoids are applied to high-D distances (``a_high``, ``b_high``) and
    low-D distances (``a_low``, ``b_low``), allowing different compression behaviors.

    The fitting process proceeds in stages:

      1. Initialization: classical MDS provides starting coordinates
      2. MDS refinement (optional): optimize raw distance stress
      3. Pre-optimization: initial sigmoid-transformed optimization
      4. Main optimization: full optimization
      5. Global optimization (optional): basin-hopping for better minima

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in the target embedding space.

    params : dict or None, default=None
        Sigmoid parameters dictionary. Keys are:

          - ``sigma`` : float - Switching distance where :math:`s(\sigma) = 0.5`
          - ``a_high`` : float - High-D sigmoid short-range steepness
          - ``b_high`` : float - High-D sigmoid long-range steepness
          - ``a_low`` : float - Low-D sigmoid short-range steepness
          - ``b_low`` : float - Low-D sigmoid long-range steepness

        If None, all parameters are estimated automatically from the data.
        Partial dictionaries are allowed; missing keys use auto-estimated values.

    mds_opt_steps : int, default=100
        Number of MDS optimization steps (raw distances, no sigmoid) before
        applying the sigmoid transformation. Set to 0 to skip this stage.
        This helps refine the classical MDS initialization.

    optimizer : str, default="L-BFGS-B"
        Optimization algorithm. Options: ``"L-BFGS-B"`` or ``"CG"``.

    preopt_steps : int, default=100
        Number of pre-optimization steps with sigmoid transformation.
        Uses the same settings as main optimization but fewer iterations.

    max_iter : int, default=1000
        Maximum iterations for the main optimization stage.

    global_opt_steps : int or None, default=None
        Number of basin-hopping iterations for global optimization.
        Basin-hopping can escape local minima but is computationally expensive.
        Set to None to disable.

    mixing_ratio : float, default=0.0
        Balance between raw distance stress and transformed distance stress:

          - 0.0: Pure sigmoid-transformed stress
          - 1.0: Pure raw distance stress
          - Values in between: linear combination

    center : bool, default=True
        Whether to center the input data (subtract mean) before computing
        pairwise distances.

    init : array-like of shape (n_samples, n_components) or None, default=None
        Initial embedding coordinates. If None, classical MDS is used.

    random_state : int or None, default=None
        Random seed for reproducibility (affects global optimization).

    verbose : bool, default=False
        If True, print progress information during fitting.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        The fitted low-dimensional embedding coordinates.

    stress_ : float
        Final stress value (lower is better).

    params_ : dict
        The sigmoid parameters actually used (combination of user-provided
        and auto-estimated values).

    suggested_params_ : dict
        Auto-suggested sigmoid parameters (``sigma``, ``a_high``, ``b_high``,
        ``a_low``, ``b_low``) based on distance distribution analysis.
        Available after fitting, useful for understanding the data.

    distance_analysis_ : dict
        Results from the distance distribution analysis, including peak
        distance, Gaussian range estimates, and histogram data.

    n_samples_ : int
        Number of samples in the training data.

    n_features_ : int
        Number of features in the training data.

    See Also
    --------
    sklearn.manifold.MDS : Multidimensional scaling (linear).
    sklearn.manifold.TSNE : t-SNE for visualization.

    Examples
    --------
    Basic usage with automatic parameter estimation:

      >>> from skmatter.decomposition import SketchMap
      >>> import numpy as np
      >>> X = np.random.randn(100, 50)
      >>> sm = SketchMap(n_components=2, random_state=42)
      >>> embedding = sm.fit_transform(X)
      >>> print(embedding.shape)
      (100, 2)

    Using specific sigmoid parameters:

      >>> params = {"sigma": 7.0, "a_high": 4.0, "b_high": 2.0,
      ...           "a_low": 2.0, "b_low": 2.0}
      >>> sm = SketchMap(n_components=2, params=params)
      >>> embedding = sm.fit_transform(X)
      >>> print(embedding.shape)
      (100, 2)
    """

    def __init__(
        self,
        n_components=2,
        params=None,
        mds_opt_steps=100,
        optimizer="L-BFGS-B",
        preopt_steps=100,
        max_iter=1000,
        global_opt_steps=None,
        mixing_ratio=0.0,
        center=True,
        init=None,
        random_state=None,
        verbose=False,
    ):
        self.n_components = n_components
        self.params = params
        self.mds_opt_steps = mds_opt_steps
        self.optimizer = optimizer
        self.preopt_steps = preopt_steps
        self.max_iter = max_iter
        self.global_opt_steps = global_opt_steps
        self.mixing_ratio = mixing_ratio
        self.center = center
        self.init = init
        self.random_state = random_state
        self.verbose = verbose

    # Stress computation

    def _compute_stress(
        self,
        embedding,
        hd_distances,
        hd_transformed,
        weights,
        total_weight,
        mixing_ratio,
        use_transform=True,
    ):
        """Compute the Sketch-Map stress function.

        The stress measures the discrepancy between (transformed) high-D distances and
        (transformed) low-D distances. Lower stress indicates a better embedding.

        The stress formula is::

            stress = sum_ij w_ij * [(1-m)*(s_hd(D_ij) - s_ld(d_ij))^2
                     + m*(D_ij - d_ij)^2]

        where:

        - D_ij is the high-D distance between points i and j
        - d_ij is the low-D distance
        - s_hd and s_ld are the high-D and low-D sigmoid functions
        - w_ij are pairwise weights
        - m is the mixing_ratio

        Parameters
        ----------
        embedding : ndarray, shape (n_samples * n_components,) or (n, d)
            Low-dimensional coordinates (flat or 2D).
        hd_distances : ndarray of shape (n_samples, n_samples)
            High-dimensional pairwise distances.
        hd_transformed : ndarray of shape (n_samples, n_samples)
            Sigmoid-transformed high-dimensional distances.
        weights : ndarray of shape (n_samples, n_samples)
            Pairwise weight matrix.
        total_weight : float
            Sum of upper-triangle weights (for normalization).
        mixing_ratio : float
            Balance between transformed (0) and raw (1) stress.
        use_transform : bool, default=True
            Whether to apply sigmoid to low-D distances.

        Returns
        -------
        stress : float
            Normalized stress value.
        """
        # Reshape flat embedding to matrix form
        if embedding.ndim == 1:
            n_samples = hd_distances.shape[0]
            embedding = embedding.reshape((n_samples, self.n_components))

        # Compute low-dimensional pairwise distances
        ld_distances = cdist(embedding, embedding, metric="euclidean")

        # Apply sigmoid transformation to low-D distances
        if use_transform:
            # Use precomputed coefficient for speed
            r_normalized = ld_distances / self.params_["sigma"]
            term = self._A_low_ * np.power(r_normalized, self.params_["a_low"])
            ld_transformed = 1.0 - np.power(
                1.0 + term, -self.params_["b_low"] / self.params_["a_low"]
            )
            ld_transformed[ld_distances <= 0.0] = 0.0
        else:
            ld_transformed = ld_distances

        # Compute stress only on upper triangle (use precomputed indices)
        triu_idx = self._triu_idx_

        diff_transformed = (hd_transformed[triu_idx] - ld_transformed[triu_idx]) ** 2
        diff_direct = (hd_distances[triu_idx] - ld_distances[triu_idx]) ** 2
        weights_triu = weights[triu_idx]

        # Combine transformed and direct stress according to mixing_ratio
        stress = np.sum(
            weights_triu
            * ((1.0 - mixing_ratio) * diff_transformed + mixing_ratio * diff_direct)
        )

        return stress / total_weight

    # Gradient computation

    def _compute_gradient(
        self,
        embedding,
        hd_distances,
        hd_transformed,
        weights,
        total_weight,
        mixing_ratio,
        use_transform=True,
    ):
        """Compute the gradient of the stress function.

        The gradient is used by optimization algorithms (L-BFGS-B, CG) to
        find the embedding that minimizes stress.

        Parameters
        ----------
        embedding : ndarray
            Low-dimensional coordinates (flat or 2D).
        hd_distances : ndarray
            High-dimensional pairwise distances.
        hd_transformed : ndarray
            Transformed high-dimensional distances.
        weights : ndarray
            Pairwise weight matrix.
        total_weight : float
            Total weight for normalization.
        mixing_ratio : float
            Balance between transformed and raw stress.
        use_transform : bool, default=True
            Whether sigmoid transformation is applied.

        Returns
        -------
        gradient : ndarray of shape (n_samples * n_components,)
            Flattened gradient vector.
        """
        # Reshape flat embedding to matrix form
        if embedding.ndim == 1:
            n_samples = hd_distances.shape[0]
            embedding = embedding.reshape((n_samples, self.n_components))

        # Compute low-dimensional pairwise distances using optimized cdist
        ld_distances = cdist(embedding, embedding, metric="euclidean")

        # Compute sigmoid transformation and its derivative
        if use_transform:
            sigma = self.params_["sigma"]
            a_low = self.params_["a_low"]
            b_low = self.params_["b_low"]
            A = self._A_low_

            # Optimized sigmoid transform
            r_normalized = ld_distances / sigma
            r_pow_a = np.power(r_normalized, a_low)
            term = A * r_pow_a
            base = 1.0 + term
            ld_transformed = 1.0 - np.power(base, -b_low / a_low)
            ld_transformed[ld_distances <= 0.0] = 0.0

            # Optimized derivative:
            # ds/dr = b * A * r^(a-1) / sigma^a * (1+term)^(-b/a-1)
            ld_derivative = np.zeros_like(ld_distances)
            positive_mask = ld_distances > 0.0
            if np.any(positive_mask):
                r = ld_distances[positive_mask]
                u = A * np.power(r / sigma, a_low)
                prefactor = (
                    b_low * A * np.power(r, a_low - 1.0) / np.power(sigma, a_low)
                )
                ld_derivative[positive_mask] = prefactor * np.power(
                    1.0 + u, -b_low / a_low - 1.0
                )
        else:
            ld_transformed = ld_distances
            ld_derivative = np.ones_like(ld_distances)

        # Compute gradient coefficients based on mixing_ratio
        # The gradient comes from differentiating the stress w.r.t. coordinates

        if mixing_ratio == 0.0:
            # Pure transformed stress
            coefficients = weights * (hd_transformed - ld_transformed) * ld_derivative

        else:
            # Mixed stress: includes both transformed and raw distance terms
            eps = 1e-100
            inv_distance = 1.0 / np.maximum(ld_distances, eps)
            coefficients = (
                weights
                * (
                    (1.0 - mixing_ratio)
                    * (hd_transformed - ld_transformed)
                    * ld_derivative
                    + mixing_ratio * (hd_distances - ld_distances)
                )
                * inv_distance
            )

        np.fill_diagonal(coefficients, 0.0)

        # Compute gradient using efficient matrix operations
        # d(stress)/d(x_i) = -2 * sum_j c_ij * (x_i - x_j)

        row_sums = coefficients.sum(axis=1)
        gradient = (row_sums[:, None] * embedding) - (coefficients @ embedding)
        gradient = -2.0 * gradient / total_weight

        return gradient.ravel()

    # Optimization methods

    def _optimize(
        self,
        initial_embedding,
        hd_distances,
        hd_transformed,
        weights,
        total_weight,
        mixing_ratio,
        n_steps,
        use_transform=True,
    ):
        """Run gradient-based optimization to minimize stress.

        Uses scipy.optimize.minimize with the configured optimizer
        (L-BFGS-B or CG).

        Parameters
        ----------
        initial_embedding : ndarray of shape (n_samples, n_components)
            Starting coordinates.
        hd_distances : ndarray
            High-dimensional pairwise distances.
        hd_transformed : ndarray
            Transformed high-dimensional distances.
        weights : ndarray
            Pairwise weight matrix.
        total_weight : float
            Total weight for normalization.
        mixing_ratio : float
            Balance between transformed and raw stress.
        n_steps : int
            Maximum number of optimization iterations.
        use_transform : bool, default=True
            Whether to apply sigmoid transformation.

        Returns
        -------
        optimized_embedding : ndarray of shape (n_samples, n_components)
            Optimized coordinates.
        final_stress : float
            Final stress value.
        """
        n_samples = initial_embedding.shape[0]
        x0 = initial_embedding.ravel()

        # Define objective and gradient functions

        def objective(x):
            return self._compute_stress(
                x,
                hd_distances,
                hd_transformed,
                weights,
                total_weight,
                mixing_ratio,
                use_transform,
            )

        def gradient(x):
            return self._compute_gradient(
                x,
                hd_distances,
                hd_transformed,
                weights,
                total_weight,
                mixing_ratio,
                use_transform,
            )

        # Run optimization

        result = minimize(
            objective,
            x0,
            method=self.optimizer,
            jac=gradient,
            options={"maxiter": n_steps, "gtol": 1e-8},
        )

        if self.verbose:
            print(f"  Optimization finished: stress = {result.fun:.6f}")

        optimized_embedding = result.x.reshape((n_samples, self.n_components))
        return optimized_embedding, result.fun

    def _global_optimize(
        self,
        embedding,
        hd_distances,
        hd_transformed,
        weights,
        total_weight,
        n_iterations,
    ):
        """Run global optimization using basin-hopping which combines local optimization
        with random perturbations to escape local minima. This is computationally
        expensive but can find better solutions for complex landscapes.

        Parameters
        ----------
        embedding : ndarray of shape (n_samples, n_components)
            Current embedding (starting point).
        hd_distances : ndarray
            High-dimensional pairwise distances.
        hd_transformed : ndarray
            Transformed distances.
        weights : ndarray
            Pairwise weight matrix.
        total_weight : float
            Total weight for normalization.
        n_iterations : int
            Number of basin-hopping iterations.

        Returns
        -------
        optimized_embedding : ndarray of shape (n_samples, n_components)
            Globally optimized coordinates.
        final_stress : float
            Final stress value.
        """
        n_samples = embedding.shape[0]
        x0 = embedding.ravel()

        def objective(x):
            return self._compute_stress(
                x,
                hd_distances,
                hd_transformed,
                weights,
                total_weight,
                self.mixing_ratio,
                use_transform=True,
            )

        def gradient(x):
            return self._compute_gradient(
                x,
                hd_distances,
                hd_transformed,
                weights,
                total_weight,
                self.mixing_ratio,
                use_transform=True,
            )

        minimizer_kwargs = {
            "method": self.optimizer,
            "jac": gradient,
            "options": {"maxiter": 100},
        }

        if self.verbose:
            print(f"\n=== Global optimization (basin hopping, {n_iterations} iter) ===")

        # Custom random displacement for basin-hopping

        rng = np.random.default_rng(self.random_state)

        class RandomDisplacement:
            """Random step generator for basin-hopping."""

            def __init__(self, stepsize, rng):
                self.stepsize = stepsize
                self.rng = rng

            def __call__(self, x):
                return x + self.rng.uniform(-self.stepsize, self.stepsize, x.shape)

        # Step size proportional to embedding scale
        scale = np.std(embedding)
        take_step = RandomDisplacement(stepsize=scale * 0.5, rng=rng)

        result = basinhopping(
            objective,
            x0,
            niter=n_iterations,
            minimizer_kwargs=minimizer_kwargs,
            take_step=take_step,
            seed=int(rng.integers(0, 2**31)) if self.random_state else None,
        )

        if self.verbose:
            print(f"  Basin hopping finished: stress = {result.fun:.6f}")

        return result.x.reshape((n_samples, self.n_components)), result.fun

    # Public API: fit, transform, fit_transform

    def fit(self, X, y=None, sample_weights=None):
        """Fit the Sketch-Map model to the training data.

        This method computes the low-dimensional embedding by 1) computing pairwise
        distances in the input space 2) estimating sigmoid parameters (if not provided)
        3) transforming distances using the sigmoid function 4) optimizing embedding
        coordinates to minimize stress.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Each row is a sample, each column is a feature.
        y : Ignored
            Not used, present for scikit-learn API compatibility.
        sample_weights : array-like of shape (n_samples,), optional
            Per-sample weights. Samples with higher weights have more
            influence on the embedding. Default is uniform weights.

        Returns
        -------
        self : SketchMap
            Returns the fitted instance.
        """
        X = validate_data(self, X, reset=True, dtype=np.float64)
        self.X_ = X.copy()

        if sparse.issparse(X):
            raise ValueError("Sparse input is not supported")
        if np.any(~np.isfinite(X)):
            raise ValueError("Input contains NaN or infinity")

        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(
                f"Found array with {n_samples} sample(s) while minimum of 2 required."
            )

        self.n_samples_ = n_samples
        self.n_features_ = n_features

        if self.verbose:
            print(f"Fitting Sketch-Map: {n_samples} samples, {n_features} features")

        # Compute pairwise distances

        X_processed = X.copy()
        if self.center:
            X_processed = X_processed - X_processed.mean(axis=0, keepdims=True)

            if self.verbose:
                print("Data centered")

        if self.verbose:
            print("Computing pairwise distances...")

        hd_distances = cdist(X_processed, X_processed, metric="euclidean")

        # Determine sigmoid parameters

        suggested, analysis = suggest_sigmoid_params(
            hd_distances, self.n_components, n_features
        )
        self.suggested_params_ = suggested
        self.distance_analysis_ = analysis

        if self.params is None:
            # Use all suggested parameters
            self.params_ = suggested.copy()

            if self.verbose:
                print("Using auto-estimated sigmoid parameters:")

        else:
            # Start with suggested, override with user-provided values
            self.params_ = suggested.copy()

            for key in ["sigma", "a_high", "b_high", "a_low", "b_low"]:
                if key in self.params and self.params[key] is not None:
                    self.params_[key] = self.params[key]

            if self.verbose:
                print("Using sigmoid parameters (user + auto-estimated):")

        if self.verbose:
            print(f"sigma = {self.params_['sigma']:.4f}")
            print(
                f"a_high = {self.params_['a_high']:.2f}, "
                f"b_high = {self.params_['b_high']:.2f}"
            )
            print(
                f"a_low = {self.params_['a_low']:.2f}, "
                f"b_low = {self.params_['b_low']:.2f}"
            )
            print(f"(peak distance = {analysis['peak_distance']:.4f})")

        # Apply sigmoid transformation to high-D distances

        hd_transformed = sigmoid_transform(
            hd_distances,
            self.params_["sigma"],
            self.params_["a_high"],
            self.params_["b_high"],
        )

        # Precompute indices for upper triangle
        self._triu_idx_ = np.triu_indices(n_samples, k=1)

        # Precompute sigmoid coefficient for low-D transform
        self._A_low_ = 2 ** (self.params_["a_low"] / self.params_["b_low"]) - 1.0

        # Setup weight matrix

        if sample_weights is not None:
            w = np.asarray(sample_weights)

            if w.shape[0] != n_samples:
                raise ValueError(
                    f"sample_weights length {w.shape[0]} != n_samples {n_samples}"
                )
            weights = np.outer(w, w)
            total_weight = np.sum(np.triu(weights, k=1))

            if self.verbose:
                print(f"Using sample weights (total_weight = {total_weight:.4f})")

        else:
            weights = np.ones_like(hd_distances)
            total_weight = n_samples * (n_samples - 1) / 2.0

            if self.verbose:
                print(f"Using uniform weights (total_weight = {total_weight:.0f})")

        # Initialize embedding

        if self.init is not None:
            embedding = np.asarray(self.init).copy()

            if self.verbose:
                print(f"Using provided initialization, shape = {embedding.shape}")

        else:
            if self.verbose:
                print("Initializing with classical MDS...")
            embedding = classical_mds(hd_distances, self.n_components)

        # MDS pre-optimization

        if self.mds_opt_steps > 0 and self.init is None:
            if self.verbose:
                print(
                    f"\n=== Stage 0: MDS optimization ({self.mds_opt_steps} steps) ==="
                )

            embedding, mds_stress = self._optimize(
                embedding,
                hd_distances,
                hd_distances,  # target = raw distances (no transform)
                weights,
                total_weight,
                mixing_ratio=1.0,  # pure raw distance stress
                n_steps=self.mds_opt_steps,
                use_transform=False,
            )

            if self.verbose:
                print(f"MDS stress: {mds_stress:.6f}")

        # Pre-optimization with sigmoid transformation

        stress = 0.0
        if self.preopt_steps > 0:
            if self.verbose:
                print(
                    f"\n=== Stage 1: Pre-optimization ({self.preopt_steps} steps) ==="
                )

            embedding, stress = self._optimize(
                embedding,
                hd_distances,
                hd_transformed,
                weights,
                total_weight,
                self.mixing_ratio,
                self.preopt_steps,
                use_transform=True,
            )

            if self.verbose:
                print(f"Pre-optimization stress: {stress:.6f}")

        # Main optimization

        if self.max_iter > 0:
            if self.verbose:
                print(f"\n=== Stage 2: Main optimization ({self.max_iter} steps) ===")

            embedding, stress = self._optimize(
                embedding,
                hd_distances,
                hd_transformed,
                weights,
                total_weight,
                self.mixing_ratio,
                self.max_iter,
                use_transform=True,
            )

            if self.verbose:
                print(f"Final stress: {stress:.6f}")

        # Global optimization

        if self.global_opt_steps is not None:
            if not isinstance(self.global_opt_steps, int) or self.global_opt_steps < 1:
                raise ValueError(
                    "global_opt_steps must be a positive int, got "
                    f"{self.global_opt_steps}"
                )

            embedding, stress = self._global_optimize(
                embedding,
                hd_distances,
                hd_transformed,
                weights,
                total_weight,
                self.global_opt_steps,
            )

        # Store results
        self.embedding_ = embedding
        self.stress_ = stress

        # Track total iterations for sklearn compatibility
        n_iter = self.mds_opt_steps + self.preopt_steps + self.max_iter

        if self.global_opt_steps is not None:
            n_iter += self.global_opt_steps

        self.n_iter_ = n_iter

        if self.verbose:
            print("\nSketch-Map fitting complete!")

        return self

    def fit_transform(self, X, y=None, sample_weights=None):
        """Fit the model and return the embedding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for scikit-learn API compatibility.
        sample_weights : array-like of shape (n_samples,), optional
            Per-sample weights.

        Returns
        -------
        embedding : ndarray of shape (n_samples, n_components)
            Low-dimensional embedding coordinates.
        """
        self.fit(X, y, sample_weights=sample_weights)
        return self.embedding_

    def transform(self, X):
        """Project data to the embedding space.

        This method only supports in-sample transformation (data points that were used
        during fitting). Out-of-sample projection is not currently implemented.

        For new data points, consider using landmark-based approaches.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform. Must be rows from the original training data.

        Returns
        -------
        embedding : ndarray of shape (n_samples, n_components)
            Embedded coordinates for the input samples.
        """
        check_is_fitted(self, ["embedding_", "X_"])
        X = validate_data(self, X, reset=False)

        indices = []
        for row in X:
            matches = np.all(np.isclose(self.X_, row, rtol=1e-8, atol=1e-12), axis=1)
            if not np.any(matches):
                warnings.warn(
                    "SketchMap.transform only supports in-sample rows. "
                    "Out-of-sample projection is not yet implemented.",
                    UserWarning,
                )
            indices.append(int(np.argmax(matches)))

        return self.embedding_[indices]

    def predict(self, X):
        """Alias for :meth:`transform` (for API compatibility)."""
        return self.transform(X)

    def score(self, X, y=None):
        """Return negative stress as a score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. Used only for feature validation.
        y : array-like
            Ignored, present for scikit-learn API compatibility.

        Returns
        -------
        score : float
            Negative of the final stress value.
        """
        check_is_fitted(self, ["stress_"])
        validate_data(self, X, reset=False)
        return -self.stress_
