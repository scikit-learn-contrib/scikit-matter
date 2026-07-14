"""
Sketch-Map: Nonlinear dimensionality reduction

References
----------
See Ceriotti et al. [Ceriotti2011]_, which introduces Sketch-Map, and the follow-up
[Ceriotti2013]_.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ._sketchmap_utils import (
    _maybe_tqdm,
    _sigmoid_fdf,
    classical_mds,
    sigmoid_transform,
    suggest_sigmoid_params,
)

# Resolution of the pointwise global search, as points per axis (the grids are
# square, so 21 means a 21x21 grid). The single-point stress is evaluated exactly
# on the coarse grid, then a bicubic interpolant of it is scanned on the much
# finer grid to locate the minimum cheaply. Values match the C++ ``-grid gw,g1,g2``
# defaults; they are fixed because the user never needs to tune them.
_GRID_COARSE_POINTS = 21
_GRID_FINE_POINTS = 201

# Optimizer iterations used to relax the whole embedding after each global
# optimization cycle's grid pass. Fixed because the number of cycles
# (``global_opt_steps``) is the dial users actually reach for.
_GLOBAL_OPT_REFINE_STEPS = 100

# transform() projects new points in row-blocks so the (block x n_landmarks)
# distance matrix stays bounded (2e7 elements ~= 160 MB at float64); projecting
# 856k points against 3k landmarks in one shot would need ~20 GB.
_TRANSFORM_BATCH_ELEMENTS = 20_000_000


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

    where :math:`\sigma` is the "switching distance" where :math:`s(\sigma) = 0.5`
    :math:`a` controls the short-range exponent (how fast :math:`s \to 0`) :math:`b`
    controls the long-range exponent (how fast :math:`s \to 1`) :math:`A = 2^{a/b} - 1`
    ensures :math:`s(\sigma) = 0.5`.

    Separate sigmoids are applied to high-D distances (``a_high``, ``b_high``) and low-D
    distances (``a_low``, ``b_low``) giving slightly different compression behaviors.

    All five parameters are estimated automatically from the data when left at ``None``.
    The values actually used are stored in ``params_`` after fitting. If one needs to
    tune them by hand:

      - ``sigma`` is the most important parameter: distances below it are treated as
        "close", distances above as "far". Place it just below the peak of the
        pairwise distance histogram.
      - ``a_high``, ``b_high`` shape the high-D sigmoid: larger ``a_high``
        compresses short-range (noise) distances more aggressively, smaller
        ``b_high`` saturates long distances more softly.
      - ``a_low``, ``b_low`` shape the low-D sigmoid and are usually smaller
        than their high-D counterparts to compensate for the volume difference
        between the spaces.

    These correspond one-to-one to the ``-fun-hd sigma,a_high,b_high`` and
    ``-fun-ld sigma,a_low,b_low`` flags of the reference C++ implementation
    from `sketchmap.org <https://sketchmap.org>`_.

    The fit proceeds through:

      1. classical MDS for the initial coordinates,
      2. refinement of those coordinates against the raw distances,
      3. main optimization of the sigmoid-transformed stress,
      4. grid-based global optimization (2D only) with an annealed mixing schedule.

    The optimization is deterministic.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in the target embedding space.

    sigma : float or None, default=None
        Switching distance where :math:`s(\sigma) = 0.5`, applied to both
        high-D and low-D distances.
        If None, it is estimated automatically as 90% of the peak of the
        pairwise distance distribution.

    a_high, b_high : float or None, default=None
        Short- and long-range steepness exponents of the high-D sigmoid.
        If None, both are estimated from the distance distribution.

    a_low, b_low : float or None, default=None
        Short- and long-range steepness exponents of the low-D sigmoid.
        If None, both are estimated from the input dimensionality.

    mds_opt_steps : int, default=100
        Number of MDS optimization steps (raw distances, no sigmoid) before
        applying the sigmoid transformation. Set to 0 to skip this stage.
        This helps refine the classical MDS initialization.

    optimizer : str, default="L-BFGS-B"
        Optimization algorithm. Options: ``"L-BFGS-B"`` or ``"CG"``.

    max_iter : int, default=1000
        Maximum iterations for the main optimization stage.

    global_opt_steps : int, "auto" or None, default="auto"
        Number of annealing levels for the graduated global optimization, the
        main quality dial. The sigmoid stress is non-convex, so after the main
        optimization the embedding is re-relaxed with L-BFGS while the mixing
        ratio is annealed from the smooth MDS stress toward the pure sigmoid
        stress (see ``mixing_schedule``); this escapes local minima without the
        per-point "jumps" of a grid search, so it never creates spurious
        outliers. More levels give a smoother (slower) anneal. ``"auto"``
        (default) uses 5 levels. Set to 0 or None to disable.

    mixing_ratio : float, default=0.0
        Balance between raw distance stress and transformed distance stress:

          - 0.0: Pure sigmoid-transformed stress
          - 1.0: Pure raw distance stress
          - Values in between: linear combination

        Used as the constant mixing ratio for the main optimization stage and
        as the final target of the annealing schedule.

    mixing_schedule : sequence of float, "auto" or None, default="auto"
        Mixing ratios to anneal through during global optimization, highest
        (MDS-like) first. At mixing 1.0 the stress is the raw-distance MDS
        stress (smooth, ~convex); at 0.0 it is the pure sigmoid stress. Each
        level is relaxed with L-BFGS, warm-started from the previous one, so
        the solution tracks continuously into the sigmoid basin. ``"auto"``
        (default) anneals geometrically from 1.0 down to exactly 0 over
        ``global_opt_steps`` levels. A custom sequence is used verbatim; set to
        ``None`` for a single relaxation at the fixed ``mixing_ratio``.

    center : bool, default=True
        Whether to center the input data (subtract mean) before computing
        pairwise distances.

    init : array-like of shape (n_samples, n_components) or None, default=None
        Initial embedding coordinates. If None, classical MDS is used.

    verbose : bool, default=False
        If True, print progress information during fitting.

    progress_bar : bool, default=False
        If True, display tqdm progress bars for the two long-running
        point-by-point loops: the pointwise global optimization during
        :meth:`fit` and the out-of-sample projection during
        :meth:`transform`. Requires the optional dependency ``tqdm``.

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

    n_iter_ : int
        Number of optimizer iterations summed over the full-embedding
        optimization passes (MDS refinement, the main optimization, and the
        per-cycle relaxations of global optimization). The individual
        pointwise grid moves are not counted.

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
      >>> sm = SketchMap(n_components=2)
      >>> embedding = sm.fit_transform(X)
      >>> print(embedding.shape)
      (100, 2)

    Using specific sigmoid parameters:

      >>> sm = SketchMap(
      ...     n_components=2, sigma=7.0, a_high=4.0, b_high=2.0, a_low=2.0, b_low=2.0
      ... )
      >>> embedding = sm.fit_transform(X)
      >>> print(embedding.shape)
      (100, 2)
    """

    def __init__(
        self,
        n_components=2,
        sigma=None,
        a_high=None,
        b_high=None,
        a_low=None,
        b_low=None,
        mds_opt_steps=100,
        optimizer="L-BFGS-B",
        max_iter=1000,
        global_opt_steps="auto",
        mixing_ratio=0.0,
        mixing_schedule="auto",
        center=True,
        init=None,
        verbose=False,
        progress_bar=False,
    ):
        self.n_components = n_components
        self.sigma = sigma
        self.a_high = a_high
        self.b_high = b_high
        self.a_low = a_low
        self.b_low = b_low
        self.mds_opt_steps = mds_opt_steps
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.global_opt_steps = global_opt_steps
        self.mixing_ratio = mixing_ratio
        self.mixing_schedule = mixing_schedule
        self.center = center
        self.init = init
        self.verbose = verbose
        self.progress_bar = progress_bar

    def _resolve_global_opt(self):
        """Resolve the mixing-ratio annealing schedule for global optimization.

        Returns
        -------
        schedule : tuple of float or None
            Mixing ratios to anneal through (highest, MDS-like, first), or
            None to skip global optimization.
        """
        n_levels = self.global_opt_steps
        if n_levels == "auto":
            n_levels = 5
        elif n_levels is None:
            n_levels = 0
        elif not isinstance(n_levels, (int, np.integer)) or n_levels < 0:
            raise ValueError(
                'global_opt_steps must be a non-negative int, "auto" or None, '
                f"got {self.global_opt_steps!r}"
            )

        if n_levels == 0:
            return None

        schedule = self.mixing_schedule
        if isinstance(schedule, str):
            if schedule != "auto":
                raise ValueError(
                    f'mixing_schedule must be a sequence, "auto" or None, '
                    f"got {self.mixing_schedule!r}"
                )
            # Graduated optimization: anneal from the raw-distance MDS stress
            # (mixing 1.0, smooth) down to the pure sigmoid stress (mixing 0.0).
            schedule = tuple(0.5**c for c in range(n_levels - 1)) + (0.0,)
        elif schedule is None:
            # no annealing requested: a single relaxation at the fixed mixing
            schedule = (self.mixing_ratio,)
        else:
            schedule = tuple(schedule)

        return schedule

    # Stress and gradient computation

    def _stress_and_grad(
        self,
        x,
        hd_distances,
        hd_transformed,
        weights,
        total_weight,
        mixing_ratio,
        use_transform=True,
    ):
        r"""Compute the Sketch-Map stress and its gradient in one pass.

        The stress measures the discrepancy between (transformed) high-D
        and (transformed) low-D distances::

            stress = sum_{i<j} w_ij [ (1-m) (s_hd(D_ij) - s_ld(d_ij))^2
                                      + m (D_ij - d_ij)^2 ] / total_weight

        where ``D_ij``/``d_ij`` are high-/low-D distances, ``s_hd``/``s_ld``
        the corresponding sigmoids and ``m`` the mixing ratio. The gradient
        follows the C++ ``NLDRITERChi::set_vars`` weighted path::

            dstress/dx_i = -2/tw sum_j c_ij (x_i - x_j),
            c_ij = w_ij ((1-m)(s_hd - s_ld) s_ld' + m (D - d)) / d

        Value and gradient share one ``cdist`` and one sigmoid evaluation,
        which roughly halves the cost per optimizer iteration compared to
        evaluating them separately.

        Parameters
        ----------
        x : ndarray of shape (n_samples * n_components,)
            Flattened low-dimensional coordinates.
        hd_distances : ndarray of shape (n_samples, n_samples)
            High-dimensional pairwise distances.
        hd_transformed : ndarray of shape (n_samples, n_samples)
            Sigmoid-transformed high-dimensional distances.
        weights : ndarray of shape (n_samples, n_samples) or None
            Pairwise weight matrix, or None for the uniform (all-ones) case,
            which skips building and multiplying the matrix.
        total_weight : float
            Sum of upper-triangle weights (for normalization).
        mixing_ratio : float
            Balance between transformed (0) and raw (1) stress.
        use_transform : bool, default=True
            Whether to apply the sigmoid to low-D distances.

        Returns
        -------
        stress : float
        gradient : ndarray of shape (n_samples * n_components,)
        """
        n_samples = hd_distances.shape[0]
        # work in the fitted dtype (scipy's x and cdist's output are float64)
        embedding = x.reshape(n_samples, self.n_components).astype(
            hd_distances.dtype, copy=False
        )
        ld_distances = cdist(embedding, embedding).astype(
            hd_distances.dtype, copy=False
        )

        if use_transform:
            ld_transformed, ld_derivative = _sigmoid_fdf(
                ld_distances, *self._ld_sigmoid_
            )
        else:
            ld_transformed, ld_derivative = ld_distances, 1.0

        m = mixing_ratio
        diff_transformed = hd_transformed - ld_transformed
        diff_raw = hd_distances - ld_distances

        stress_terms = (1.0 - m) * diff_transformed**2 + m * diff_raw**2
        coeff = (1.0 - m) * diff_transformed * ld_derivative + m * diff_raw
        # weights is None for the uniform case: skip the all-ones matrix entirely
        if weights is not None:
            stress_terms = weights * stress_terms
            coeff = weights * coeff

        # factor 0.5: each pair is counted twice in the full matrix (diagonal is 0)
        stress = 0.5 * np.sum(stress_terms, dtype=np.float64) / total_weight

        # floor avoids 0/0 at coincident points (diagonal is zeroed next); tiny()
        # rather than a literal so it does not underflow to 0 in float32
        coeff /= np.maximum(ld_distances, np.finfo(ld_distances.dtype).tiny)
        np.fill_diagonal(coeff, 0.0)

        # sum_j c_ij (x_i - x_j) for all i, via two matrix products
        gradient = coeff.sum(axis=1)[:, None] * embedding - coeff @ embedding
        gradient *= -2.0 / total_weight

        return float(stress), gradient.ravel().astype(np.float64, copy=False)

    def _compute_stress(self, embedding, *args, **kwargs):
        """Stress value only (used for reporting); see :meth:`_stress_and_grad`."""
        return self._stress_and_grad(np.ravel(embedding), *args, **kwargs)[0]

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
        """Run gradient-based optimization of the full embedding.

        Parameters
        ----------
        initial_embedding : ndarray of shape (n_samples, n_components)
            Starting coordinates.
        hd_distances, hd_transformed, weights, total_weight, mixing_ratio,
        use_transform :
            See :meth:`_stress_and_grad`.
        n_steps : int
            Maximum number of optimizer iterations.

        Returns
        -------
        optimized_embedding : ndarray of shape (n_samples, n_components)
        final_stress : float
        """
        result = minimize(
            self._stress_and_grad,
            initial_embedding.ravel(),
            args=(
                hd_distances,
                hd_transformed,
                weights,
                total_weight,
                mixing_ratio,
                use_transform,
            ),
            jac=True,
            method=self.optimizer,
            options={"maxiter": n_steps, "gtol": 1e-8},
        )
        self._n_iter_total_ += result.nit

        if self.verbose:
            print(f"  Optimization finished: stress = {result.fun:.6f}")

        return result.x.reshape(initial_embedding.shape), result.fun

    # Single-point (chi1) machinery shared by global optimization and
    # out-of-sample projection

    def _point_stress_grad(
        self, x, others, hd_distances, hd_transformed, point_weights, mixing_ratio
    ):
        """Stress and gradient for one movable point against fixed others.

        This is the C++ ``compute_chi1``: the stress contribution of a single
        point at position ``x`` with respect to fixed points ``others``,
        normalized by the sum of the per-point weights.

        Parameters
        ----------
        x : ndarray of shape (n_components,)
            Position of the movable point.
        others : ndarray of shape (n_other, n_components)
            Fixed low-D coordinates (must not contain the movable point).
        hd_distances, hd_transformed : ndarray of shape (n_other,)
            Raw and sigmoid-transformed high-D distances from the movable
            point to each fixed point.
        point_weights : ndarray of shape (n_other,)
            Per-point weights of the fixed points.
        mixing_ratio : float
            Balance between transformed (0) and raw (1) stress.

        Returns
        -------
        stress : float
        gradient : ndarray of shape (n_components,)
        """
        diffs = others - x
        ld_distances = np.sqrt(np.sum(diffs**2, axis=1))

        # Skip coinciding points (C++ "if (ld<=0.0) continue")
        positive = ld_distances > 0.0
        if not np.all(positive):
            diffs = diffs[positive]
            ld_distances = ld_distances[positive]
            hd_distances = hd_distances[positive]
            hd_transformed = hd_transformed[positive]
            point_weights = point_weights[positive]
            if ld_distances.size == 0:
                return 0.0, np.zeros(self.n_components)

        ld_transformed, ld_derivative = _sigmoid_fdf(ld_distances, *self._ld_sigmoid_)
        diff_transformed = hd_transformed - ld_transformed
        diff_raw = hd_distances - ld_distances
        total_weight = np.sum(point_weights)

        m = mixing_ratio
        pair_stress = (1.0 - m) * diff_transformed**2 + m * diff_raw**2
        stress = np.sum(point_weights * pair_stress) / total_weight

        pair_force = (1.0 - m) * diff_transformed * ld_derivative + m * diff_raw
        coeff = 2.0 * point_weights * pair_force / ld_distances
        gradient = (coeff[:, None] * diffs).sum(axis=0) / total_weight

        # float64 gradient for scipy, regardless of the working dtype
        return float(stress), np.asarray(gradient, dtype=np.float64)

    def _global_optimize(
        self,
        embedding,
        hd_distances,
        hd_transformed,
        weights,
        total_weight,
        schedule,
    ):
        """Escape local minima by graduated (annealed) gradient optimization.

        The sigmoid stress is highly non-convex, so plain gradient descent from
        the MDS initialization settles in a local minimum. Instead of the C++
        tool's pointwise grid search -- which jumps each point to the lowest
        single-point-stress grid cell and therefore flings HD-outlier points
        (whose stress only decreases outward) to the periphery -- we morph the
        objective from the well-behaved MDS stress toward the full sigmoid
        stress and re-relax with L-BFGS at each step.

        Concretely, the mixing ratio is annealed through ``schedule`` (e.g.
        ``1.0 -> ... -> 0.0``); at ``m=1`` the stress is the raw-distance MDS
        stress (smooth, ~convex), at ``m=0`` it is the pure sigmoid stress.
        Each level warm-starts from the previous one, so the solution tracks
        continuously into the sigmoid basin. Being purely gradient-based, a
        saturated point simply stops where its gradient vanishes -- it is never
        pushed to the boundary, so no spurious outliers are created.

        Parameters
        ----------
        embedding : ndarray of shape (n_samples, n_components)
            Current embedding (starting point, e.g. the MDS-refined coords).
        hd_distances, hd_transformed, weights, total_weight :
            See :meth:`_stress_and_grad`.
        schedule : sequence of float
            Mixing ratios to anneal through, highest (MDS-like) first.

        Returns
        -------
        optimized_embedding : ndarray of shape (n_samples, n_components)
        final_stress : float
            Stress at the user-facing ``mixing_ratio`` (typically the pure
            sigmoid stress, regardless of the annealing path).
        """
        if self.verbose:
            print(
                f"\n=== Global optimization "
                f"(annealed gradient, mixing schedule {list(schedule)!r}) ==="
            )

        levels = _maybe_tqdm(schedule, self.progress_bar, desc="Annealing")
        for mixing in levels:
            embedding, level_stress = self._optimize(
                embedding,
                hd_distances,
                hd_transformed,
                weights,
                total_weight,
                mixing,
                _GLOBAL_OPT_REFINE_STEPS,
            )
            if self.verbose:
                print(f"  mixing={mixing:.4g}: stress = {level_stress:.6f}")

        final_stress = self._compute_stress(
            embedding,
            hd_distances,
            hd_transformed,
            weights,
            total_weight,
            self.mixing_ratio,
        )

        if self.verbose:
            print(f"  Grid optimization finished: stress = {final_stress:.6f}")

        return embedding, final_stress

    # Public API: fit, transform, fit_transform

    def fit(self, X, y=None, sample_weight=None):
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
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights. Samples with higher weights have more
            influence on the embedding (pair weights are products of sample
            weights). Default is uniform weights.

        Returns
        -------
        self : SketchMap
            Returns the fitted instance.
        """
        gopt_schedule = self._resolve_global_opt()

        if self.optimizer not in ("L-BFGS-B", "CG"):
            raise ValueError(
                f"optimizer must be 'L-BFGS-B' or 'CG', got {self.optimizer!r}"
            )

        # accept float32 or float64 and work in that dtype throughout: float32
        # halves the n x n distance/sigmoid matrices and is plenty for a 2D map
        X = validate_data(self, X, reset=True, dtype=[np.float64, np.float32])

        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(
                f"Found array with {n_samples} sample(s) while minimum of 2 required."
            )

        self.n_samples_ = n_samples
        self.n_features_ = n_features
        self._n_iter_total_ = 0

        if self.verbose:
            print(f"Fitting Sketch-Map: {n_samples} samples, {n_features} features")

        # Compute pairwise distances

        if self.center:
            self._X_mean_ = X.mean(axis=0, keepdims=True)
        else:
            self._X_mean_ = np.zeros((1, n_features), dtype=X.dtype)

        # Stored for out-of-sample projection (transform on new points)
        self._X_processed_ = X - self._X_mean_

        # cdist returns float64; keep the fitted dtype (see ``validate_data`` above)
        hd_distances = cdist(self._X_processed_, self._X_processed_).astype(
            X.dtype, copy=False
        )

        # Determine sigmoid parameters: user-provided keyword arguments take
        # precedence, anything left as None falls back to the auto-estimate.

        suggested, analysis = suggest_sigmoid_params(hd_distances, self.n_components)
        self.suggested_params_ = suggested
        self.distance_analysis_ = analysis

        self.params_ = suggested.copy()
        for key in ["sigma", "a_high", "b_high", "a_low", "b_low"]:
            value = getattr(self, key)
            if value is not None:
                self.params_[key] = value

        self._ld_sigmoid_ = (
            self.params_["sigma"],
            self.params_["a_low"],
            self.params_["b_low"],
        )

        if self.verbose:
            print(
                "Using sigmoid parameters: "
                f"sigma = {self.params_['sigma']:.4f}, "
                f"a_high = {self.params_['a_high']:.2f}, "
                f"b_high = {self.params_['b_high']:.2f}, "
                f"a_low = {self.params_['a_low']:.2f}, "
                f"b_low = {self.params_['b_low']:.2f} "
                f"(peak distance = {analysis['peak_distance']:.4f})"
            )

        hd_transformed = sigmoid_transform(
            hd_distances,
            self.params_["sigma"],
            self.params_["a_high"],
            self.params_["b_high"],
        )

        # Set up the pairwise weight matrix. ``weights = None`` is the uniform
        # case: the stress code treats it as all-ones without ever building the
        # n x n matrix, which saves ~800 MB at 10k landmarks.

        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=X.dtype)
            if w.shape[0] != n_samples:
                raise ValueError(
                    f"sample_weight length {w.shape[0]} != n_samples {n_samples}"
                )
            self._sample_weight_ = w
            weights = np.outer(w, w)
            total_weight = float(np.sum(np.triu(weights, k=1)))
        else:
            self._sample_weight_ = np.ones(n_samples, dtype=X.dtype)
            weights = None
            total_weight = n_samples * (n_samples - 1) / 2.0

        # Initialize embedding

        if self.init is not None:
            embedding = np.asarray(self.init, dtype=X.dtype).copy()
            if embedding.shape != (n_samples, self.n_components):
                raise ValueError(
                    f"init has shape {embedding.shape}, expected "
                    f"({n_samples}, {self.n_components})"
                )
            if self.verbose:
                print(f"Using provided initialization, shape = {embedding.shape}")
        else:
            if self.verbose:
                print("Initializing with classical MDS...")
            embedding = classical_mds(hd_distances, self.n_components)

        # Refine the initial coordinates against the raw distances (no sigmoid)

        if self.mds_opt_steps > 0 and self.init is None:
            if self.verbose:
                print(f"\n=== MDS refinement ({self.mds_opt_steps} steps) ===")
            embedding, _ = self._optimize(
                embedding,
                hd_distances,
                hd_distances,  # target = raw distances
                weights,
                total_weight,
                mixing_ratio=1.0,  # pure raw distance stress
                n_steps=self.mds_opt_steps,
                use_transform=False,
            )

        # Main optimization of the sigmoid-transformed stress

        stress = 0.0
        if self.max_iter > 0:
            if self.verbose:
                print(f"\n=== Main optimization ({self.max_iter} steps) ===")
            embedding, stress = self._optimize(
                embedding,
                hd_distances,
                hd_transformed,
                weights,
                total_weight,
                self.mixing_ratio,
                self.max_iter,
            )

        # Global optimization: graduated (annealed) gradient descent

        if gopt_schedule is not None:
            embedding, stress = self._global_optimize(
                embedding,
                hd_distances,
                hd_transformed,
                weights,
                total_weight,
                gopt_schedule,
            )

        self.embedding_ = embedding
        self.stress_ = stress
        self.n_iter_ = self._n_iter_total_

        if self.verbose:
            print(f"\nSketch-Map fitting complete! Final stress: {stress:.6f}")

        return self

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit the model and return the embedding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for scikit-learn API compatibility.
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights.

        Returns
        -------
        embedding : ndarray of shape (n_samples, n_components)
            Low-dimensional embedding coordinates.
        """
        self.fit(X, y, sample_weight=sample_weight)
        return self.embedding_

    def transform(self, X, n_jobs=None):
        """Project data to the embedding space.

        For points that coincide with a training row, returns the
        joint-optimized embedding stored from ``fit``. For new points, runs
        an out-of-sample projection: each new point's low-D position is
        chosen to minimize the single-point Sketch-Map stress against the
        fixed training landmarks (using the same grid + bicubic + gradient
        refinement machinery as the C++ ``dimproj`` tool).

        This makes the typical landmark workflow trivial: fit Sketch-Map on
        a few hundred FPS landmarks (cheap), then ``transform`` the full
        dataset. Points are processed in row-blocks so memory stays bounded
        regardless of ``len(X)``, and the projection is embarrassingly
        parallel, so large datasets can be spread across cores with ``n_jobs``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Points to embed. May be training rows, new rows, or a mix.
        n_jobs : int or None, default=None
            Number of parallel workers for the projection (``joblib``
            semantics: ``None``/1 = serial, ``-1`` = all cores). Blocks are
            projected independently, so this gives a near-linear speedup on
            large datasets.

        Returns
        -------
        embedding : ndarray of shape (n_samples, n_components)
            Low-D coordinates for each input row.
        """
        check_is_fitted(self, "embedding_")
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        X_processed = X - self._X_mean_

        n_landmarks = self._X_processed_.shape[0]
        batch = max(1, _TRANSFORM_BATCH_ELEMENTS // n_landmarks)
        blocks = [
            X_processed[i : i + batch] for i in range(0, X_processed.shape[0], batch)
        ]

        if n_jobs in (None, 1):
            blocks = _maybe_tqdm(blocks, self.progress_bar, desc="Projecting")
            parts = [self._transform_block(block) for block in blocks]
        else:
            from joblib import Parallel, delayed

            parts = Parallel(n_jobs=n_jobs)(
                delayed(self._transform_block)(block) for block in blocks
            )

        return np.vstack(parts)

    def _transform_block(self, X_block):
        """Project one row-block of centered points.

        Rows coinciding with a landmark (within ~1e-8 sigma, indistinguishable
        at the map's resolution) are snapped to that landmark's coordinates;
        the rest are projected with :meth:`_project_oos`.
        """
        hd_distances = cdist(X_block, self._X_processed_)
        nearest = np.argmin(hd_distances, axis=1)
        nearest_dist = hd_distances[np.arange(X_block.shape[0]), nearest]
        matched = nearest_dist <= 1e-8 * self.params_["sigma"]

        out = np.empty((X_block.shape[0], self.n_components))
        out[matched] = self.embedding_[nearest[matched]]
        if not np.all(matched):
            out[~matched] = self._project_oos(hd_distances[~matched])
        return out

    def _project_oos(self, hd_distances):
        """Project new points onto the fitted map from their HD distances.

        For each new point this runs the same per-point algorithm as the
        in-sample grid global optimization, but with all training landmarks
        held fixed and pure sigmoid stress (``mixing_ratio = 0``, the
        objective the C++ ``dimproj`` tool minimizes):

        1. Apply the high-D sigmoid to the distances to the landmarks.
        2. Evaluate the single-point stress on a coarse 2D grid covering the
           embedding (one matrix product for all new points at once).
        3. Bicubic-interpolate to a finer grid and pick the minimum.
        4. Refine with L-BFGS-B from the grid minimum and from the low-D
           positions of the two nearest landmarks, keeping the best result.

        The multi-seed refinement in step 4 absorbs most of the sigmoid
        stress non-convexity: if a new point is close to landmark ``k`` in
        high dimensions, its low-D position should be close to that
        landmark's, but a coarse grid can miss the corresponding stress dip
        and lock onto a spurious far-away minimum.

        Parameters
        ----------
        hd_distances : ndarray of shape (n_new, n_landmarks)
            High-dimensional distances from each new point to the training
            landmarks (computed on centered data).

        Returns
        -------
        embedding_new : ndarray of shape (n_new, n_components)
        """
        if self.n_components != 2:
            raise NotImplementedError(
                "Out-of-sample projection is currently only implemented for "
                f"n_components=2; got {self.n_components}."
            )

        landmark_emb = self.embedding_
        landmark_w = self._sample_weight_
        total_weight = np.sum(landmark_w)

        hd_transformed = sigmoid_transform(
            hd_distances,
            self.params_["sigma"],
            self.params_["a_high"],
            self.params_["b_high"],
        )

        # Grid spans the landmark embedding with a 1.2x margin
        extent = float(np.max(np.abs(landmark_emb))) * 1.2
        if extent < 1e-10:
            extent = 1.0
        gx = np.linspace(-extent, extent, _GRID_COARSE_POINTS)
        gx_fine = np.linspace(-extent, extent, _GRID_FINE_POINTS)

        grid_x, grid_y = np.meshgrid(gx, gx, indexing="ij")
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        # Transformed grid -> landmark distances, shared by all new points
        fld_grid = sigmoid_transform(
            cdist(grid_points, landmark_emb), *self._ld_sigmoid_
        )  # (n_grid, n_landmarks)

        # Grid stress for all new points at once. Expanding
        # sum_j w_j (fhd_ij - fld_gj)^2 turns the per-point grid scan into a
        # single matrix product:
        #   stress_ig = sum_j w_j fhd_ij^2 - 2 sum_j w_j fhd_ij fld_gj
        #               + sum_j w_j fld_gj^2
        grid_stress_all = (
            (hd_transformed**2 @ landmark_w)[:, None]
            - 2.0 * (hd_transformed * landmark_w) @ fld_grid.T
            + (fld_grid**2 @ landmark_w)[None, :]
        ) / total_weight  # (n_new, n_grid)

        n_new = hd_distances.shape[0]
        embedding_new = np.empty((n_new, 2))
        bounds = [(-extent, extent), (-extent, extent)]

        def objective(x, point_hd_distances, point_hd_transformed):
            return self._point_stress_grad(
                x,
                landmark_emb,
                point_hd_distances,
                point_hd_transformed,
                landmark_w,
                mixing_ratio=0.0,
            )

        for i in range(n_new):
            grid_stress = grid_stress_all[i].reshape(
                _GRID_COARSE_POINTS, _GRID_COARSE_POINTS
            )

            # Seed 1: bicubic-interpolated minimum on the fine grid
            interp = RectBivariateSpline(gx, gx, grid_stress)
            fine_stress = interp(gx_fine, gx_fine)
            min_idx = np.unravel_index(np.argmin(fine_stress), fine_stress.shape)
            seeds = [np.array([gx_fine[min_idx[0]], gx_fine[min_idx[1]]])]

            # Seeds 2-3: low-D positions of the two nearest HD landmarks
            seeds.extend(landmark_emb[np.argsort(hd_distances[i])[:2]])

            best_x, best_stress = None, np.inf
            for seed in seeds:
                result = minimize(
                    objective,
                    seed,
                    args=(hd_distances[i], hd_transformed[i]),
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 100},
                )
                if result.fun < best_stress:
                    best_stress = result.fun
                    best_x = result.x
            embedding_new[i] = best_x

        return embedding_new
