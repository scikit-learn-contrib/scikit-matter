"""
Sketch-Map: Nonlinear dimensionality reduction

References
----------
See Ceriotti et al. [Ceriotti2011]_, which introduces Sketch-Map, and the follow-up
[Ceriotti2013]_.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data

from ._sketchmap_utils import (
    _maybe_tqdm,
    _sigmoid_fdf,
    classical_mds,
    sigmoid_transform,
    suggest_sigmoid_params,
)


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
      4. graduated global optimization with an annealed mixing schedule.

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

    init : array-like of shape (n_samples, n_components) or None, default=None
        Initial embedding coordinates. If None, classical MDS is used.

    verbose : bool, default=False
        If True, print progress information during fitting.

    progress_bar : bool, default=False
        If True, display a tqdm progress bar over the annealing levels of
        :meth:`fit`. Requires the optional dependency ``tqdm``.

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
        The auto-estimated sigmoid parameters, whether or not they were used.

    distance_analysis_ : dict
        Distance distribution analysis: peak distance, Gaussian range
        estimates and histogram data.

    n_iter_ : int
        Optimizer iterations summed over the full-embedding passes (MDS
        refinement, main optimization and the per-level relaxations of global
        optimization).

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
        self.init = init
        self.verbose = verbose
        self.progress_bar = progress_bar

    def _resolve_global_opt(self):
        """Mixing ratios to anneal through (MDS-like first), or None to skip."""
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
            schedule = tuple(0.5**c for c in range(n_levels - 1)) + (0.0,)
        elif schedule is None:
            schedule = (self.mixing_ratio,)
        else:
            schedule = tuple(schedule)

        return schedule

    def _stress_and_grad(
        self, flat_embedding, problem, mixing_ratio, use_transform=True
    ):
        r"""Stress and gradient of the whole embedding, in one pass::

        stress = sum_{i<j} w_ij [ (1-m) (s_hd(D_ij) - s_ld(d_ij))^2
                                  + m (D_ij - d_ij)^2 ] / total_weight
        dstress/dx_i = -2/tw sum_j c_ij (x_i - x_j),
        c_ij = w_ij ((1-m)(s_hd - s_ld) s_ld' + m (D - d)) / d
        """
        hd_distances, hd_transformed, weights, total_weight = problem
        n_samples = hd_distances.shape[0]
        # scipy and cdist return float64, so cast back to the fitted dtype
        embedding = flat_embedding.reshape(n_samples, self.n_components).astype(
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

        diff_transformed = hd_transformed - ld_transformed
        diff_raw = hd_distances - ld_distances

        sigmoid_share = 1.0 - mixing_ratio
        stress_terms = sigmoid_share * diff_transformed**2 + mixing_ratio * diff_raw**2
        pair_coeff = (
            sigmoid_share * diff_transformed * ld_derivative + mixing_ratio * diff_raw
        )
        if weights is not None:
            stress_terms = weights * stress_terms
            pair_coeff = weights * pair_coeff

        # 0.5: the full matrix counts each pair twice
        stress = 0.5 * np.sum(stress_terms, dtype=np.float64) / total_weight

        # tiny() rather than a literal so it does not underflow in float32
        pair_coeff /= np.maximum(ld_distances, np.finfo(ld_distances.dtype).tiny)
        np.fill_diagonal(pair_coeff, 0.0)

        gradient = pair_coeff.sum(axis=1)[:, None] * embedding - pair_coeff @ embedding
        gradient *= -2.0 / total_weight

        return float(stress), gradient.ravel().astype(np.float64, copy=False)

    def _optimize(
        self, initial_embedding, problem, mixing_ratio, n_steps, use_transform=True
    ):
        """Relax the full embedding for at most ``n_steps`` optimizer steps."""
        result = minimize(
            self._stress_and_grad,
            initial_embedding.ravel(),
            args=(problem, mixing_ratio, use_transform),
            jac=True,
            method=self.optimizer,
            options={"maxiter": n_steps, "gtol": 1e-8},
        )
        self._n_iter_total_ += result.nit

        if self.verbose:
            print(f"  Optimization finished: stress = {result.fun:.6f}")

        return result.x.reshape(initial_embedding.shape)

    def _global_optimize(self, embedding, problem, schedule, global_refine_steps=100):
        """Escape local minima by graduated (annealed) gradient optimization.

        The sigmoid stress is non-convex, so the mixing ratio is annealed
        through ``schedule``, morphing the objective from the smooth
        raw-distance MDS stress (``m=1``) into the pure sigmoid stress
        (``m=0``) and warm-starting each level from the previous one. Being
        purely gradient-based, saturated points stop where their gradient
        vanishes instead of being flung outward as a grid search would.
        """
        levels = _maybe_tqdm(schedule, self.progress_bar, desc="Annealing")
        for mixing in levels:
            if self.verbose:
                print(f"  mixing={mixing:.4g}")
            embedding = self._optimize(embedding, problem, mixing, global_refine_steps)

        return embedding

    def _resolve_params(self, hd_distances):
        """Combine the auto-estimated sigmoid parameters with user overrides."""
        suggested, analysis = suggest_sigmoid_params(hd_distances, self.n_components)
        self.suggested_params_ = suggested
        self.distance_analysis_ = analysis

        self.params_ = suggested.copy()
        for key in ("sigma", "a_high", "b_high", "a_low", "b_low"):
            value = getattr(self, key)
            if value is not None:
                self.params_[key] = value

        self._ld_sigmoid_ = (
            self.params_["sigma"],
            self.params_["a_low"],
            self.params_["b_low"],
        )

    def _pair_weights(self, sample_weight, n_samples, dtype):
        """Pairwise weights as w_i w_j, or None for the uniform case."""
        if sample_weight is None:
            return None, n_samples * (n_samples - 1) / 2.0

        per_sample = np.asarray(sample_weight, dtype=dtype)
        if per_sample.shape[0] != n_samples:
            raise ValueError(
                f"sample_weight length {per_sample.shape[0]} != n_samples {n_samples}"
            )
        weights = np.outer(per_sample, per_sample)
        return weights, float(np.sum(np.triu(weights, k=1)))

    def _initial_embedding(self, hd_distances, n_samples, dtype):
        """Starting coordinates: the user's ``init``, else classical MDS."""
        if self.init is None:
            if self.verbose:
                print("Initializing with classical MDS...")
            return classical_mds(hd_distances, self.n_components)

        embedding = np.asarray(self.init, dtype=dtype).copy()
        if embedding.shape != (n_samples, self.n_components):
            raise ValueError(
                f"init has shape {embedding.shape}, expected "
                f"({n_samples}, {self.n_components})"
            )
        return embedding

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

        X = validate_data(self, X, reset=True, dtype=[np.float64, np.float32])

        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(
                f"Found array with {n_samples} sample(s) while minimum of 2 required."
            )

        self._n_iter_total_ = 0
        if self.verbose:
            print(f"Fitting Sketch-Map: {n_samples} samples, {n_features} features")

        hd_distances = cdist(X, X).astype(X.dtype, copy=False)

        self._resolve_params(hd_distances)
        if self.verbose:
            formatted = ", ".join(f"{k} = {v:.4g}" for k, v in self.params_.items())
            print(f"Using sigmoid parameters: {formatted}")

        hd_transformed = sigmoid_transform(
            hd_distances,
            self.params_["sigma"],
            self.params_["a_high"],
            self.params_["b_high"],
        )
        weights, total_weight = self._pair_weights(sample_weight, n_samples, X.dtype)
        problem = (hd_distances, hd_transformed, weights, total_weight)

        embedding = self._initial_embedding(hd_distances, n_samples, X.dtype)

        if self.mds_opt_steps > 0 and self.init is None:
            if self.verbose:
                print(f"\n=== MDS refinement ({self.mds_opt_steps} steps) ===")
            embedding = self._optimize(
                embedding,
                problem,
                mixing_ratio=1.0,
                n_steps=self.mds_opt_steps,
                use_transform=False,
            )

        if self.max_iter > 0:
            if self.verbose:
                print(f"\n=== Main optimization ({self.max_iter} steps) ===")
            embedding = self._optimize(
                embedding, problem, self.mixing_ratio, self.max_iter
            )

        if gopt_schedule is not None:
            if self.verbose:
                print(
                    f"\n=== Global optimization "
                    f"(annealed gradient, mixing schedule {list(gopt_schedule)!r}) ==="
                )
            embedding = self._global_optimize(embedding, problem, gopt_schedule)

        self.embedding_ = embedding
        self.stress_ = self._stress_and_grad(
            embedding.ravel(), problem, self.mixing_ratio
        )[0]
        self.n_iter_ = self._n_iter_total_

        if self.verbose:
            print(f"\nSketch-Map fitting complete! Final stress: {self.stress_:.6f}")

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
