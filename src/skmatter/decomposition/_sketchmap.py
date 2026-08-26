import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data

from ..utils import get_progress_bar
from ._sketchmap_utils import (
    _classical_mds,
    _sigmoid_and_derivative,
    _sigmoid_transform,
    _suggest_sigmoid_params,
)

# Constants taken from the original C++ implementation
_GRID_COARSE_POINTS = 21
_GRID_FINE_POINTS = 201
_GRID_MARGIN = 1.2
_GRID_RELAX_STEPS = 10
_GRID_ANNEAL_RELAX_STEPS = 3
_GRID_ANNEAL_REFINE_STEPS = 50
_GRID_ANNEAL_MAX_ROUNDS = 10
_GRID_ANNEAL_FACTOR_BOUNDS = (0.1, 0.5)
_GRID_ANNEAL_TOL = 1e-2


class SketchMap(TransformerMixin, BaseEstimator):
    r"""
    sketch-map embeds high-dimensional data by matching pairwise distances, but only
    over a chosen range of scales. MDS tries to reproduce every distance and t-SNE keeps
    only local neighborhoods. sketch-map instead passes both the high- and
    low-dimensional distances through a sigmoid before comparing them. The sigmoid
    saturates, so distances far from the switching scale :math:`\sigma` contribute
    little and the fit concentrates on the intermediate range.

    Distances much shorter than :math:`\sigma` collapse toward 0. In a molecular
    ensemble these are mostly thermal noise and should not drive the layout. Distances
    much longer than :math:`\sigma` saturate toward 1. In high dimensions long distances
    are unreliable, so their exact values are dropped while their ordering is kept.
    Distances near :math:`\sigma` pass through almost unchanged and carry the structure
    the embedding is built around.

    The sigmoid is

    .. math::

        s(r) = 1 - \left(1 + A \left(\frac{r}{\sigma}\right)^a\right)^{-b/a},
        \qquad A = 2^{a/b} - 1.

    Here :math:`\sigma` is the switching distance, where :math:`s(\sigma) = 0.5`. The
    exponent :math:`a` sets how sharply :math:`s \to 0` at short range and :math:`b` how
    sharply :math:`s \to 1` at long range. The prefactor :math:`A` is fixed by :math:`a`
    and :math:`b` so the curve always crosses one half at :math:`\sigma`.

    High- and low-dimensional distances use their own exponents (``a_high``, ``b_high``
    and ``a_low``, ``b_low``), letting the low dimension take a gentler curve to make
    room for the structure that cannot fit otherwise.

    All five parameters are estimated automatically from the data when left at ``None``.
    The values actually used are stored in ``params_`` after fitting. If one needs to
    tune them by hand:

      - ``sigma`` is the most important parameter, distances below it are treated as
        "close", distances above as "far". Place it just below the peak of the pairwise
        distance histogram.
      - ``a_high``, ``b_high`` shape the high-dimensional sigmoid: larger ``a_high``
        compresses short-range (noise) distances more aggressively, smaller ``b_high``
        saturates long distances more softly.
      - ``a_low``, ``b_low`` shape the low-dimensional sigmoid and are usually smaller
        than their high-dimensional counterparts, to compensate for the volume
        difference between the spaces.

    The fit proceeds through four stages:

      1. classical MDS for the initial coordinates (skipped when ``init`` is given),
      2. refinement of those coordinates against the raw distances (``mds_opt_steps``),
      3. main optimization of the sigmoid-transformed stress (``max_iter``),
      4. global optimization to escape local minima of the non-convex sigmoid stress
         (``global_opt_steps`` rounds of ``global_optimizer``, with optional mixing
         annealing via ``mixing_schedule``).

    Stages 1-3 always run the same way. The accuracy/cost trade-off lives in stage 4:

      - ``global_optimizer="gradient"`` (default): repeated L-BFGS relaxations. Fast,
        any dimensionality.
      - ``global_optimizer="grid"``: the reference implementation's pointwise relocation
        sweeps. Lower stress, 2D only, considerably slower.
      - ``global_optimizer="grid"`` with ``mixing_schedule="auto"``: also anneals the
        mixing ratio like the reference pipeline. The slowest mode and the one that
        reproduces published sketch-maps from scratch.

    The optimization is deterministic.

    Every stage works on the full pairwise distance matrix, so cost and memory grow as
    :math:`O(n^2)`. On large datasets the estimator is therefore fit on a set of
    landmarks, chosen for example by farthest-point sampling and weighted by the
    population of their Voronoi cells through ``sample_weight``.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in the target embedding space.

    sigma : float or None, default=None
        Switching distance where :math:`s(\sigma) = 0.5`, applied to both the high- and
        low-dimensional distances. If None, it is estimated automatically as 90% of the
        peak of the pairwise distance distribution. When ``sample_weight`` is given the
        histogram counts each pair with weight :math:`w_i w_j`, the same statistic the
        stress sums over, so with Voronoi-weighted landmarks the estimate reflects the
        full dataset rather than the landmarks uniform spread.

    a_high, b_high : float or None, default=None
        Short- and long-range steepness exponents of the high-dimensional sigmoid. If
        None, both are estimated from the distance distribution.

    a_low, b_low : float or None, default=None
        Short- and long-range steepness exponents of the low-dimensional sigmoid. If
        None, both are estimated from the input dimensionality.

    mds_opt_steps : int, default=100
        Number of stage-2 optimization steps against the raw distances (no sigmoid),
        refining the classical MDS initialization before the sigmoid stress. Set to 0 to
        skip the stage. Equivalent to ``-preopt`` refinement of the reference
        C++ implementation.

    optimizer : str, default="L-BFGS-B"
        Algorithm used for every full-embedding optimization (stages 2-4). Options:
        ``"L-BFGS-B"`` or ``"CG"``.

    max_iter : int, default=1000
        Maximum iterations for the stage-3 main optimization of the
        (sigmoid-transformed) stress at the fixed ``mixing_ratio``.

    global_opt_steps : int, "auto" or None, default="auto"
        Number of global optimization rounds run after the main optimization. The
        sigmoid stress is non-convex, so the embedding is refined further to escape
        local minima. What a round does depends on ``global_optimizer``. ``"auto"``
        (default) uses 5 rounds or up to 10 rounds with early stopping for the annealed
        ``global_optimizer="grid"``. Set to 0 or None to disable.

    global_optimizer : {"gradient", "grid"}, default="gradient"
        How the global optimization rounds escape local minima.

        ``"gradient"`` re-optimizes the whole embedding with L-BFGS, optionally
        annealing the mixing ratio (see ``mixing_schedule``). It is cheap, works for any
        ``n_components``, but it only ever moves downhill from the basin it starts in.

        ``"grid"`` scans one point at a time over a grid covering the embedding and
        relocates it when that strictly lowers its stress, briefly optimizing the whole
        embedding after every accepted move. Moving a single point across the map is a
        large enough step to cross into another basin, which the gradient rounds cannot
        do, so it reaches lower stress. It follows the reference C++ implementation.
        Combined with ``mixing_schedule="auto"`` it also anneals the mixing ratio the
        way the reference C++ pipeline does, which is the closest reproduction of the
        published sketch-maps this estimator offers. It costs :math:`O(n^2)` stress
        evaluations per sweep, so it is much slower, and it is restricted to
        ``n_components=2``.

    mixing_ratio : float, default=0.0
        Balance between raw distance stress and transformed distance stress:
          - 0.0: Pure sigmoid-transformed stress
          - 1.0: Pure raw distance stress
          - values in between: linear combination

        Used as the constant mixing ratio for the main optimization stage and as the
        final target of the annealing schedule.

    mixing_schedule : sequence of float, "auto" or None, default=None
        Mixing ratios to anneal through during global optimization. ``None`` (default)
        does not anneal and optimizes at the fixed ``mixing_ratio``, matching a single
        run of the reference C++ implementation, which applies no mixing unless asked.

        With ``global_optimizer="grid"``, ``"auto"`` replicates the annealing loop of
        the reference C++ pipeline: the mixing ratio starts at 1 and is multiplied each
        round by a factor derived from the ratio of the current stresses, so the
        embedding grows gradually from a small MDS-like map into the sketch-map
        solution. Because the raw-distance term is typically orders of magnitude larger
        than the sigmoid term, this error-driven geometric decay is the only way the
        intermediate mixing values pass through the regime where both terms actually
        compete.

        With ``global_optimizer="gradient"``, each level of the schedule is relaxed with
        L-BFGS, warm-started from the previous one. ``"auto"`` anneals geometrically
        from 1.0 down to exactly 0 over ``global_opt_steps`` levels.

    dissimilarity : {"euclidean", "precomputed"}, default="euclidean"
        How ``X`` is interpreted in :meth:`fit`. ``"euclidean"`` computes pairwise
        Euclidean distances from the feature vectors. ``"precomputed"`` treats ``X`` as
        a square pairwise distance matrix, so any distance computed elsewhere --
        periodic, dot-product, or a custom kernel -- can be used directly.

    init : array-like of shape (n_samples, n_components) or None, default=None
        Initial embedding coordinates. If None, classical MDS is used.

    verbose : bool, default=False
        If True, print progress information during fitting.

    progress_bar : bool, default=False
        If True, display a tqdm progress bar over the annealing levels of :meth:`fit`.
        Requires the optional dependency ``tqdm``.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        The fitted low-dimensional embedding coordinates.

    stress_ : float
        Final stress value (lower is better).

    params_ : dict
        The sigmoid parameters actually used (combination of user-provided and
        auto-estimated values).

    suggested_params_ : dict
        The auto-estimated sigmoid parameters, whether or not they were used.

    distance_analysis_ : dict
        Distance distribution analysis: peak distance, Gaussian range estimates and
        histogram data.

    n_iter_ : int
        Optimizer iterations summed over the full-embedding passes (MDS refinement, main
        optimization and the per-level relaxations of global optimization).

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

    Passing a precomputed distance matrix:

      >>> from scipy.spatial.distance import cdist
      >>> distances = cdist(X, X)
      >>> sm = SketchMap(n_components=2, dissimilarity="precomputed")
      >>> embedding = sm.fit_transform(distances)
      >>> print(embedding.shape)
      (100, 2)

    Escaping local minima with the grid global optimizer, at a higher cost:

      >>> sm = SketchMap(n_components=2, global_optimizer="grid")
      >>> embedding = sm.fit_transform(X)
      >>> print(embedding.shape)
      (100, 2)

    Reducing very high-dimensional data with PCA before sketch-map:

      >>> from sklearn.decomposition import PCA
      >>> from sklearn.pipeline import make_pipeline
      >>> pipe = make_pipeline(PCA(n_components=8), SketchMap(n_components=2))
      >>> embedding = pipe.fit_transform(X)
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
        global_optimizer="gradient",
        mixing_ratio=0.0,
        mixing_schedule=None,
        dissimilarity="euclidean",
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
        self.global_optimizer = global_optimizer
        self.mixing_ratio = mixing_ratio
        self.mixing_schedule = mixing_schedule
        self.dissimilarity = dissimilarity
        self.init = init
        self.verbose = verbose
        self.progress_bar = progress_bar

    def _resolve_n_levels(self):
        """Number of global optimization rounds, 0 to skip the stage."""
        n_levels = self.global_opt_steps
        if n_levels == "auto":
            return 5

        if n_levels is None:
            return 0

        if not isinstance(n_levels, (int, np.integer)) or n_levels < 0:
            raise ValueError(
                'global_opt_steps must be a non-negative int, "auto" or None, '
                f"got {self.global_opt_steps!r}"
            )

        return n_levels

    def _resolve_global_opt(self):
        """Mixing ratios to anneal through (MDS-like first) or None to skip."""
        n_levels = self._resolve_n_levels()

        if n_levels == 0:
            return None

        schedule = self.mixing_schedule
        if isinstance(schedule, str):
            if schedule != "auto":
                raise ValueError(
                    f'mixing_schedule must be a sequence, "auto" or None, '
                    f"got {self.mixing_schedule!r}"
                )

            # halve each level, then land exactly on the target
            schedule = tuple(0.5**c for c in range(n_levels - 1)) + (0.0,)
        elif schedule is None:
            schedule = (self.mixing_ratio,)
        else:
            schedule = tuple(schedule)

        return schedule

    def _stress_and_grad(self, flat_embedding, problem, mixing_ratio):
        r"""Stress and gradient of the whole embedding, in one pass.

        Writing :math:`D_{ij}` for a high-dimensional distance, :math:`d_{ij}` for the
        low-dimensional one, :math:`s` and :math:`\tilde{s}` for the two sigmoids,
        :math:`w_{ij} = w_i w_j` for the pair weight and :math:`W` for their sum, the
        stress mixes the transformed and the raw discrepancy in the ratio :math:`m`:

        .. math::

            \chi = \frac{1}{W} \sum_{i<j} w_{ij} \left[
                (1 - m) \left( s(D_{ij}) - \tilde{s}(d_{ij}) \right)^2
                + m \left( D_{ij} - d_{ij} \right)^2 \right]

        Both terms depend on the embedding only through :math:`d_{ij}`, so the two
        gradients share the same form and differ only in the coefficient:

        .. math::

            \frac{\partial \chi}{\partial \mathbf{x}_i}
                = -\frac{2}{W} \sum_j c_{ij}
                  \left( \mathbf{x}_i - \mathbf{x}_j \right),

        .. math::

            c_{ij} = \frac{w_{ij}}{d_{ij}} \left[
                (1 - m) \left( s(D_{ij}) - \tilde{s}(d_{ij}) \right)
                \tilde{s}'(d_{ij}) + m \left( D_{ij} - d_{ij} \right) \right]
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

        # at mixing 1 the sigmoid term drops out, so skip the expensive powers
        if mixing_ratio < 1.0:
            ld_transformed, ld_derivative = _sigmoid_and_derivative(
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

    def _optimize(self, initial_embedding, problem, mixing_ratio, n_steps):
        """Optimize the full embedding for at most ``n_steps`` optimizer steps"""
        result = minimize(
            self._stress_and_grad,
            initial_embedding.ravel(),
            args=(problem, mixing_ratio),
            jac=True,
            method=self.optimizer,
            options={"maxiter": n_steps, "gtol": 1e-8},
        )

        self._n_iter_total_ += result.nit

        if self.verbose:
            print(f"  Optimization finished: stress = {result.fun:.6f}")

        return result.x.reshape(initial_embedding.shape)

    def _progress(self, iterable, description):
        if not self.progress_bar:
            return iterable

        return get_progress_bar()(iterable, desc=description)

    def _global_optimize(self, embedding, problem, schedule, global_refine_steps=100):
        """Escape local minima by graduated (annealed) gradient optimization.

        The sigmoid stress is non-convex, so the mixing ratio is annealed through
        ``schedule``, interpolating between the raw-distance MDS stress (``m=1``) and
        the pure sigmoid stress (``m=0``) and warm-starting each level from the previous
        one. Because it only follows the gradient, points whose distances are already
        saturated do not move, since their gradient vanishes.
        """
        levels = self._progress(schedule, "Annealing")
        for mixing in levels:
            if self.verbose:
                print(f"  mixing={mixing:.4g}")
            embedding = self._optimize(embedding, problem, mixing, global_refine_steps)

        return embedding

    def _point_stress(self, candidates, index, embedding, problem, mixing_ratio):
        """Stress of a single point placed at each of ``candidates``.

        Only the terms involving ``index`` are summed, since the rest of the embedding
        is held fixed
        """
        hd_distances, hd_transformed, weights, _ = problem
        ld_distances = cdist(candidates, embedding)
        ld_transformed = _sigmoid_transform(ld_distances, *self._ld_sigmoid_)

        sigmoid_share = 1.0 - mixing_ratio
        terms = sigmoid_share * (hd_transformed[index] - ld_transformed) ** 2

        if mixing_ratio:
            terms += mixing_ratio * (hd_distances[index] - ld_distances) ** 2

        if weights is not None:
            terms = terms * weights[index]

        terms[:, index] = 0.0

        return terms.sum(axis=1)

    def _grid_relocate(self, embedding, problem, mixing_ratio, relax_steps):
        """Move each point to the best position found anywhere on the map.

        Following the reference implementation, the true single-point stress is
        evaluated on a coarse grid spanning the current embedding, a bicubic interpolant
        of it is scanned on a fine grid covering the whole map, and the interpolated
        winner is re-evaluated with the true stress so interpolation false positives are
        rejected. After every accepted move the whole embedding is briefly optimized,
        letting the rest of the map react before the next point is examined.
        """
        half_width = _GRID_MARGIN * np.sqrt((embedding**2).sum(axis=1)).max()
        if half_width == 0:
            return embedding, 0

        coarse_axis = np.linspace(-half_width, half_width, _GRID_COARSE_POINTS)
        coarse_x, coarse_y = np.meshgrid(coarse_axis, coarse_axis, indexing="ij")
        coarse = np.column_stack([coarse_x.ravel(), coarse_y.ravel()])

        # one cell in from the edge: the rim is where saturated points run off to
        fine_axis = np.linspace(coarse_axis[1], coarse_axis[-2], _GRID_FINE_POINTS)
        fine_x, fine_y = np.meshgrid(fine_axis, fine_axis, indexing="ij")
        fine = np.column_stack([fine_x.ravel(), fine_y.ravel()])

        n_moved = 0
        for index in range(embedding.shape[0]):
            current = self._point_stress(
                embedding[index][None, :], index, embedding, problem, mixing_ratio
            )[0]

            coarse_stress = self._point_stress(
                coarse, index, embedding, problem, mixing_ratio
            )
            interpolant = RectBivariateSpline(
                coarse_axis,
                coarse_axis,
                coarse_stress.reshape(_GRID_COARSE_POINTS, _GRID_COARSE_POINTS),
            )

            # the fine scan is on the interpolant, far cheaper than the real stress
            fine_stress = interpolant(fine_axis, fine_axis, grid=True).ravel()
            best = np.argmin(fine_stress)
            if fine_stress[best] >= current:
                continue

            # the interpolant can promise an improvement that is not there
            candidate = fine[best]
            true_stress = self._point_stress(
                candidate[None, :], index, embedding, problem, mixing_ratio
            )[0]
            if true_stress >= current:
                continue

            embedding[index] = candidate
            n_moved += 1
            # let the rest of the map react before the next point is examined
            embedding = self._optimize(embedding, problem, mixing_ratio, relax_steps)

        return embedding, n_moved

    def _grid_optimize(self, embedding, problem, n_cycles, global_refine_steps=100):
        """Escape local minima by relocating one point at a time on a grid.

        Gradient optimization can only slide the embedding downhill within the basin it
        starts in. Moving a single point across the map is a much larger step, which
        lets the optimization cross into a different basin. Each sweep starts from a
        fully optimized embedding and every accepted move is followed by a short
        optimization.
        """
        embedding = embedding.copy()

        cycles = self._progress(range(n_cycles), "Grid")

        for _ in cycles:
            embedding = self._optimize(
                embedding, problem, self.mixing_ratio, global_refine_steps
            )
            embedding, n_moved = self._grid_relocate(
                embedding, problem, self.mixing_ratio, _GRID_RELAX_STEPS
            )
            if self.verbose:
                print(f"  grid sweep moved {n_moved} points")

        return embedding

    def _grid_optimize_annealed(
        self, embedding, problem, n_rounds, global_refine_steps=100
    ):
        """Grid optimization with the reference implementation's annealing.

        The two stress terms live on very different scales, so instead of a fixed mixing
        schedule the mixing ratio starts at 1 (pure raw-distance stress, whose minimum
        is a MDS-like map) and decays by a factor set from the ratio of the pure-sigmoid
        stress to the current mixed stress.
        """
        embedding = embedding.copy()
        sigmoid_stress = self._stress_and_grad(
            embedding.ravel(), problem, self.mixing_ratio
        )[0]

        # start from pure MDS and let the errors decide how fast to decay
        mixing = 1.0
        previous_error = None

        rounds = self._progress(range(n_rounds), "Annealed grid")

        for round_index in rounds:
            embedding = self._optimize(
                embedding, problem, mixing, _GRID_ANNEAL_REFINE_STEPS
            )
            embedding, n_moved = self._grid_relocate(
                embedding, problem, mixing, _GRID_ANNEAL_RELAX_STEPS
            )
            mixed_error = self._stress_and_grad(embedding.ravel(), problem, mixing)[0]

            if self.verbose:
                print(
                    f"  annealing round {round_index + 1}: mixing={mixing:.3g}, "
                    f"moved {n_moved} points, stress={mixed_error:.6f}"
                )

            # a perfectly embedded dataset has nothing left to anneal
            if mixed_error == 0:
                break

            if (
                previous_error is not None
                and abs(previous_error - mixed_error) / mixed_error < _GRID_ANNEAL_TOL
            ):
                break

            previous_error = mixed_error

            # the closer the two stresses, the more mixing is dropped
            factor = np.clip(
                sigmoid_stress / (sigmoid_stress + mixed_error),
                *_GRID_ANNEAL_FACTOR_BOUNDS,
            )
            mixing *= factor

        embedding = self._optimize(
            embedding, problem, self.mixing_ratio, global_refine_steps
        )
        embedding, n_moved = self._grid_relocate(
            embedding, problem, self.mixing_ratio, _GRID_RELAX_STEPS
        )

        if self.verbose:
            print(f"  final sweep moved {n_moved} points")

        return embedding

    def _resolve_params(self, hd_distances, sample_weight=None):
        """Combine the auto-estimated sigmoid parameters with user overrides"""
        suggested, analysis = _suggest_sigmoid_params(
            hd_distances, self.n_components, sample_weight=sample_weight
        )
        self.suggested_params_ = suggested
        self.distance_analysis_ = analysis

        # anything the user pinned wins over the estimate
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
        """Pairwise weights as w_i w_j, or None for the uniform case"""
        if sample_weight is None:
            return None, n_samples * (n_samples - 1) / 2.0

        per_sample = np.asarray(sample_weight, dtype=dtype)
        if per_sample.shape[0] != n_samples:
            raise ValueError(
                f"sample_weight length {per_sample.shape[0]} != n_samples {n_samples}"
            )

        weights = np.outer(per_sample, per_sample)

        # the normalization counts each pair once, matching the stress
        return weights, float(np.sum(np.triu(weights, k=1)))

    def _initial_embedding(self, hd_distances, n_samples, dtype):
        """Starting coordinates: the user's ``init`` else classical MDS"""
        if self.init is None:
            if self.verbose:
                print("Initializing with classical MDS")

            return _classical_mds(hd_distances, self.n_components)

        embedding = np.asarray(self.init, dtype=dtype).copy()
        if embedding.shape != (n_samples, self.n_components):
            raise ValueError(
                f"init has shape {embedding.shape}, expected "
                f"({n_samples}, {self.n_components})"
            )

        return embedding

    def fit(self, X, y=None, sample_weight=None):
        """Fit the sketch-map embedding to the training data

        The pairwise distances of ``X`` are computed and passed through the sigmoid, the
        sigmoid parameters are estimated where they were not given and the embedding
        coordinates are relaxed until the stress is minimized.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), or (n_samples, n_samples)
            Training data. Feature vectors by default, a square pairwise distance matrix
            when ``dissimilarity="precomputed"``.
        y : Ignored
            Not used, present for scikit-learn API compatibility.
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights. Samples with higher weights have more influence on the
            embedding (pair weights are products of sample weights). Default is uniform
            weights.

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

        if self.dissimilarity not in ("euclidean", "precomputed"):
            raise ValueError(
                "dissimilarity must be 'euclidean' or 'precomputed', "
                f"got {self.dissimilarity!r}"
            )

        if self.global_optimizer not in ("gradient", "grid"):
            raise ValueError(
                "global_optimizer must be 'gradient' or 'grid', "
                f"got {self.global_optimizer!r}"
            )

        if self.global_optimizer == "grid" and self.n_components != 2:
            raise ValueError(
                "global_optimizer='grid' is only implemented for "
                f"n_components=2, got {self.n_components}"
            )

        if self.n_components < 1:
            raise ValueError(
                f"n_components must be at least 1, got {self.n_components}"
            )

        if not 0.0 <= self.mixing_ratio <= 1.0:
            raise ValueError(
                f"mixing_ratio must lie between 0 and 1, got {self.mixing_ratio}"
            )

        for name in ("sigma", "a_high", "b_high", "a_low", "b_low"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        for name in ("mds_opt_steps", "max_iter"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        X = validate_data(self, X, reset=True, dtype=[np.float64, np.float32])

        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError(
                f"Found array with {n_samples} sample(s) while minimum of 2 required."
            )

        if self.n_components > n_samples:
            raise ValueError(
                f"n_components={self.n_components} cannot exceed the number of "
                f"samples, got {n_samples}."
            )

        if self.dissimilarity == "precomputed" and n_features != n_samples:
            raise ValueError(
                "dissimilarity='precomputed' expects a square distance matrix, "
                f"got shape {X.shape}"
            )

        self._n_iter_total_ = 0

        if self.verbose:
            print(f"Fitting sketch-map: {n_samples} samples, {n_features} features")

        if self.dissimilarity == "precomputed":
            hd_distances = X
        else:
            hd_distances = cdist(X, X).astype(X.dtype, copy=False)

        self._resolve_params(hd_distances, sample_weight)
        if self.verbose:
            formatted = ", ".join(f"{k} = {v:.4g}" for k, v in self.params_.items())
            print(f"Using sigmoid parameters: {formatted}")

        hd_transformed = _sigmoid_transform(
            hd_distances,
            self.params_["sigma"],
            self.params_["a_high"],
            self.params_["b_high"],
        )

        weights, total_weight = self._pair_weights(sample_weight, n_samples, X.dtype)

        if total_weight <= 0:
            raise ValueError(
                "sample_weight must contain at least two positive entries, "
                "otherwise no pair carries any weight."
            )

        problem = (hd_distances, hd_transformed, weights, total_weight)

        embedding = self._initial_embedding(hd_distances, n_samples, X.dtype)

        if self.mds_opt_steps > 0 and self.init is None:
            if self.verbose:
                print(f"MDS refinement ({self.mds_opt_steps} steps)")

            embedding = self._optimize(
                embedding,
                problem,
                mixing_ratio=1.0,
                n_steps=self.mds_opt_steps,
            )

        if self.max_iter > 0:
            if self.verbose:
                print(f"Main optimization ({self.max_iter} steps)")

            embedding = self._optimize(
                embedding, problem, self.mixing_ratio, self.max_iter
            )

        if gopt_schedule is not None:
            if self.global_optimizer == "grid":
                if self.mixing_schedule is not None and not isinstance(
                    self.mixing_schedule, str
                ):
                    raise ValueError(
                        "explicit mixing schedules are only supported by "
                        "global_optimizer='gradient'; the grid optimizer takes "
                        "mixing_schedule=None or the adaptive 'auto'"
                    )

                if isinstance(self.mixing_schedule, str):
                    n_rounds = (
                        _GRID_ANNEAL_MAX_ROUNDS
                        if self.global_opt_steps == "auto"
                        else self._resolve_n_levels()
                    )

                    if self.verbose:
                        print(
                            "Global optimization, annealed grid up to "
                            f"{n_rounds} rounds"
                        )
                    embedding = self._grid_optimize_annealed(
                        embedding, problem, n_rounds
                    )
                else:
                    n_cycles = self._resolve_n_levels()
                    if self.verbose:
                        print(f"Global optimization, {n_cycles} grid sweeps")
                    embedding = self._grid_optimize(embedding, problem, n_cycles)
            else:
                if self.verbose:
                    print(
                        f"Global optimization, mixing schedule {list(gopt_schedule)!r}"
                    )
                embedding = self._global_optimize(embedding, problem, gopt_schedule)

        self.embedding_ = embedding

        self.stress_ = self._stress_and_grad(
            embedding.ravel(), problem, self.mixing_ratio
        )[0]
        self.n_iter_ = self._n_iter_total_

        if self.verbose:
            print(f"Final stress: {self.stress_:.6f}")

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
