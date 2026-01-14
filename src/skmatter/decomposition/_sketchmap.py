"""SketchMap dimensionality reduction (scikit-learn style skeleton)

This module provides a scikit-learn style estimator for the Sketch-map
non-linear dimensionality reduction algorithm. The implementation below
focuses on a clear public API and helper function signatures informed by
the Sketch-map tutorial (histogram analysis and sigmoid parameter
selection). Numerical optimization and advanced options are left as
implementation tasks to follow after API review.

References
----------
- https://sketchmap.org/index.html?page=tuts&psub=analysis
"""

from __future__ import annotations

from typing import Optional, Callable, Dict
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform, cdist


class _NLDRFunction:
    """Transfer function used by Sketch-map (high/low dim).

    Supports 'identity' and 'xsigmoid' (the s_{sigma,a,b} from tutorial).
    """

    def __init__(self, mode="identity", pars=None):
        self.mode = mode
        self.pars = pars if pars is not None else {}

    def f(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "identity":
            return x
        if self.mode == "xsigmoid":
            # pars: sigma, a, b
            sigma = float(self.pars.get("sigma", 1.0))
            a = float(self.pars.get("a", 2.0))
            b = float(self.pars.get("b", 6.0))
            x = np.asarray(x, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                term = (2 ** (a / b) - 1.0) * (x / sigma) ** a
                val = 1.0 - (1.0 + term) ** (-b / a)
                val[x <= 0.0] = 0.0
            return val
        raise ValueError(f"Unknown NLDRFunction mode: {self.mode}")

    def df(self, x: np.ndarray) -> np.ndarray:
        """Derivative ds/dr of transfer function with respect to distance r.

        Returns an array shaped like `x`.
        """
        x = np.asarray(x, dtype=float)
        if self.mode == "identity":
            return np.ones_like(x)
        if self.mode == "xsigmoid":
            sigma = float(self.pars.get("sigma", 1.0))
            a = float(self.pars.get("a", 2.0))
            b = float(self.pars.get("b", 6.0))
            A = 2 ** (a / b) - 1.0
            # handle zero distances safely
            out = np.zeros_like(x, dtype=float)
            pos = x > 0.0
            r = x[pos]
            if r.size > 0:
                u = A * (r / sigma) ** a
                # ds/dr = b * A * r^{a-1} / sigma^a * (1+u)^{-b/a -1}
                pref = b * A * (r ** (a - 1.0)) / (sigma**a)
                out[pos] = pref * (1.0 + u) ** (-b / a - 1.0)
            return out
        raise ValueError(f"Unknown NLDRFunction mode: {self.mode}")


def _classical_mds(distances: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Classical MDS (metric MDS) initialization from a distance matrix.

    distances: square (n,n) array of pairwise Euclidean distances.
    Returns coordinates of shape (n, n_components).
    """
    D = np.asarray(distances, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distances must be square matrix")
    n = D.shape[0]
    # double center
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J.dot(D**2).dot(J)
    # eigendecomposition
    w, V = np.linalg.eigh(B)
    # sort descending
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    pos = V[:, :n_components] * np.sqrt(np.maximum(w[:n_components], 0.0))
    return pos


def _estimate_histogram_params(
    distances: np.ndarray, n_bins: int = 200
) -> Dict[str, float]:
    """Estimate sketch-map histogram-based parameters (sigma, a, b).

    This is a lightweight helper to analyse the distribution of pairwise
    distances and return a suggested `sigma`, `a` and `b` following the
    heuristics described in the Sketch-map tutorial: choose sigma in the
    intermediate-distance region (between Gaussian-basin short-range and
    high-dimensional dominated long-range), and a/b to control the
    sharpness of the sigmoid.

    The function currently provides a conservative automatic guess and
    should be refined later to reproduce the `dimdist`/interactive
    selection used in the original implementation.
    """
    # simple placeholder: pick sigma at the distance where the histogram
    # first drops below half the peak after the main mode. This is a
    # heuristic; a more robust implementation will follow.
    d = distances.ravel()
    d = d[np.isfinite(d) & (d >= 0)]
    if d.size == 0:
        raise ValueError("Empty distances array")
    hist, edges = np.histogram(d, bins=n_bins)
    peak_idx = np.argmax(hist)
    peak_val = hist[peak_idx]
    half_idx = np.where(hist[peak_idx + 1 :] <= peak_val * 0.5)[0]
    if half_idx.size > 0:
        sigma = 0.5 * (
            edges[peak_idx + 1 + half_idx[0]] + edges[peak_idx + half_idx[0]]
        )
    else:
        sigma = float(np.median(d))

    # default exponents: use conservative values (can be tuned by user)
    a = 2.0
    b = 6.0
    return {"sigma": float(sigma), "a": float(a), "b": float(b)}


class SketchMap(BaseEstimator, TransformerMixin):
    """SketchMap dimensionality reducer (API skeleton).

    Parameters
    ----------
    n_components : int, default=2
        Dimensionality of the target embedding.

    n_landmarks : Optional[int]
        If set, use FPS or another selector to pick a subset of landmarks for
        optimization. If ``None`` use all points (may be slow).

    fps_selector : object or None
        Optional selector object implementing `fit(X)` and exposing
        `selected_idx_`. If provided, it is used to select landmarks instead
        of the internal `n_landmarks` heuristic.

    auto_histogram : bool, default=True
        If True, estimate `sigma`, `a`, `b` automatically from the
        pairwise-distance histogram following the Sketch-map tutorial.

    fun_hd : str or callable, default="xsigmoid"
        High-dimensional transfer function. Supported standard names: "identity",
        "gamma", "xsigmoid", or a callable that accepts a distance and
        returns a transformed value.

    fun_ld : str or callable, default="xsigmoid"
        Low-dimensional transfer function (same conventions as ``fun_hd``).

    preopt_steps : int, default=0
        Number of iterative pre-optimization steps (conjugate gradient) to run
        after MDS initialization.

    random_state : int or None
        Random seed for initialization.

    verbose : int, default=0
        Verbosity level.

    Notes
    -----
    This class currently implements the public API and lightweight helpers
    informed by the Sketch-map tutorial. The numerical optimization core
    (stress function, gradient, iterative minimizer) will be implemented
    after the API is confirmed.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_landmarks: Optional[int] = None,
        auto_histogram: bool = True,
        fun_hd: object = "xsigmoid",
        fun_ld: object = "xsigmoid",
        preopt_steps: int = 0,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        self.n_components = n_components
        self.n_landmarks = n_landmarks
        self.auto_histogram = auto_histogram
        self.fun_hd = fun_hd
        self.fun_ld = fun_ld
        self.preopt_steps = int(preopt_steps)
        self.random_state = random_state
        self.verbose = int(verbose)

        # attributes set during fit
        self.landmarks_ = None
        self.X_landmarks_ = None
        self.embedding_ = None
        self.params_ = None

    def fit(
        self,
        X,
        y=None,
        sample_weights=None,
        n_components=2,
        mixing_ratio: float = 0.0,
        local_opt_num_steps: Optional[int] = None,
        global_opt_num_steps: int = 0,
        learning_rate: float = 1.0,
    ):
        """Fit the Sketch-map model to the data X.

        Steps (high-level):

        1. validate and store X
        2. use all points as landmarks (callers may pre-select subsets)
        3. compute pairwise distances on landmarks
        4. if ``auto_histogram`` estimate `sigma`, `a`, `b` using
           `_estimate_histogram_params`
        5. initialize low-dim coordinates (classical MDS or provided init)
        6. run optimization (preopt / global) to minimize sketch-map stress

        The implementation supports a mixing ratio between direct-distance
        stress and transformed (sigmoid) stress, and accepts optional
        `sample_weights` as a per-sample 1D weight vector.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_features)")

        self._n_samples, self._n_features = X.shape
        self.X_ = X.copy()

        # honor n_components passed to fit (overrides estimator attribute)
        self.n_components = int(n_components)

        # For the current API we use all points as landmarks. Do not
        # attempt any FPS/sample-selection fallback here; callers that want
        # a subset should pre-select and pass only those points to `fit`.
        self.landmarks_ = np.arange(X.shape[0])
        self.X_landmarks_ = X[self.landmarks_]

        # store sample_weights if provided
        if sample_weights is not None:
            sw = np.asarray(sample_weights)
            if sw.shape[0] != self._n_samples:
                raise ValueError("sample_weights must have length n_samples")
            self.sample_weights_ = sw
        else:
            self.sample_weights_ = None

        # compute pairwise distances on landmarks (square matrix)
        # user can later replace with metric or periodic options
        dists = squareform(pdist(self.X_landmarks_, metric="euclidean"))

        if self.auto_histogram:
            self.params_ = _estimate_histogram_params(dists)
        else:
            self.params_ = {"sigma": None, "a": None, "b": None}

        # build transfer functions
        hd_pars = {k: self.params_.get(k) for k in ("sigma", "a", "b")}
        self._tfun_hd = (
            _NLDRFunction("xsigmoid", hd_pars)
            if isinstance(self.fun_hd, str) and self.fun_hd == "xsigmoid"
            else _NLDRFunction("identity", {})
        )
        self._tfun_ld = (
            _NLDRFunction("xsigmoid", hd_pars)
            if isinstance(self.fun_ld, str) and self.fun_ld == "xsigmoid"
            else _NLDRFunction("identity", {})
        )

        # initialize low-dim coordinates using classical MDS on the landmark
        init_pos = _classical_mds(dists, n_components=self.n_components)

        # build pairwise weight matrix from sample_weights (1D per-sample)
        if self.sample_weights_ is not None:
            wvec = np.asarray(self.sample_weights_, dtype=float)
            # restrict to landmarks only
            w = wvec[self.landmarks_]
            W = np.outer(w, w)
        else:
            W = np.ones_like(dists, dtype=float)

        # optimization: minimize combined direct+transformed stress over landmark low-d
        # coords
        s_hd = self._tfun_hd.f(dists)

        nL = self.X_landmarks_.shape[0]
        x0 = init_pos.ravel()

        def _objective(xvec):
            Xld = xvec.reshape((nL, self.n_components))
            Dld = squareform(pdist(Xld, metric="euclidean"))
            s_ld = self._tfun_ld.f(Dld)
            # direct and transformed differences
            diff_direct = dists - Dld
            diff_trans = s_hd - s_ld
            m = float(mixing_ratio)
            # combined stress over all pairs, normalized by total weight
            tot = np.sum(W)
            val = 0.5 * np.sum(W * (m * (diff_direct**2) + (1.0 - m) * (diff_trans**2)))
            return float(val / (tot if tot != 0 else 1.0))

        def _jac(xvec):
            Xld = xvec.reshape((nL, self.n_components))
            # pairwise displacement tensor (nL,nL,ncomp): X_i - X_j
            dif = Xld[:, None, :] - Xld[None, :, :]
            Dld = np.sqrt(np.sum(dif**2, axis=2))
            # transformed distances and derivatives
            s_ld = self._tfun_ld.f(Dld)
            ds_dr = self._tfun_ld.df(Dld)
            m = float(mixing_ratio)
            # avoid division by zero
            eps = np.finfo(float).eps
            inv_r = 1.0 / (Dld + eps)
            # combined matrix M_{ij} = W_ij * [ m*(Dld - Dhd) + (1-m)*(s_ld -
            # s_hd)*ds_dr ] / r
            M = W * (m * (Dld - dists) + (1.0 - m) * (s_ld - s_hd) * ds_dr) * inv_r
            np.fill_diagonal(M, 0.0)
            row_sums = M.sum(axis=1)
            G = (row_sums[:, None] * Xld) - (M @ Xld)
            # normalize gradient consistent with objective normalization
            tot = np.sum(W)
            if tot != 0:
                G = G / float(tot)
            return G.ravel()

        # Optional pre-optimization stage (mimics `-preopt` in the CLI)
        rng = np.random.RandomState(self.random_state)
        x_cur = x0.copy()
        if self.preopt_steps and int(self.preopt_steps) > 0:
            pre_res = minimize(
                _objective,
                x_cur,
                method="L-BFGS-B",
                jac=_jac,
                options={"maxiter": int(self.preopt_steps)},
            )
            if pre_res.success:
                x_cur = pre_res.x

        # Lightweight iterative 'mixing' inspired by sketch-map.sh: perform
        # a few short global restarts with small jitter and keep the best
        # solution. This is a cheap approximation of grid/global opt.
        best_res = None
        IMIX = 1.0
        MAX_MIX = 4
        for it in range(1, MAX_MIX + 1):
            # run a short local optimization
            res = minimize(
                _objective, x_cur, method="L-BFGS-B", jac=_jac, options={"maxiter": 250}
            )
            if best_res is None or res.fun < best_res.fun:
                best_res = res
            # if no improvement, try a global jittered restart
            if it < MAX_MIX:
                # jitter amplitude scales with IMIX and median pairwise HD distance
                med = np.median(dists)
                amp = IMIX * 0.01 * (med if med > 0 else 1.0)
                jitter = rng.normal(scale=amp, size=x_cur.shape)
                x_cur = best_res.x + jitter
                # update IMIX conservatively
                IMIX = float(max(0.1, min(0.5, IMIX * 0.9)))

        # final, longer optimization starting from best found
        res = minimize(
            _objective,
            best_res.x if best_res is not None else x_cur,
            method="L-BFGS-B",
            jac=_jac,
            options={"maxiter": 1000},
        )
        Xopt = res.x.reshape((nL, self.n_components))

        # build full embedding: assign nearest landmark embedding to non-landmarks
        embedding = np.zeros((self._n_samples, self.n_components))
        embedding[self.landmarks_] = Xopt
        if self.landmarks_.size < self._n_samples:
            # find nearest landmark in HD space
            full_to_land = cdist(self.X_, self.X_landmarks_, metric="euclidean")
            nn = np.argmin(full_to_land, axis=1)
            for i in range(self._n_samples):
                embedding[i] = Xopt[nn[i]]

        self.embedding_ = embedding

        # store diagnostics
        self.n_iter_ = int(getattr(res, "nit", 0))
        self.stress_ = float(res.fun)

        return self

    def transform(self, X):
        """Project new samples into the fitted low-dimensional space.

        Out-of-sample extension not implemented in this skeleton.
        """
        if self.embedding_ is None:
            raise ValueError("SketchMap instance is not fitted yet")
        raise NotImplementedError("Out-of-sample projection is not yet implemented")

    def predict(self, X):
        """Predict method not implemented for dimensionality reducers."""
        raise NotImplementedError("SketchMap is a transformer, not a predictor")

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y=y, **fit_params)
        return self.embedding_

    def score(self, X, y=None):
        """Return a simple negative-stress score if available.

        Higher is better; this is a convenience wrapper. If `stress_` is not
        available the method raises `NotImplementedError`.
        """
        if not hasattr(self, "stress_") or np.isnan(self.stress_):
            raise NotImplementedError(
                "Stress-based score not available until optimization is implemented"
            )
        return -float(self.stress_)
