"""Complete Sketch-Map implementation in Python matching PyTorch version

This implementation follows the two-stage optimization approach:
1. Pre-optimization with identity transform
2. Refinement with sigmoid transform
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import warnings


class SketchMap(BaseEstimator, TransformerMixin):
    """Sketch-Map dimensionality reducer.

    Parameters
    ----------
    n_components : int, default=2
        Dimensionality of the target embedding.

    sigma : float, default=None
        Sigmoid parameter for both high and low dimensional transforms.
        If None, estimated automatically from distance histogram.

    a_hd : float, default=2.0
        High-dimensional sigmoid exponent a.

    b_hd : float, default=6.0
        High-dimensional sigmoid exponent b.

    a_ld : float, default=2.0
        Low-dimensional sigmoid exponent a.

    b_ld : float, default=6.0
        Low-dimensional sigmoid exponent b.

    auto_histogram : bool, default=True
        If True, estimate sigma from pairwise-distance histogram.

    preopt_steps : int, default=100
        Number of pre-optimization steps with identity transform.

    opt_steps : int, default=100
        Number of optimization steps with sigmoid transform.

    global_opt_steps : int, default=0
        Number of global optimization steps (not implemented).

    mixing_ratio : float, default=0.0
        Balance between direct (1.0) and transformed (0.0) stress.

    center : bool, default=True
        If True, center the input data around origin.

    random_state : int or None
        Random seed.

    verbose : int, default=0
        Verbosity level.
    """

    def __init__(
        self,
        n_components=2,
        sigma=None,
        a_hd=2.0,
        b_hd=6.0,
        a_ld=2.0,
        b_ld=6.0,
        auto_histogram=True,
        preopt_steps=100,
        opt_steps=100,
        global_opt_steps=0,
        mixing_ratio=0.0,
        center=True,
        random_state=None,
        verbose=0,
    ):
        self.n_components = n_components
        self.sigma = sigma
        self.a_hd = a_hd
        self.b_hd = b_hd
        self.a_ld = a_ld
        self.b_ld = b_ld
        self.auto_histogram = auto_histogram
        self.preopt_steps = preopt_steps
        self.opt_steps = opt_steps
        self.global_opt_steps = global_opt_steps
        self.mixing_ratio = mixing_ratio
        self.center = center
        self.random_state = random_state
        self.verbose = verbose
        self._first_stress_call = True

    def _sigmoid_transform(self, distances, sigma, a, b):
        """Apply xsigmoid transformation: 1-(1+(2^(a/b)-1)(x/s)^a)^(-b/a)"""
        with np.errstate(divide="ignore", invalid="ignore"):
            x = distances / sigma
            A = 2 ** (a / b) - 1.0
            term = A * (x**a)
            val = 1.0 - (1.0 + term) ** (-b / a)
            val[distances <= 0.0] = 0.0
        return val

    def _sigmoid_derivative(self, distances, sigma, a, b):
        """Derivative of sigmoid with respect to distance."""
        A = 2 ** (a / b) - 1.0
        out = np.zeros_like(distances, dtype=float)
        pos = distances > 0.0

        if np.any(pos):
            r = distances[pos]
            u = A * (r / sigma) ** a
            pref = b * A * (r ** (a - 1.0)) / (sigma**a)
            out[pos] = pref * (1.0 + u) ** (-b / a - 1.0)

        return out

    def _classical_mds(self, distances):
        """Classical MDS initialization matching C++ implementation."""
        D = distances

        # Compute Gram matrix with centering as in C++
        M = -0.5 * (D**2)
        M -= M.mean(axis=0, keepdims=True)  # subtract column means
        M -= M.mean(axis=1, keepdims=True)  # subtract row means

        # Eigendecomposition
        w, V = np.linalg.eigh(M)

        # Sort descending and take top n_components
        idx = np.argsort(w)[::-1]
        top_idx = idx[: self.n_components]

        top_eigenvalues = w[top_idx]
        top_eigenvectors = V[:, top_idx]

        # Compute coordinates
        coordinates = top_eigenvectors * np.sqrt(np.maximum(top_eigenvalues, 0.0))

        # Align signs to match C++ convention: make max abs value positive
        for i in range(self.n_components):
            col = coordinates[:, i]
            max_id = np.argmax(np.abs(col))
            if col[max_id] < 0:
                coordinates[:, i] *= -1

        return coordinates

    def _estimate_sigma(self, distances, n_bins=200):
        """Estimate sigma from histogram of distances."""
        d = distances[np.triu_indices_from(distances, k=1)]
        d = d[np.isfinite(d) & (d >= 0)]

        if d.size == 0:
            raise ValueError("Empty distances array")

        hist, edges = np.histogram(d, bins=n_bins)
        peak_idx = np.argmax(hist)
        peak_val = hist[peak_idx]

        # Find where histogram drops below half peak
        half_idx = np.where(hist[peak_idx + 1 :] <= peak_val * 0.5)[0]

        if half_idx.size > 0:
            sigma = 0.5 * (
                edges[peak_idx + 1 + half_idx[0]] + edges[peak_idx + half_idx[0]]
            )
        else:
            sigma = float(np.median(d))

        return sigma

    def _compute_stress(self, X_ld, D_hd, S_hd, W, mixing_ratio, use_transform=True):
        """Compute sketch-map stress function.

        Parameters
        ----------
        X_ld : array, shape (n*d,) or (n, d)
            Low-dimensional coordinates (flattened or matrix)
        D_hd : array, shape (n, n)
            High-dimensional distances
        S_hd : array, shape (n, n)
            Transformed high-dimensional distances
        W : array, shape (n, n)
            Weight matrix
        mixing_ratio : float
            Balance between direct and transformed stress
        use_transform : bool
            Whether to apply sigmoid transform to low-dim distances
        """
        # Handle both flat and matrix inputs
        if X_ld.ndim == 1:
            n = D_hd.shape[0]
            X_ld_mat = X_ld.reshape((n, self.n_components))
        else:
            X_ld_mat = X_ld

        # Low-dimensional distances
        D_ld = squareform(pdist(X_ld_mat, metric="euclidean"))

        # Transform low-dimensional distances if requested
        if use_transform:
            S_ld = self._sigmoid_transform(D_ld, self.sigma_, self.a_ld, self.b_ld)
        else:
            S_ld = D_ld

        # Compute stress components
        direct_stress = np.sum(W * (D_hd - D_ld) ** 2)
        transformed_stress = np.sum(W * (S_hd - S_ld) ** 2)

        combined_stress = (
            mixing_ratio * direct_stress + (1 - mixing_ratio) * transformed_stress
        )

        # Debug logging on first call
        if self.verbose > 1 and self._first_stress_call:
            print("\n=== First Stress Calculation Debug ===")
            print(f"High-dim distances (sample 3x3):\n{D_hd[:3, :3]}")
            print(f"\nLow-dim distances (sample 3x3):\n{D_ld[:3, :3]}")
            print(f"\nTransformed high-dim (sample 3x3):\n{S_hd[:3, :3]}")
            print(f"Transformed low-dim (sample 3x3):\n{S_ld[:3, :3]}")
            print(f"\nWeights (sample 3x3):\n{W[:3, :3]}")
            print(f"Weight sum: {np.sum(W):.4f}")
            print("\nStress components:")
            print(f"  Direct stress: {direct_stress:.4f}")
            print(f"  Transformed stress: {transformed_stress:.4f}")
            print(f"  Combined stress: {combined_stress:.4f}")
            print(f"  Mixing ratio: {mixing_ratio:.2f}")
            print(f"  Normalized stress: {combined_stress / np.sum(W):.6f}")
            print("=" * 40 + "\n")
            self._first_stress_call = False

        # Normalize by sum of weights
        return combined_stress / np.sum(W)

    def _compute_gradient(self, X_ld, D_hd, S_hd, W, mixing_ratio, use_transform=True):
        """Compute gradient of stress function."""
        # Handle both flat and matrix inputs
        if X_ld.ndim == 1:
            n = D_hd.shape[0]
            X_ld_mat = X_ld.reshape((n, self.n_components))
        else:
            X_ld_mat = X_ld

        # Pairwise differences
        dif = X_ld_mat[:, None, :] - X_ld_mat[None, :, :]
        D_ld = np.sqrt(np.sum(dif**2, axis=2))

        # Transform and derivatives
        if use_transform:
            S_ld = self._sigmoid_transform(D_ld, self.sigma_, self.a_ld, self.b_ld)
            dS_dD = self._sigmoid_derivative(D_ld, self.sigma_, self.a_ld, self.b_ld)
        else:
            S_ld = D_ld
            dS_dD = np.ones_like(D_ld)

        # Avoid division by zero
        eps = np.finfo(float).eps
        inv_D = 1.0 / (D_ld + eps)

        # Gradient matrix
        M = (
            W
            * (
                mixing_ratio * (D_ld - D_hd)
                + (1 - mixing_ratio) * (S_ld - S_hd) * dS_dD
            )
            * inv_D
        )
        np.fill_diagonal(M, 0.0)

        row_sums = M.sum(axis=1)
        G = (row_sums[:, None] * X_ld_mat) - (M @ X_ld_mat)

        # Normalize by sum of weights and apply factor of 2
        G = 2.0 * G / np.sum(W)

        return G.ravel()

    def _optimize(
        self, X_init, D_hd, S_hd, W, mixing_ratio, n_steps, use_transform=True
    ):
        """Run L-BFGS-B optimization."""
        n = X_init.shape[0]
        x0 = X_init.ravel()

        def objective(x):
            return self._compute_stress(x, D_hd, S_hd, W, mixing_ratio, use_transform)

        def gradient(x):
            return self._compute_gradient(x, D_hd, S_hd, W, mixing_ratio, use_transform)

        # Optimize
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            jac=gradient,
            options={"maxiter": n_steps, "disp": self.verbose > 1},
        )

        return result.x.reshape((n, self.n_components)), result.fun

    def fit(self, X, y=None, sample_weights=None):
        """Fit the Sketch-Map model following PyTorch two-stage approach.

        Stage 1: Pre-optimization with identity transform
        Stage 2: Refinement with sigmoid transform

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency.

        sample_weights : array-like, shape (n_samples,), optional
            Per-sample weights.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D array")

        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features

        # Use weights from fit() call if provided
        weights_to_use = (
            sample_weights if sample_weights is not None else self.sample_weights
        )

        if self.verbose:
            print(f"Fitting Sketch-Map with {n_samples} samples, {n_features} features")
            if weights_to_use is not None:
                print(
                    f"Using sample weights: min={np.min(weights_to_use):.4f}, "
                    f"max={np.max(weights_to_use):.4f}, "
                    f"mean={np.mean(weights_to_use):.4f}"
                )

        # Center data if requested
        if self.center:
            if self.verbose:
                print("Centering the data")
            X = X - X.mean(axis=0, keepdims=True)

        # Compute pairwise distances
        if self.verbose:
            print("Computing pairwise distances...")
        D_hd = squareform(pdist(X, metric="euclidean"))

        # Estimate or set sigma
        if self.auto_histogram and self.sigma is None:
            self.sigma_ = self._estimate_sigma(D_hd)
            if self.verbose:
                print(f"Estimated sigma: {self.sigma_:.4f}")
        else:
            self.sigma_ = self.sigma if self.sigma is not None else 1.0

        # Transform high-dimensional distances (for stage 2)
        S_hd = self._sigmoid_transform(D_hd, self.sigma_, self.a_hd, self.b_hd)

        # Setup weight matrix: W_ij = w_i * w_j
        if weights_to_use is not None:
            w = np.asarray(weights_to_use)
            if w.shape[0] != n_samples:
                raise ValueError(
                    f"sample_weights must have length {n_samples}, got {w.shape[0]}"
                )
            W = np.outer(w, w)
            if self.verbose:
                print(f"Weight matrix: sum={np.sum(W):.4f}, shape={W.shape}")
        else:
            W = np.ones_like(D_hd)
            if self.verbose:
                print("Using uniform weights")

        # Reset debug flag
        self._first_stress_call = True

        # Initialize with classical MDS
        if self.verbose:
            print("Initializing with classical MDS...")
        X_ld = self._classical_mds(D_hd)

        # Stage 1: Pre-optimization with identity transform
        if self.preopt_steps > 0:
            if self.verbose:
                print("\n=== Stage 1: Pre-optimization with identity transform ===")
                print(f"Running {self.preopt_steps} optimization steps...")

            X_ld, stress = self._optimize(
                X_ld,
                D_hd,
                D_hd,
                W,  # Use D_hd for both (identity)
                self.mixing_ratio,
                self.preopt_steps,
                use_transform=False,
            )

            if self.verbose:
                print(f"Pre-optimization stress: {stress:.6f}")

        # Stage 2: Refinement with sigmoid transform
        if self.opt_steps > 0:
            if self.verbose:
                print("\n=== Stage 2: Refinement with sigmoid transform ===")
                print(f"Running {self.opt_steps} optimization steps...")

            # Reset debug flag for second stage
            self._first_stress_call = True

            X_ld, stress = self._optimize(
                X_ld,
                D_hd,
                S_hd,
                W,
                self.mixing_ratio,
                self.opt_steps,
                use_transform=True,
            )

            if self.verbose:
                print(f"Final stress: {stress:.6f}")

        # Note: Global optimization not implemented
        if self.global_opt_steps > 0:
            warnings.warn("Global optimization not yet implemented, skipping")

        self.embedding_ = X_ld
        self.stress_ = stress

        if self.verbose:
            print("\nSketch-Map fitting complete!")

        return self

    def fit_transform(self, X, y=None, sample_weights=None):
        """Fit the model and return the embedding.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency.

        sample_weights : array-like, shape (n_samples,), optional
            Per-sample weights.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedded coordinates.
        """
        self.fit(X, y, sample_weights=sample_weights)
        return self.embedding_

    def transform(self, X):
        """Project new data (not implemented for out-of-sample)."""
        raise NotImplementedError("Out-of-sample projection not yet implemented")
