import numbers
import warnings

import numpy as np
from numpy.linalg import LinAlgError
from scipy import linalg
from scipy.linalg import sqrtm as MatrixSqrt
from scipy.sparse.linalg import svds
from sklearn.decomposition._base import _BasePCA
from sklearn.decomposition._pca import _infer_dimension
from sklearn.linear_model._base import LinearModel
from sklearn.utils import check_random_state
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils.extmath import randomized_svd, stable_cumsum, svd_flip
from sklearn.utils.validation import check_is_fitted

from skmatter.utils import pcovr_covariance, pcovr_kernel


class _BasePCov(_BasePCA, LinearModel):
    def __init__(
        self,
        mixing=0.5,
        n_components=None,
        svd_solver="auto",
        tol=1e-12,
        space="auto",
        iterated_power="auto",
        random_state=None,
        whiten=False,
    ):
        self.mixing = mixing
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.tol = tol
        self.space = space
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.whiten = whiten

    def fit(self, X):
        """Contains the common functionality for the PCovR and PCovC fit methods,
        but leaves the rest of the functionality to the subclass.
        """
        # saved for inverse transformations from the latent space,
        # should be zero in the case that the features have been properly centered
        self.mean_ = np.mean(X, axis=0)

        if np.max(np.abs(self.mean_)) > self.tol:
            warnings.warn(
                "This class does not automatically center data, and your data mean is"
                " greater than the supplied tolerance.",
                stacklevel=1,
            )

        if self.space is not None and self.space not in [
            "feature",
            "sample",
            "auto",
        ]:
            raise ValueError("Only feature and sample space are supported.")

        # Handle self.n_components==None
        if self.n_components is None:
            if self.svd_solver != "arpack":
                self.n_components_ = min(X.shape)
            else:
                self.n_components_ = min(X.shape) - 1
        else:
            self.n_components_ = self.n_components

        # Handle svd_solver
        self.fit_svd_solver_ = self.svd_solver
        if self.fit_svd_solver_ == "auto":
            # Small problem or self.n_components_ == 'mle', just call full PCA
            if max(X.shape) <= 500 or self.n_components_ == "mle":
                self.fit_svd_solver_ = "full"
            elif self.n_components_ >= 1 and self.n_components_ < 0.8 * min(X.shape):
                self.fit_svd_solver_ = "randomized"
            # This is also the case of self.n_components_ in (0,1)
            else:
                self.fit_svd_solver_ = "full"

        self.n_samples_in_, self.n_features_in_ = X.shape
        self.space_ = self.space
        if self.space_ is None or self.space_ == "auto":
            if self.n_samples_in_ > self.n_features_in_:
                self.space_ = "feature"
            else:
                self.space_ = "sample"

    def _fit_feature_space(self, X, Y, Yhat, compute_pty_=True):
        Ct, iCsqrt = pcovr_covariance(
            mixing=self.mixing,
            X=X,
            Y=Yhat,
            rcond=self.tol,
            return_isqrt=True,
        )
        try:
            Csqrt = np.linalg.lstsq(iCsqrt, np.eye(len(iCsqrt)), rcond=None)[0]

        # if we can avoid recomputing Csqrt, we should, but sometimes we
        # run into a singular matrix, which is what we do here
        except LinAlgError:
            Csqrt = np.real(MatrixSqrt(X.T @ X))

        if self.fit_svd_solver_ == "full":
            U, S, Vt = self._decompose_full(Ct)
        elif self.fit_svd_solver_ in ["arpack", "randomized"]:
            U, S, Vt = self._decompose_truncated(Ct)
        else:
            raise ValueError(f"Unrecognized svd_solver='{self.fit_svd_solver_}'")

        self.singular_values_ = np.sqrt(S.copy())
        self.explained_variance_ = S / (X.shape[0] - 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / self.explained_variance_.sum()
        )

        S_sqrt = np.diagflat([np.sqrt(s) if s > self.tol else 0.0 for s in S])
        S_sqrt_inv = np.diagflat([1.0 / np.sqrt(s) if s > self.tol else 0.0 for s in S])

        self.pxt_ = np.linalg.multi_dot([iCsqrt, Vt.T, S_sqrt])
        self.ptx_ = np.linalg.multi_dot([S_sqrt_inv, Vt, Csqrt])

        if compute_pty_:
            self.pty_ = np.linalg.multi_dot([S_sqrt_inv, Vt, iCsqrt, X.T, Y])

    def _fit_sample_space(self, X, Y, Yhat, W, compute_pty_=True):
        Kt = pcovr_kernel(mixing=self.mixing, X=X, Y=Yhat)

        if self.fit_svd_solver_ == "full":
            U, S, Vt = self._decompose_full(Kt)
        elif self.fit_svd_solver_ in ["arpack", "randomized"]:
            U, S, Vt = self._decompose_truncated(Kt)
        else:
            raise ValueError(
                "Unrecognized svd_solver='{0}'".format(self.fit_svd_solver_)
            )

        self.singular_values_ = np.sqrt(S.copy())
        self.explained_variance_ = S / (X.shape[0] - 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / self.explained_variance_.sum()
        )

        P = (self.mixing * X.T) + (1.0 - self.mixing) * W @ Yhat.T
        S_sqrt_inv = np.diagflat([1.0 / np.sqrt(s) if s > self.tol else 0.0 for s in S])
        T = Vt.T @ S_sqrt_inv

        self.pxt_ = P @ T
        self.ptx_ = T.T @ X

        if compute_pty_:
            self.pty_ = T.T @ Y

    def inverse_transform(self, T):
        if np.max(np.abs(self.mean_)) > self.tol:
            warnings.warn(
                "This class does not automatically un-center data, and your data mean "
                "is greater than the supplied tolerance, so the inverse transformation "
                "will be off by the original data mean.",
                stacklevel=1,
            )

        return T @ self.ptx_

    def transform(self, X=None):
        check_is_fitted(self, ["pxt_", "mean_"])

        return super().transform(X)

    def _decompose_truncated(self, mat):
        if not 1 <= self.n_components_ <= min(self.n_samples_in_, self.n_features_in_):
            raise ValueError(
                "n_components=%r must be between 1 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    self.n_components_,
                    min(self.n_samples_in_, self.n_features_in_),
                    self.svd_solver,
                )
            )
        elif not isinstance(self.n_components_, numbers.Integral):
            raise ValueError(
                "n_components=%r must be of type int "
                "when greater than or equal to 1, was of type=%r"
                % (self.n_components_, type(self.n_components_))
            )
        elif self.svd_solver == "arpack" and self.n_components_ == min(
            self.n_samples_in_, self.n_features_in_
        ):
            raise ValueError(
                "n_components=%r must be strictly less than "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    self.n_components_,
                    min(self.n_samples_in_, self.n_features_in_),
                    self.svd_solver,
                )
            )

        random_state = check_random_state(self.random_state)

        if self.fit_svd_solver_ == "arpack":
            v0 = _init_arpack_v0(min(mat.shape), random_state)
            U, S, Vt = svds(mat, k=self.n_components_, tol=self.tol, v0=v0)
            # svds doesn't abide by scipy.linalg.svd/randomized_svd
            # conventions, so reverse its outputs.
            S = S[::-1]
            # flip eigenvectors' sign to enforce deterministic output
            U, Vt = svd_flip(U[:, ::-1], Vt[::-1])

        # We have already eliminated all other solvers, so this must be "randomized"
        else:
            # sign flipping is done inside
            U, S, Vt = randomized_svd(
                mat,
                n_components=self.n_components_,
                n_iter=self.iterated_power,
                flip_sign=True,
                random_state=random_state,
            )

        return U, S, Vt

    def _decompose_full(self, mat):
        if self.n_components_ == "mle":
            if self.n_samples_in_ < self.n_features_in_:
                raise ValueError(
                    "n_components='mle' is only supported if n_samples >= n_features"
                )
        elif (
            not 0 <= self.n_components_ <= min(self.n_samples_in_, self.n_features_in_)
        ):
            raise ValueError(
                "n_components=%r must be between 1 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    self.n_components_,
                    min(self.n_samples_in_, self.n_features_in_),
                    self.svd_solver,
                )
            )
        elif self.n_components_ >= 1:
            if not isinstance(self.n_components_, numbers.Integral):
                raise ValueError(
                    "n_components=%r must be of type int "
                    "when greater than or equal to 1, "
                    "was of type=%r" % (self.n_components_, type(self.n_components_))
                )

        U, S, Vt = linalg.svd(mat, full_matrices=False)

        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        # Get variance explained by singular values
        explained_variance_ = S / (self.n_samples_in_ - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var

        # Postprocess the number of components required
        if self.n_components_ == "mle":
            self.n_components_ = _infer_dimension(
                explained_variance_, self.n_samples_in_
            )
        elif 0 < self.n_components_ < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            # side='right' ensures that number of features selected
            # their variance is always greater than self.n_components_ float
            # passed. More discussion in issue: #15669
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            self.n_components_ = (
                np.searchsorted(ratio_cumsum, self.n_components_, side="right") + 1
            )
        return (
            U[:, : self.n_components_],
            S[: self.n_components_],
            Vt[: self.n_components_],
        )
