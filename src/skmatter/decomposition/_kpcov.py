import numbers
import numpy as np

from abc import ABCMeta, abstractmethod
from scipy import linalg
from scipy.sparse.linalg import svds
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError

from sklearn.decomposition._base import _BasePCA
from sklearn.linear_model._base import LinearModel
from sklearn.decomposition._pca import _infer_dimension
from sklearn.utils import check_random_state
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils.extmath import randomized_svd, stable_cumsum, svd_flip
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from sklearn.metrics.pairwise import pairwise_kernels

import numpy as np
from numpy.linalg import LinAlgError
import scipy.sparse as sp
from scipy import linalg
from scipy.linalg import sqrtm as MatrixSqrt
from scipy.sparse.linalg import svds

from skmatter.utils import pcovr_kernel, pcovr_covariance


class _BaseKPCov(_BasePCA, LinearModel, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        mixing=0.5,
        n_components=None,
        svd_solver="auto",
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        center=False,
        fit_inverse_transform=False,
        tol=1e-12,
        n_jobs=None,
        iterated_power="auto",
        random_state=None,
    ):
        self.mixing = mixing
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.center = center
        self.fit_inverse_transform = fit_inverse_transform
        self.tol = tol
        self.n_jobs = n_jobs
        self.iterated_power = iterated_power
        self.random_state = random_state

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {
                "gamma": getattr(self, "gamma_", self.gamma),
                "degree": self.degree,
                "coef0": self.coef0,
            }

        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params
        )

    @abstractmethod
    def fit(self, X):
        """This contains the common functionality for KPCovR and KPCovC fit methods,
        but leaves the rest of the fit functionality to the subclass.
        """
        self.X_fit_ = X.copy()

        if self.n_components is None:
            if self.svd_solver != "arpack":
                self.n_components_ = X.shape[0]
            else:
                self.n_components_ = X.shape[0] - 1
        else:
            self.n_components_ = self.n_components

        self.n_samples_in_, self.n_features_in_ = X.shape

        # Handle svd_solver
        self.fit_svd_solver_ = self.svd_solver
        if self.fit_svd_solver_ == "auto":
            # Small problem or self.n_components_ == 'mle', just call full PCA
            if (
                max(self.n_samples_in_, self.n_features_in_) <= 500
                or self.n_components_ == "mle"
            ):
                self.fit_svd_solver_ = "full"
            elif self.n_components_ >= 1 and self.n_components_ < 0.8 * max(
                self.n_samples_in_, self.n_features_in_
            ):
                self.fit_svd_solver_ = "randomized"
            # This is also the case of self.n_components_ in (0,1)
            else:
                self.fit_svd_solver_ = "full"

    def _fit_covariance(self, K, Z):
        """
        Fit the model with the computed kernel and approximated properties. Uses Covariance Matrix
        """
        print(f"KPCovC K: {K[:5, 0]}")
        Ct, iCsqrt = pcovr_covariance(
            mixing=self.mixing,
            X=K,
            Y=Z,
            rcond=self.tol,
            return_isqrt=True,
        )
        try:
            Csqrt = np.linalg.lstsq(iCsqrt, np.eye(len(iCsqrt)), rcond=None)[0]

        # if we can avoid recomputing Csqrt, we should, but sometimes we
        # run into a singular matrix, which is what we do here
        except LinAlgError:
            Csqrt = np.real(MatrixSqrt(K.T @ K))

        if self.fit_svd_solver_ == "full":
            U, S, Vt = self._decompose_full(Ct)
        elif self.fit_svd_solver_ in ["arpack", "randomized"]:
            U, S, Vt = self._decompose_truncated(Ct)
        else:
            raise ValueError(f"Unrecognized svd_solver='{self.fit_svd_solver_}'")

        S_sqrt = np.diagflat([np.sqrt(s) if s > self.tol else 0.0 for s in S])
        S_sqrt_inv = np.diagflat([1.0 / np.sqrt(s) if s > self.tol else 0.0 for s in S])

        self.pkt_ = np.linalg.multi_dot([iCsqrt, Vt.T, S_sqrt])
        self.ptk_ = np.linalg.multi_dot([S_sqrt_inv, Vt, Csqrt])

        # if self.mixing == 1.0:
        #     lambda_i = np.sqrt(S)
        #     self.pkt_ = self.pkt_ / np.sqrt(lambda_i)[np.newaxis, :]

        T = K @ self.pkt_
        self.pt__ = np.linalg.lstsq(T, np.eye(T.shape[0]), rcond=self.tol)[0]

    def _fit_gram(self, K, Yhat, W):
        """
        Fit the model with the computed kernel and approximated properties.
        """
        K_tilde = pcovr_kernel(mixing=self.mixing, X=K, Y=Yhat, kernel="precomputed")

        print("KPCovC K: " + str(K[:5, 0]))
        print("KPCovC Yhat: " + str(Yhat[:5, 0]))
        print("KPCovC K_tilde: " + str(K_tilde[:5, 0]))

        if self.fit_svd_solver_ == "full":
            _, S, Vt = self._decompose_full(K_tilde)
        elif self.fit_svd_solver_ in ["arpack", "randomized"]:
            _, S, Vt = self._decompose_truncated(K_tilde)
        else:
            raise ValueError(
                "Unrecognized svd_solver='{0}'" "".format(self.fit_svd_solver_)
            )

        U = Vt.T

        P = (self.mixing * np.eye(K.shape[0])) + (1.0 - self.mixing) * (W @ Yhat.T)
        print("KPCovC P: " + str(P[:5, 0]))

        S_inv = np.array([1.0 / s if s > self.tol else 0.0 for s in S])

        self.pkt_ = P @ U @ np.sqrt(np.diagflat(S_inv))

        T = K @ self.pkt_
        self.pt__ = np.linalg.lstsq(T, np.eye(T.shape[0]), rcond=self.tol)[0]
        # np.linalg.lstsq(K @ self.pkt_, np.eye(K @ self.pkt_.shape[0]), rcond=self.tol)[0]
        # self.ptx = self.pt__ @ X

    def transform(self, X=None):
        check_is_fitted(self, ["pkt_", "X_fit_"])

        X = validate_data(self, X, reset=False)
        K = self._get_kernel(X, self.X_fit_)

        if self.center:
            K = self.centerer_.transform(K)

        return K @ self.pkt_

    def inverse_transform(self, T):
        if not self.fit_inverse_transform:
            raise NotFittedError(
                "The fit_inverse_transform parameter was not"
                " set to True when instantiating and hence "
                "the inverse transform is not available."
            )

        return T @ self.ptx_

    def _decompose_truncated(self, mat):
        if not 1 <= self.n_components_ <= self.n_samples_in_:
            raise ValueError(
                "n_components=%r must be between 1 and "
                "n_samples=%r with "
                "svd_solver='%s'"
                % (
                    self.n_components_,
                    self.n_samples_in_,
                    self.svd_solver,
                )
            )
        elif not isinstance(self.n_components_, numbers.Integral):
            raise ValueError(
                "n_components=%r must be of type int "
                "when greater than or equal to 1, was of type=%r"
                % (self.n_components_, type(self.n_components_))
            )
        elif self.svd_solver == "arpack" and self.n_components_ == self.n_samples_in_:
            raise ValueError(
                "n_components=%r must be strictly less than "
                "n_samples=%r with "
                "svd_solver='%s'"
                % (
                    self.n_components_,
                    self.n_samples_in_,
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

        U[:, S < self.tol] = 0.0
        Vt[S < self.tol] = 0.0
        S[S < self.tol] = 0.0

        return U, S, Vt

    def _decompose_full(self, mat):
        if self.n_components_ != "mle":
            if not (0 <= self.n_components_ <= self.n_samples_in_):
                raise ValueError(
                    "n_components=%r must be between 1 and "
                    "n_samples=%r with "
                    "svd_solver='%s'"
                    % (
                        self.n_components_,
                        self.n_samples_in_,
                        self.svd_solver,
                    )
                )
            elif self.n_components_ >= 1:
                if not isinstance(self.n_components_, numbers.Integral):
                    raise ValueError(
                        "n_components=%r must be of type int "
                        "when greater than or equal to 1, "
                        "was of type=%r"
                        % (self.n_components_, type(self.n_components_))
                    )

        U, S, Vt = linalg.svd(mat, full_matrices=False)
        U[:, S < self.tol] = 0.0
        Vt[S < self.tol] = 0.0
        S[S < self.tol] = 0.0

        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        # Get variance explained by singular values
        explained_variance_ = (S**2) / (self.n_samples_in_ - 1)
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
