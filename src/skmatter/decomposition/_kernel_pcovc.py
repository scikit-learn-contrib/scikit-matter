import numpy as np
import numbers

from scipy import linalg
import scipy.sparse as sp
from sklearn import clone
from sklearn.metrics import accuracy_score
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import LabelBinarizer
from sklearn.linear_model import (
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV,
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier,
)
from sklearn.calibration import column_or_1d
from sklearn.utils import check_array, check_random_state, column_or_1d
from sklearn.utils.validation import check_is_fitted, validate_data
from scipy.sparse.linalg import svds
from sklearn.decomposition._pca import _infer_dimension
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils.extmath import randomized_svd, stable_cumsum, svd_flip

from skmatter.utils import check_cl_fit, pcovr_kernel
from skmatter.utils import pcovr_kernel
from sklearn.utils._array_api import get_namespace

from skmatter.preprocessing import KernelNormalizer
from skmatter.decomposition import PCovC


class KernelPCovC(PCovC):
    def __init__(
        self,
        mixing=0.5,
        n_components=None,
        svd_solver="auto",
        tol=1e-12,
        space="auto",
        classifier=None,
        iterated_power="auto",
        random_state=None,
        kernel="rbf",
        gamma="scale",
        degree=3,
        coef0=0,
        kernel_params=None,
        center=True,  # False in KPCovR, but getting error:
        # "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT" sometimes
        # when training due to unscaled X
        fit_inverse_transform=False,
        n_jobs=None,
    ):
        super().__init__(
            mixing=mixing,
            n_components=n_components,
            svd_solver=svd_solver,
            tol=tol,
            space=space,
            classifier=classifier,
            iterated_power=iterated_power,
            random_state=random_state,
        )
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.center = center
        self.fit_inverse_transform = fit_inverse_transform
        self.n_jobs = n_jobs

    def _get_kernel(self, X, Y=None):
        sparse = sp.issparse(X)

        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            # from BaseSVC:
            if self.gamma == "scale":
                X_var = (X.multiply(X)).mean() - (X.mean()) ** 2 if sparse else X.var()
                self.gamma_ = 1.0 / (X.shape[1] * X_var) if X_var != 0 else 1.0
            elif self.gamma == "auto":
                self.gamma_ = 1.0 / X.shape[1]
            else:
                self.gamma_ = self.gamma
            params = {"gamma": self.gamma_, "degree": self.degree, "coef0": self.coef0}

        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params
        )

    def _fit(self, K, Z, W):
        """
        Fit the model with the computed kernel and approximated properties.
        """

        K_tilde = pcovr_kernel(mixing=self.mixing, X=K, Y=Z, kernel="precomputed")

        if self.fit_svd_solver_ == "full":
            _, S, Vt = self._decompose_full(K_tilde)
        elif self.fit_svd_solver_ in ["arpack", "randomized"]:
            _, S, Vt = self._decompose_truncated(K_tilde)
        else:
            raise ValueError(
                "Unrecognized svd_solver='{0}'" "".format(self.fit_svd_solver_)
            )

        U = Vt.T

        P = (self.mixing * np.eye(K.shape[0])) + (1.0 - self.mixing) * (W @ Z.T)
        # print("P: " +str(P.shape))
        # print("U: " + str(U.shape))

        S_inv = np.array([1.0 / s if s > self.tol else 0.0 for s in S])

        self.pkt_ = P @ U @ np.sqrt(np.diagflat(S_inv))
        # print("Pkt: "+str(self.pkt_.shape))
        T = K @ self.pkt_
        self.pt__ = np.linalg.lstsq(T, np.eye(T.shape[0]), rcond=self.tol)[0]

    def fit(self, X, y, W=None):
        X, y = validate_data(self, X, y, multi_output=True)
        self.X_fit_ = X.copy()

        if self.n_components is None:
            if self.svd_solver != "arpack":
                self.n_components_ = X.shape[0]
            else:
                self.n_components_ = X.shape[0] - 1
        else:
            self.n_components_ = self.n_components

        K = self._get_kernel(X)

        if self.center:
            self.centerer_ = KernelNormalizer()
            K = self.centerer_.fit_transform(K)

        self.n_samples_in_, self.n_features_in_ = X.shape

        compatible_classifiers = (
            LinearDiscriminantAnalysis,
            LinearSVC,
            LogisticRegression,
            LogisticRegressionCV,
            MultiOutputClassifier,
            Perceptron,
            RidgeClassifier,
            RidgeClassifierCV,
            SGDClassifier,
        )

        if self.classifier not in ["precomputed", None] and not isinstance(
            self.classifier, compatible_classifiers
        ):
            raise ValueError(
                "Classifier must be an instance of `"
                f"{'`, `'.join(c.__name__ for c in compatible_classifiers)}`"
                ", or `precomputed`"
            )

        if self.classifier != "precomputed":
            if self.classifier is None:
                classifier = LogisticRegression()
            else:
                classifier = self.classifier

            # Check if classifier is fitted; if not, fit with precomputed K
            # to avoid needing to compute the kernel a second time
            self.z_classifier_ = check_cl_fit(
                classifier, K, X, y
            )  # Pkz as weights - fits on K, y

            if isinstance(self.z_classifier_, MultiOutputClassifier):
                W = np.hstack([est_.coef_.T for est_ in self.z_classifier_.estimators_])
                Z = K @ W  # computes Z, basically Z=XPxz
            else:
                print("Coef: " + str(self.z_classifier_.coef_.shape))
                # this fails with prefit classifier on X, y, since weights are shape (1, n_features)
                # and K_features != X_features
                # In KPCovR, this is OK since Kernel Ridge Regression
                W = self.z_classifier_.coef_.T.reshape(K.shape[1], -1)
                print("W: " + str(W.shape))
                Z = self.z_classifier_.decision_function(K).reshape(K.shape[0], -1)

            # Use this instead of `self.classifier_.predict(K)`
            # so that we can handle the case of the pre-fitted classifier
            # Z = K @ W #K @ Pkz

            # When we have an unfitted classifier,
            # we fit it with a precomputed K
            # so we must subsequently "reset" it so that
            # it will work on the particular X
            # of the KPCovR call. The dual coefficients are kept.
            # Can be bypassed if the classifier is pre-fitted.
            # try:
            #     check_is_fitted(classifier)
            # except NotFittedError:
            #     self.z_classifier_.set_params(**classifier.get_params())
            #     self.z_classifier_.X_fit_ = self.X_fit_
            #     self.z_classifier_._check_n_features(self.X_fit_, reset=True)
        else:
            Z = X @ W
            # Do we want precomputed classifier to be trained on K and Y, X and Y?
            if W is None:
                W = np.linalg.lstsq(X, Z, self.tol)[0]

        self._label_binarizer = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self._label_binarizer.fit_transform(y)
        if not self._label_binarizer.y_type_.startswith("multilabel"):
            y = column_or_1d(y, warn=True)

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

        self._fit(K, Z, W)  # gives us T, Pkt, self.pt__

        self.ptk_ = self.pt__ @ K

        if self.fit_inverse_transform:
            self.ptx_ = self.pt__ @ X

        # self.classifier_ = check_cl_fit(classifier, K @ self.pkt_, y) # Extract weights to get Ptz
        if self.classifier != "precomputed":
            self.classifier_ = clone(classifier).fit(K @ self.pkt_, y)
        else:
            self.classifier_ = LogisticRegression().fit(K @ self.pkt_, y)
        # self.classifier_._validate_data(K @ self.pkt_, y, reset=False)

        if isinstance(self.classifier_, MultiOutputClassifier):
            self.ptz_ = np.hstack(
                [est_.coef_.T for est_ in self.classifier_.estimators_]
            )
            self.pkz_ = self.pkt_ @ self.ptz_
        else:
            self.ptz_ = self.classifier_.coef_.T
            self.pkz_ = self.pkt_ @ self.ptz_

        if len(Y.shape) == 1:
            self.pkz_ = self.pkz_.reshape(
                X.shape[1],
            )
            self.ptz_ = self.ptz_.reshape(
                self.n_components_,
            )

        self.components_ = self.pkt_.T  # for sklearn compatibility
        return self

        # if self.classifier != "precomputed":
        #     if self.classifier is None:
        #         classifier = LogisticRegression()
        #     else:
        #         classifier = self.classifier

        #     self.z_classifier_ = check_cl_fit(
        #         classifier, K, X, y
        #     )  # its linear classifier on x and y to get Pxz

        #     print("K: "+str(K.shape))
        #     print("Z_clasifier_coef: "+str(self.z_classifier_.coef_.shape))
        #     if isinstance(self.z_classifier_, MultiOutputClassifier):
        #         W = np.hstack([est_.coef_.T for est_ in self.z_classifier_.estimators_])
        #         Z = K @ W  # computes Z, basically Z=XPxz

        #     else:
        #         W = self.z_classifier_.coef_.T.reshape(K.shape[1], -1) # maybe try n_features_in like KPCovR line 338
        #         Z = self.z_classifier_.decision_function(K).reshape(K.shape[0], -1)

        # else:
        #     Z = K @ W
        #     if W is None:
        #         W = np.linalg.lstsq(K, Z, self.tol)[0]  # W = weights for Pxz

        # self._label_binarizer = LabelBinarizer(neg_label=-1, pos_label=1)
        # Y = self._label_binarizer.fit_transform(y)  # check if we need this
        # if not self._label_binarizer.y_type_.startswith("multilabel"):
        #     y = column_or_1d(y, warn=True)

        # if self.space_ == "feature":
        #     self._fit_feature_space(K, Y.reshape(Z.shape), Z)
        # else:
        #     self._fit_sample_space(K, Y.reshape(Z.shape), Z, W)

        # if self.classifier != "precomputed":
        #     self.classifier_ = clone(classifier).fit(K @ self.pxt_, y)
        # else:
        #     self.classifier_ = LogisticRegression().fit(K @ self.pxt_, y)

        # if isinstance(self.classifier_, MultiOutputClassifier):
        #     self.ptz_ = np.hstack(
        #         [est_.coef_.T for est_ in self.classifier_.estimators_]
        #     )
        #     self.pxz_ = self.pxt_ @ self.ptz_
        # else:
        #     self.ptz_ = self.classifier_.coef_.T
        #     self.pxz_ = self.pxt_ @ self.ptz_

        # if len(Y.shape) == 1:
        #     self.pxz_ = self.pxz_.reshape(
        #         X.shape[1],
        #     )
        #     self.ptz_ = self.ptz_.reshape(
        #         self.n_components_,
        #     )

        # print("Components: "+str(self.pxt_.T.shape))
        # print("Pxt: "+str(self.pxt_.shape))

        # self.components_ = self.pxt_.T  # for sklearn compatibility

        # if self.fit_inverse_transform:
        #     self.inverse_coef_ = linalg.solve(K, X, assume_a="pos", overwrite_a=True)

        # return self

    def inverse_transform(self, T):
        return T @ self.ptx_

        # K = super().inverse_transform(T)
        # return np.dot(K, self.inverse_coef_)

    def decision_function(self, X=None, T=None):
        check_is_fitted(self, attributes=["_label_binarizer", "pkz_", "ptz_"])

        xp, _ = get_namespace(X)

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            X = validate_data(self, X, reset=False)
            K = self._get_kernel(X, self.X_fit_)
            if self.center:
                K = self.centerer_.transform(K)
            scores = K @ self.pkz_

        else:
            T = check_array(T)
            scores = T @ self.ptz_

        return (
            xp.reshape(scores, (-1,))
            if (scores.ndim > 1 and scores.shape[1] == 1)
            else scores
        )

    def predict(self, X=None, T=None):
        """Predicts class values from X or T."""
        check_is_fitted(self, ["_label_binarizer", "pkz_", "ptz_"])

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            X = validate_data(self, X, reset=False)
            K = self._get_kernel(X, self.X_fit_)
            if self.center:
                K = self.centerer_.transform(K)

            return self.classifier_.predict(
                K @ self.pkt_
            )  # Ptz(T) -> activation -> Y labels
        else:
            return self.classifier_.predict(T)  # Ptz(T) -> activation -> Y labels

    def transform(self, X=None):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components as determined by the
        modified Kernel PCovR distances.

        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        """
        check_is_fitted(self, ["pkt_", "X_fit_"])

        X = validate_data(self, X, reset=False)
        K = self._get_kernel(X, self.X_fit_)

        if self.center:
            K = self.centerer_.transform(K)

        return K @ self.pkt_

    def score(self, X, Y, sample_weight=None):
        X, Y = validate_data(self, X, Y, reset=False)

        return accuracy_score(Y, self.predict(X), sample_weight=sample_weight)

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
