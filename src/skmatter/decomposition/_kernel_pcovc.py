import numpy as np
from scipy import linalg
import scipy.sparse as sp

from sklearn.base import check_is_fitted
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import check_array
from sklearn.linear_model import (
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV,
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier,
)
from sklearn.svm import LinearSVC

from skmatter.preprocessing import KernelNormalizer
from skmatter.decomposition import PCovC
from sklearn.utils.validation import check_is_fitted, validate_data

from skmatter.utils import check_cl_fit


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

    def fit(self, X, y, W=None):
        X, y = validate_data(self, X, y, multi_output=True)

        K = self._get_kernel(X)

        if self.center:
            self.centerer_ = KernelNormalizer()
            K = self.centerer_.fit_transform(K)

        self.X_fit_ = X.copy()
        
        super()._fit_utils(X, y)

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

            self.z_classifier_ = check_cl_fit(
                classifier, X, y
            )  # its linear classifier on x and y to get Pxz

            if isinstance(self.z_classifier_, MultiOutputClassifier):
                W = np.hstack([est_.coef_.T for est_ in self.z_classifier_.estimators_])
                Z = X @ W  # computes Z, basically Z=XPxz

            else:
                W = self.z_classifier_.coef_.T.reshape(X.shape[1], -1)
                Z = self.z_classifier_.decision_function(X).reshape(X.shape[0], -1)

        else:
            Z = X @ W
            if W is None:
                W = np.linalg.lstsq(X, Z, self.tol)[0]  # W = weights for Pxz

        self._label_binarizer = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self._label_binarizer.fit_transform(y)  # check if we need this
        if not self._label_binarizer.y_type_.startswith("multilabel"):
            y = column_or_1d(y, warn=True)

        if self.space_ == "feature":
            self._fit_feature_space(X, Y.reshape(Z.shape), Z)
        else:
            self._fit_sample_space(X, Y.reshape(Z.shape), Z, W)

        if self.classifier != "precomputed":
            self.classifier_ = clone(classifier).fit(X @ self.pxt_, y)
        else:
            self.classifier_ = LogisticRegression().fit(X @ self.pxt_, y)

        if isinstance(self.classifier_, MultiOutputClassifier):
            self.ptz_ = np.hstack(
                [est_.coef_.T for est_ in self.classifier_.estimators_]
            )
            self.pxz_ = self.pxt_ @ self.ptz_
        else:
            self.ptz_ = self.classifier_.coef_.T
            self.pxz_ = self.pxt_ @ self.ptz_

        if len(Y.shape) == 1:
            self.pxz_ = self.pxz_.reshape(
                X.shape[1],
            )
            self.ptz_ = self.ptz_.reshape(
                self.n_components_,
            )

        self.components_ = self.pxt_.T  # for sklearn compatibility
        return self

        if self.fit_inverse_transform:
            self.inverse_coef_ = linalg.solve(K, X, assume_a="pos", overwrite_a=True)

        return self

    def inverse_transform(self, T):
        if not self.fit_inverse_transform:
            raise NotFittedError(
                "The fit_inverse_transform parameter was not"
                " set to True when instantiating and hence "
                "the inverse transform is not available."
            )

        K = super().inverse_transform(T)
        return np.dot(K, self.inverse_coef_)

    def decision_function(self, X=None, T=None):
        check_is_fitted(self, attributes=["_label_binarizer", "pxz_", "ptz_"])
        X = check_array(X)
        K = self._get_kernel(X, self.X_fit_)

        if self.center:
            K = self.centerer_.transform(K)

        return super().decision_function(K, T)

    def predict(self, X=None, T=None):
        check_is_fitted(self, attributes=["_label_binarizer", "pxz_", "ptz_"])
        X = check_array(X)
        K = self._get_kernel(X, self.X_fit_)

        if self.center:
            K = self.centerer_.transform(K)

        return super().predict(K, T)

    def transform(self, X=None):
        check_is_fitted(self, ["pxt_", "mean_"])
        X = check_array(X)
        K = self._get_kernel(X, self.X_fit_)

        if self.center:
            K = self.centerer_.transform(K)

        return super().transform(K)

    def score(self, X, Y, sample_weight=None):
        return super().score(X, Y, sample_weight)
