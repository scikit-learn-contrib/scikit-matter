import numpy as np
from sklearn import clone

from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import (
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV,
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier,
)
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.utils.multiclass import check_classification_targets, type_of_target

from skmatter.utils import check_kcl_fit
from skmatter.decomposition import _BaseKPCov


class KernelPCovC(LinearClassifierMixin, _BaseKPCov):
    def __init__(
        self,
        mixing=0.5,
        n_components=None,
        svd_solver="auto",
        classifier=None,
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
        super().__init__(
            mixing=mixing,
            n_components=n_components,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
            center=center,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            n_jobs=n_jobs,
            fit_inverse_transform=fit_inverse_transform,
        )
        self.classifier = classifier

    def fit(self, X, y, W=None):
        X, y = validate_data(self, X, y, y_numeric=False, multi_output=True)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        K = super()._fit_utils(X)

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
            self.z_classifier_ = check_kcl_fit(classifier, K, X, y)

            if isinstance(self.z_classifier_, MultiOutputClassifier):
                W = np.hstack([est_.coef_.T for est_ in self.z_classifier_.estimators_])
                Z = K @ W
            else:
                # this fails with prefit classifier on X, y, since weights are shape (1, n_features)
                # and K_features != X_features
                # In KPCovR, this is OK since Kernel Ridge Regression
                W = self.z_classifier_.coef_.T.reshape(K.shape[1], -1)

                Z = K @ W

        else:
            if W is None:
                W = np.linalg.lstsq(K, Z, self.tol)[0]
            Z = K @ W

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

        if len(y.shape) == 1 and type_of_target(y) == "binary":
            self.pkz_ = self.pkz_.reshape(
                K.shape[1],
            )
            self.ptz_ = self.ptz_.reshape(
                self.n_components_,
            )

        self.components_ = self.pkt_.T  # for sklearn compatibility
        return self

    def predict(self, X=None, T=None):
        """Predicts class values from X or T."""
        check_is_fitted(self, ["pkz_", "ptz_"])

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            X = validate_data(self, X, reset=False)
            K = self._get_kernel(X, self.X_fit_)
            if self.center:
                K = self.centerer_.transform(K)

            return self.classifier_.predict(K @ self.pkt_)
        else:
            return self.classifier_.predict(T)

    def inverse_transform(self, T):
        r"""Transform input data back to its original space.

        .. math::
            \mathbf{\hat{X}} = \mathbf{T} \mathbf{P}_{TX}
                              = \mathbf{K} \mathbf{P}_{KT} \mathbf{P}_{TX}

        Similar to KPCA, the original features are not always recoverable,
        as the projection is computed from the kernel features, not the original
        features, and the mapping between the original and kernel features
        is not one-to-one.

        Parameters
        ----------
        T : numpy.ndarray, shape (n_samples, n_components)
            Projected data, where n_samples is the number of samples and n_components is
            the number of components.

        Returns
        -------
        X_original : numpy.ndarray, shape (n_samples, n_features)
        """
        return super().inverse_transform(T)

    def transform(self, X):
        """Apply dimensionality reduction to X.

        ``X`` is projected on the first principal components as determined by the
        modified Kernel PCovR distances.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        """
        return super().transform(X)

    def decision_function(self, X=None, T=None):
        check_is_fitted(self, attributes=["pkz_", "ptz_"])

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            X = validate_data(self, X, reset=False)
            K = self._get_kernel(X, self.X_fit_)
            if self.center:
                K = self.centerer_.transform(K)
            return K @ self.pkz_ + self.classifier_.intercept_

        else:
            T = check_array(T)
            return T @ self.ptz_ + self.classifier_.intercept_
