from sklearn.decomposition._base import _BasePCA
from sklearn.linear_model import (
    RidgeClassifier,
    RidgeClassifierCV,
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier
)
from sklearn.svm import LinearSVC
from sklearn.linear_model._base import LinearModel
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.multioutput import MultiOutputClassifier

import sys
sys.path.append('scikit-matter')
from src.skmatter.decomposition._pcov import _BasePCov

class PCovC(_BasePCov):
   
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
        whiten=False,
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
            whiten=whiten,
            subclass="PCovC")

    def fit(self, X, Y, W=None):
        if not any(
            [
                self.classifier is None,
                self.classifier == "precomputed",
                isinstance(
                    self.classifier,
                    (
                        RidgeClassifier,
                        RidgeClassifierCV,
                        LogisticRegression,
                        LogisticRegressionCV,
                        SGDClassifier,
                        LinearSVC,
                        MultiOutputClassifier,
                    ),
                ),
            ]
        ):
            raise ValueError(
                "classifier must be an instance of "
                "`RidgeClassifier`, `RidgeClassifierCV`, `LogisticRegression`,"
                "`Logistic RegressionCV`, `SGDClassifier`, `LinearSVC`,"
                "`MultiOutputClassifier`, or `precomputed`"
            )
        return super().fit(X, Y, W)
    
    def inverse_transform(self, T):
        return super().inverse_transform(T)

    def decision_function(self, X=None, T=None):
        check_is_fitted(self, attributes=["_label_binarizer", "pxz_", "ptz_"])

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            X = check_array(X)
            return X @ self.pxz_
        else:
            T = check_array(T)
            return T @ self.ptz_
        
    def predict(self, X=None, T=None):
        check_is_fitted(self, attributes=["_label_binarizer", "pxz_", "ptz_"])
        return super().predict(X, T)
        
    def transform(self, X=None):
        return super().transform(X)

    def score(self, X, Y, T=None):
        return super().score(X, Y, T)
