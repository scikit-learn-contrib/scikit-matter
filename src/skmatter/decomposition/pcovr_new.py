from sklearn.decomposition._base import _BasePCA
from sklearn.linear_model import (
    LinearRegression, 
    Ridge, 
    RidgeCV
)
from sklearn.linear_model._base import LinearModel
from sklearn.utils.validation import check_is_fitted

import sys
sys.path.append('scikit-matter')
from src.skmatter.decomposition._pcov import _BasePCov


class PCovR(_BasePCov):

    def __init__(
        self,
        mixing=0.5,
        n_components=None,
        svd_solver="auto",
        tol=1e-12,
        space="auto",
        regressor=None,
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
            regressor=regressor,
            iterated_power=iterated_power,
            random_state=random_state,
            whiten=whiten,
            subclass="PCovR")

    def fit(self, X, Y, W=None):
        if not any(
            [
                self.regressor is None,
                self.regressor == "precomputed",
                isinstance(
                    self.regressor, 
                    (
                           LinearRegression,
                           Ridge,
                           RidgeCV
                        ),
                ),
            ]
        ):
            raise ValueError(
                "Regressor must be an instance of "
                "`LinearRegression`, `Ridge`, `RidgeCV`, or `precomputed`"
            )
        return super().fit(X, Y, W)

    def inverse_transform(self, T):
        return super().inverse_transform(T)

    def predict(self, X=None, T=None):
        check_is_fitted(self, ["pxy_", "pty_"])
        return super().predict(X, T)

    def transform(self, X=None):
        return super().transform(X)


    def score(self, X, Y, T=None):
        return super().score(X, Y, T)
