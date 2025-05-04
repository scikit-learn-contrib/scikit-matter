import scipy.sparse as sp

from sklearn.base import check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_array

from skmatter.preprocessing import KernelNormalizer

import sys
sys.path.append('scikit-matter')
from src.skmatter.decomposition.pcovc_new import PCovC

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
        center=True, # False in KPCovR, but getting error:
                     # "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT" sometimes 
                     # when training due to unscaled X
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
        self.kernel=kernel
        self.gamma=gamma
        self.degree=degree
        self.coef0=coef0
        self.kernel_params=kernel_params
        self.center=center
        self.n_jobs=n_jobs

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
        K = self._get_kernel(X)

        if self.center:
            self.centerer_ = KernelNormalizer()
            K = self.centerer_.fit_transform(K)
        self.X_fit_ = X.copy()

        return super().fit(K, y, W)
    
    def inverse_transform(self, T):
        return super().inverse_transform(T)

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