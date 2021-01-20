from abc import ABCMeta
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.metrics.pairwise import pairwise_kernels

from ..preprocessing.flexible_scaler import KernelFlexibleCenterer
from ..selection.FPS import SampleFPS


class _Sparsified(TransformerMixin, RegressorMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Super-class defined sparcified methods


      n_active : int, default=None
          Number of samples in the active set.
          
      center: boolean, default=True
          Whether to center the sparse kernel matrices
          
      selector: class of type consistent to those in `skcosmo.selection`, default=`skcosmo.selection.SampleFPS`
          Object class to use to select the active set. 
          
     selector_args: dictionary, default={}
          constructor arguments for selector
  
      kernel : {'linear', 'poly', \
              'rbf', 'sigmoid', 'cosine', 'precomputed'}, default='linear'
          Kernel used for PCA.
  
      gamma : float, default=None
          Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
          kernels. If ``gamma`` is ``None``, then it is set to ``1/n_features``.
  
      degree : int, default=3
          Degree for poly kernels. Ignored by other kernels.
  
      coef0 : float, default=1
          Independent term in poly and sigmoid kernels.
          Ignored by other kernels.
  
      kernel_params : dict, default={}
          Parameters (keyword arguments) and
          values for kernel passed as callable object.
          Ignored by other kernels.
 
      n_jobs : int, default=None
          The number of parallel jobs to run.
          ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
          ``-1`` means using all processors. 
    """

    def __init__(
        self,
        n_active=None,
        center=True,
        selector=SampleFPS,
        selector_args={},
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params={},
        n_jobs=1,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_active = n_active
        self.center = center
        self.n_jobs = n_jobs
        self._selector = selector
        self._selector_args = selector_args

    def _get_kernel(self, X, Y=None):
        """
        Calculate kernel for the matrix X or (optionally) for matrix X and Y
        :param X: matrix, for which we calculate kernel
        :param Y: A second feature array only if X has shape [n_samples_a, n_features].

        :return: sklearn.metrics.pairwise.pairwise_kernels(X, Y)
        """
        if self.kernel == "precomputed":
            if X.shape[-1] != self.n_active:
                raise ValueError("The supplied kernel does not match n_active.")
            return X
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params
        )

    def _get_kernel_matrix(self, X, X_sparse=None):
        """
        Calculate the Kmm and Knm matrices, which corresponds to the kernel evaluated between the active set samples
        :param X: input matrices, for which we calculate Kmm and Knm
        :param Kmm: kernel evaluated between the active set samples
        :param Knm: kernel matrix between X and X_sparse
        """
        if X_sparse is None:
            selector = self._selector(X, **self._selector_args)

            i_active = selector.select(self.n_active)

            X_sparse = X[i_active]

            if self.kernel == "precomputed":
                X_sparse = X_sparse[:, i_active]

        self.X_sparse_ = X_sparse
        K_MM_ = self._get_kernel(self.X_sparse_)

        K_NM_ = self._get_kernel(X, self.X_sparse_)
        if self.center:
            self.kfc = KernelFlexibleCenterer()
            self.kfc.fit(K_MM_)
            K_MM_ = self.kfc.transform(K_MM_)
            K_NM_ = self.kfc.transform(K_NM_)
        return K_NM_, K_MM_
