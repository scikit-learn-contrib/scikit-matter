from abc import ABCMeta
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.metrics.pairwise import pairwise_kernels

from ..selection.FPS import SampleFPS


class Sparsified(TransformerMixin, RegressorMixin, BaseEstimator, metaclass=ABCMeta):
    r"""
    Super-class defined sparsified methods. Computes the Nystroem-approximated
    kernels using a subset of the samples as an active set.

    .. math::

      \mathbf{K} \approx \mathbf{K}_{NN} = \mathbf{K}_{NM} \mathbf{K}_{MM}^{-1} \mathbf{K}_{NM}^T

    where `N` denotes the full set of samples and `M` is a subset of active points.
    Unlike `sklearn.approximation.Nystroem`, this class allows the selection or
    direct input of the active set .

    Parameters
    ==========

    n_active: int, default=None
      Number of samples in the active set.

    selector: class of type consistent to those in `skcosmo.selection`, default=`skcosmo.selection.SampleFPS`
      Object class to use to select the active set.

    selector_args: dictionary, default={}
      constructor arguments for selector

    kernel: {'linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'}, default='linear'
      Kernel used for PCA.

    gamma: float, default=None
      Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
      kernels. If ``gamma`` is ``None``, then it is set to ``1/n_features``.

    degree: int, default=3
      Degree for poly kernels. Ignored by other kernels.

    coef0: float, default=1
      Independent term in poly and sigmoid kernels.
      Ignored by other kernels.

    kernel_params: dict, default={}
      Parameters (keyword arguments) and
      values for kernel passed as callable object.
      Ignored by other kernels.

    n_jobs: int, default=None
      The number of parallel jobs to run.
      ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
      ``-1`` means using all processors.
    """

    def __init__(
        self,
        n_active=None,
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
        self.n_jobs = n_jobs
        self._selector = selector
        self._selector_args = selector_args

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        """
        Calculate kernel for the matrix X or (optionally) for matrix X and Y

        :param X: matrix, for which we calculate kernel
        :param Y: (optional) A second feature array with the same number of features to as`X`

        :return: sklearn.metrics.pairwise.pairwise_kernels(X, Y)
        """
        if self.kernel == "precomputed":
            if X.shape[-1] != self.n_active:
                raise ValueError("The supplied kernel does not match n_active.")
            return X
        if callable(self.kernel):
            params = self.kernel_params
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params
        )

    def _set_sparse_kernel(self, X, X_sparse=None):
        r"""
        Sets the active set and its corresponding kernel

        :param X: input matrix of N samples
        :param X: (optional) active set of M samples
        """

        if X_sparse is None:
            selector = self._selector(X, **self._selector_args)

            i_active = selector.select(self.n_active)

            X_sparse = X[i_active]

            # in the case that the full kernel has been passed but the active set not designated,
            # then K_MM = K[i_active][:, i_active]
            if self.kernel == "precomputed":
                X_sparse = X_sparse[:, i_active]

        self.X_sparse_ = X_sparse
        return self._get_kernel(self.X_sparse_)
