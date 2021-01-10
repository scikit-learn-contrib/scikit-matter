import numpy as np
from abc import ABCMeta
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.metrics.pairwise import pairwise_kernels
from ..preprocessing.flexible_scaler import KernelFlexibleCenterer
from ..selection.FPS import SampleFPS


class _Sparsified(TransformerMixin, RegressorMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Super-class defined sparcified methods

        :param kernel: the kernel used for this learning
        :param gamma: exponential factor of the rbf and sigmoid kernel
        :param degree: polynomial kernel degree
        :param coef0: free term of the polynomial and sigmoid kernel
        :param kernel_params: kernel parameter set
        :param n_active: the size of the small dataset used in learning
        :param regularization: regularization parameter for SparseKRR and SparseKRCovR
        :param tol: Relative accuracy for eigenvalues (stopping criterion) The default value of 0 implies machine precision.
        :param center: if True, centering of kernel during the learning is carried out
        :param n_jobs: The number of jobs to use for the computation the kernel. This works by breaking down the pairwise matrix into n_jobs even slices and computing them in parallel.
        :param n_components: number of components for which selection is carried out in SparseKPCA and SparseKRCovR
    """

    def __init__(
        self,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        n_active=None,
        center=True,
        n_jobs=1,
        selector=SampleFPS,
    ):
        """
        Initializes superclass for sparse methods

        :param kernel: the kernel used for this learning
        :param gamma: exponential factor of the rbf and sigmoid kernel
        :param degree: polynomial kernel degree
        :param coef0: free term of the polynomial and sigmoid kernel
        :param kernel_params: kernel parameter set
        :param n_active: the size of the small dataset used in learning
        :param center: if True, centering of kernel during the learning is carried out
        :param n_jobs: The number of jobs to use for the computation the kernel. This works by breaking down the pairwise matrix into n_jobs even slices and computing them in parallel.
        """
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_active = n_active
        self.center = center
        self.n_jobs = n_jobs
        self._selector = selector

    def _project(self, A, projector):
        """Apply a projector to matrix A

        :param A: projector: string corresponding to the named projection matrix of shape (a, p)
        :type array-like, shape (n, a)

        :return: A'
        :type array-like, shape (n, p)
        """
        check_is_fitted(self, projector)

        A = check_array(A)
        A_transformed = np.dot(A, self.__dict__[projector])
        return A_transformed

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

    def _define_kernel_matrix(self, X, X_sparse=None):
        """
        Calculate the Kmm and Knm matrices, which correspons to   kernel evaluated between the active set samples
        and kernel matrix between X and X_sparse respectively
        :param X: input matrices, for which we calculate Kmm and Knm
        :param Kmm: kernel evaluated between the active set samples
        :param Knm: kernel matrix between X and X_sparse
        """
        if X_sparse is None:
            selector = self._selector(X)

            i_active = selector.select(self.n_active)

            X_sparse = X[i_active, :]

            if self.kernel == "precomputed":
                X_sparse = X_sparse[:, i_active]

        self.X_sparse_ = X_sparse
        K_sparse_ = self._get_kernel(self.X_sparse_, self.X_sparse_)

        K_cross_ = self._get_kernel(X, self.X_sparse_)
        if self.center:
            self.kfc = KernelFlexibleCenterer()
            self.kfc.fit(K_sparse_)
            K_sparse_ = self.kfc.transform(K_sparse_)
            K_cross_ = self.kfc.transform(K_cross_)
        self.K_sparse_ = K_sparse_
        self.K_cross_ = K_cross_
