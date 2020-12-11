import numpy as np
from abc import abstractmethod, ABCMeta
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from scipy.sparse.linalg import eigs
from sklearn.metrics.pairwise import pairwise_kernels
from ..preprocessing.flexible_scaler import KernelFlexibleCenterer
from ..selection.FPS import SampleFPS


class _Sparsified(TransformerMixin, RegressorMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Super-class defined sparcified methods

        :param mixing: mixing parameter. Shows the relationship between SprseKPCA and SparseKRR (used only for KPCovR)
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
        mixing,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        n_active=None,
        regularization=1e-12,
        tol=0,
        center=True,
        n_jobs=1,
        n_components=None,
    ):
        """
        Initializes superclass for sparse methods

        :param mixing: mixing parameter. Shows the relationship between SprseKPCA and SparseKRR (used only for KPCovR)
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
        self.mixing = mixing
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_active = n_active
        self.regularization = regularization
        self.tol = tol
        self.center = center
        self.n_jobs = n_jobs
        self.n_components = n_components

    @abstractmethod
    def fit(self, X, Y, Yhat=None):
        """Placeholder for fit. Subclasses should implement this method!
        Fit the model with X.

        :param X: array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        :param Y: array-like, shape (n_samples, n_properties)
            Training data, where n_samples is the number of samples and
            n_properties is the number of properties
        :param Yhat: array-like, shape (n_samples, n_properties), optional
            Regressed training data, where n_samples is the number of samples and
            n_properties is the number of properties. If not supplied, computed
            by ridge regression.
        :return: Returns the instance itself.
        """

    @abstractmethod
    def transform(self, X):
        """Placeholder for transform. Subclasses should implement this method!
        Transforms the model with X.

        :param X: array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        :return: Tranformed matrix
        """

    def predict(self, X):
        """Placeholder for predict.
        Predicts the outputs given X.


        :param X: array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        :return: Predicted properties Y
        """

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
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params
        )

    def _eig_solver(self, matrix, full_matrix=False, tol=None, k=None):
        """
        Calculate eigenvectors and eigenvalues, returns the k vectors corresponding to the k largest values

        :param matrix: matrix, for wich calculates eigenvectors and eigenvalues

        :param full_matrix: Determines whether the matrix is sparse (if False) or full (if True).
        :param tol: Minimum eigenvalue
        :param k: the number of eigenvectors and eigenvalues that we need to return
        :return: k of the largest eigenvalues and the corresponding eigenvectors.
        """
        if tol is None:
            tol = self.tol
        if k is None:
            k = self.n_components
        if full_matrix == False:
            v, U = eigs(matrix, k=k, tol=tol)
        else:
            v, U = np.linalg.eig(matrix)

        U = np.real(U[:, np.argsort(-v)])
        v = np.real(v[np.argsort(-v)])

        U = U[:, v > tol]
        v = v[v > tol]

        if len(v) == 1:
            U = U.reshape(-1, 1)

        return v, U

    def _define_Kmm_Knm(self, X, Kmm=None, Knm=None):
        """
        Calculate the Kmm and Knm matrices, which correspons to   kernel evaluated between the active set samples
        and kernel matrix between X and X_sparse respectively
        :param X: input matrices, for which we calculate Kmm and Knm
        :param Kmm: kernel evaluated between the active set samples
        :param Knm: kernel matrix between X and X_sparse
        """
        if Kmm is None or Knm is None:
            fps = SampleFPS(X)

            i_sparse = fps.select(self.n_active)

            self.X_sparse = X[i_sparse, :]
            Kmm = self._get_kernel(self.X_sparse, self.X_sparse)

            Knm = self._get_kernel(X, self.X_sparse)
        if self.center:
            self.kfc = KernelFlexibleCenterer()
            self.kfc.fit(Kmm)
            Kmm = self.kfc.transform(Kmm)
            Knm = self.kfc.transform(Knm)
        self.Kmm = Kmm
        self.Knm = Knm
