import numpy as np
from abc import abstractmethod, ABCMeta
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from scipy.sparse.linalg import eigs
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels


class _Sparsified(TransformerMixin, RegressorMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Super-class defined sparcified methods
    """

    def __init__(self, mixing, kernel="linear", gamma=None, degree=3,
                 coef0=1, kernel_params=None, n_active=None,
                 regularization=1E-12, tol=0, center=True, n_jobs=None, n_components=None):
        # TODO
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
        self.n_components = n_components #TODO solve, where we should define it


    @abstractmethod
    def fit(self, X, Y, Yhat=None):
        """Placeholder for fit. Subclasses should implement this method!
        Fit the model with X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        Y : array-like, shape (n_samples, n_properties)
            Training data, where n_samples is the number of samples and
            n_properties is the number of properties
        Yhat : array-like, shape (n_samples, n_properties), optional
            Regressed training data, where n_samples is the number of samples and
            n_properties is the number of properties. If not supplied, computed
            by ridge regression.
        Returns
        -------
        self : object
            Returns the instance itself.
        """

    @abstractmethod
    def transform(self, X):
        """Placeholder for transform. Subclasses should implement this method!
        Transforms the model with X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        self : object
            Returns the instance itself.
        """

    @abstractmethod
    def predict(self, X):
        """Placeholder for transform. Subclasses should implement this method!
        Predicts the outputs given X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        self : object
            Returns the instance itself.
        """

    def _project(self, A, projector):
        """Apply a projector to matrix A
        Parameters
        ----------
        A : array-like, shape (n, a)
        projector: string corresponding to the named projection matrix of shape (a, p)
        Returns
        -------
        A' : array-like, shape (n, p)
        Examples
        --------
        >>> todo
        """
        check_is_fitted(self, projector)

        A = check_array(A)
        A_transformed = np.dot(A, self.__dict__[projector])
        return A_transformed

    def _get_kernel(self, X, Y=None):
         if callable(self.kernel):
             params = self.kernel_params or {}
         else:
             params = {"gamma": self.gamma,
                       "degree": self.degree,
                       "coef0": self.coef0}
         return pairwise_kernels(X, Y, metric=self.kernel,
                                 filter_params=True, n_jobs=self.n_jobs,
                                 **params)

    def _eig_solver(self, matrix, full_matrix=False, tol =None, k = None):
        if tol is None:
            tol=self.tol
        if k is None:
            k = self.n_components
        if(full_matrix == False):
            v, U = eigs(matrix, k=k, tol=tol)
        else:
            v, U = np.linalg.eig(matrix)

        U = np.real(U[:, np.argsort(-v)])
        v = np.real(v[np.argsort(-v)])

        U = U[:, v > tol]
        v = v[v > tol]

        if(len(v) == 1):
            U = U.reshape(-1, 1)

        return v, U

    def _define_Kmm_Knm(self,X, Kmm=None, Knm=None):
        if Kmm is None or Knm is None:
            i_sparse, _ = self.FPS(X, self.n_active)

            self.X_sparse = X[i_sparse, :]
            Kmm = self._get_kernel(self.X_sparse, self.X_sparse)

            Knm = self._get_kernel(X, self.X_sparse)

        if self.center:
            Kmm = KernelCenterer().fit_transform(Kmm)
        Knm = KernelCenterer().fit_transform( Knm )
        self.Kmm = Kmm
        self.Knm = Knm


    def FPS(self, X, n=0, idx=None):
        """
            #TODO Deside, where this function should be
            Does Farthest Point Selection on a set of points X
            Adapted from a routine by Michele Ceriotti
        """
        N = X.shape[0]

        # If desired number of points less than or equal to zero,
        # select all points
        if n <= 0:
            n = N

        # Initialize arrays to store distances and indices
        fps_idxs = np.zeros( n, dtype=np.int )
        d = np.zeros( n )

        if idx is None:
            # Pick first point at random
            idx = np.random.randint( 0, N )
        fps_idxs[0] = idx

        # Compute distance from all points to the first point
        d1 = np.linalg.norm( X - X[idx], axis=1 ) ** 2

        # Loop over the remaining points...
        for i in range( 1, n ):

            # Get maximum distance and corresponding point
            fps_idxs[i] = np.argmax( d1 )
            d[i - 1] = np.amax( d1 )

            # Compute distance from all points to the selected point
            d2 = np.linalg.norm( X - X[fps_idxs[i]], axis=1 ) ** 2

            # Set distances to minimum among the last two selected points
            d1 = np.minimum( d1, d2 )

            if np.abs( d1 ).max() == 0.0:
                print( "Only {} FPS Possible".format( i ) )
                return fps_idxs[:i], d[:i]

        return fps_idxs, d
