import numpy as np
from .Sparsified import _Sparsified
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from ..utils.eig_solver import eig_solver
from sklearn.exceptions import NotFittedError
from ..selection.FPS import SampleFPS


class SparseKPCA(_Sparsified):
    """
    ----Inherited Attributes----
    :param kernel: the kernel used for this learning
    :param gamma: exponential factor of the rbf and sigmoid kernel
    :param degree: polynomial kernel degree
    :param coef0: free term of the polynomial and sigmoid kernel
    :param kernel_params: kernel parameter set
    :param n_active: the size of the small dataset used in learning
    :param tol: Relative accuracy for eigenvalues (stopping criterion) The default value of 0 implies machine precision.
    :param center: if True, centering is carried out
    :param n_jobs: The number of jobs to use for the computation the kernel. This works by breaking down the pairwise matrix into n_jobs even slices and computing them in parallel.
    :param n_components: number of components for which selection is carried out in SparseKPCA and SparseKRCovR
    """

    def __init__(
        self,
        n_components,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        n_active=None,
        tol=0,
        center=True,
        n_jobs=1,
        selector=SampleFPS,
        fit_inverse_transform=False,
        copy_X=True,
    ):
        super().__init__(
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            center=center,
            n_active=n_active,
            n_jobs=n_jobs,
            selector=selector,
        )
        self.n_components = n_components
        self.tol = tol
        self.pkt_ = None
        self._fit_inverse_transform = fit_inverse_transform
        self.copy_X = copy_X

    def fit(self, X, X_sparse=None, y=None):
        """Fit the model from data in X.

        :param X: Training vector, where n_samples in the number of samples
                  and n_features is the number of features. This may also be the
                  precomputed kernel of shape (n_samples, n_samples)
                  in the case that self.kernel == 'precomputed'
        :type X: {array-like} of shape (n_samples, n_features)
        :param X_sparse: Active set of samples, where n_features is the number of features.
                         This may also be the precomputed active kernel of shape
                         (n_active, n_active) in the case that self.kernel == 'precomputed'
        :type X_sparse: {array-like} of shape (n_active, n_features)
        :return: Returns the instance itself.
        """
        X = check_array(X, copy=self.copy_X)
        self._define_kernel_matrix(X, X_sparse)

        vmm, Umm = eig_solver(self.K_sparse_, n_components=self.n_active, tol=self.tol)
        v_invsqrt = np.sqrt(np.linalg.pinv(np.diagflat(vmm)))
        U_active = Umm[:, : self.n_active] @ v_invsqrt

        phi_active = self.K_cross_ @ U_active

        C = phi_active.T @ phi_active
        v_C, U_C = eig_solver(C, n_components=self.n_components, tol=self.tol)

        self.pkt_ = U_active @ U_C[:, : self.n_components]
        if X is not None and self._fit_inverse_transform:
            T = self.Knm @ self.pkt_
            v_C_inv = np.linalg.pinv(np.diagflat(v_C[: self.n_components]))
            self.ptx_ = v_C_inv @ T.T @ X
        return self

    def transform(self, X=None, K_cross=None):
        """
        Projecting  feature matrix into the latent space
        :param X: feature matrix
        :param K_cross: kernel matrix between X and X_sparse
        :return: T, projection into the latent space
        """
        check_is_fitted(self, ["pkt_", "X_sparse_"])
        if X is None and K_cross is None:
            raise Exception("Error: required feature or kernel matrices")

        if K_cross is None:
            K_cross = self._get_kernel(X, self.X_sparse_)
        if self.center:
            K_cross = self.kfc.transform(K_cross)

        return self._project(K_cross, "pkt_")

    def fit_transform(self, X, X_sparse=None, y=None):
        """
        Both fit and transform

        :param X: Training vector, where n_samples in the number of samples
               and n_features is the number of features. This may also be the
               precomputed kernel of shape (n_samples, n_samples)
               in the case that self.kernel == 'precomputed'
        :type X: {array-like} of shape (n_samples, n_features)
        :param y: properties matrix
        :param X_sparse:  Active set of samples, where n_features is the number of features.
                         This may also be the precomputed active kernel of shape
                         (n_active, n_active) in the case that self.kernel == 'precomputed'
        :type X_sparse: {array-like} of shape (n_active, n_features)
        :return: T, projection into the latent space
        """
        self.fit(X, X_sparse, y)
        self.transform(X, self.K_cross_)

    def inverse_transform(self, X):
        """Transform X back to original space.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_components)

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)

        References
        ----------
        "Learning to Find Pre-Images", G BakIr et al, 2004.
        """
        if not self.fit_inverse_transform:
            raise NotFittedError(
                "The fit_inverse_transform parameter was not"
                " set to True when instantiating and hence "
                "the inverse transform is not available."
            )

        return self.transform(X) @ self.ptx_
