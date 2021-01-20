import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .sparsified import _Sparsified
from ..selection.FPS import SampleFPS
from ..utils.eig_solver import eig_solver


class SparseKPCA(_Sparsified):
    """

    :param n_components: number of components for which selection is carried out in SparseKPCA and SparseKRCovR
    :param n_active: the size of the small dataset used in learning
    Parameters
    ----------
    n_components : int, default=None
        Number of components. If None, all non-zero components are kept.
        
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

    kernel_params : dict, default=None
        Parameters (keyword arguments) and
        values for kernel passed as callable object.
        Ignored by other kernels.

    fit_inverse_transform : bool, default=False
        Learn the inverse transform for non-precomputed kernels.
        (i.e. learn to find the pre-image of a point)

    eigen_solver : {'auto', 'dense', 'arpack'}, default='auto'
        Select eigensolver to use. If n_components is much less than
        the number of training samples, arpack may be more efficient
        than the dense eigensolver.

    tol : float, default=0
        Convergence tolerance for arpack.
        If 0, optimal value will be chosen by arpack.

    copy_X : bool, default=True
        If True, input X is copied and stored by the model in the `X_fit_`
        attribute. If no further changes will be done to X, setting
        `copy_X=False` saves memory by storing a reference.

    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. 
    """

    def __init__(
        self,
        n_components,
        n_active=None,
        center=True,
        selector=SampleFPS,
        selector_args={},
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        fit_inverse_transform=False,
        tol=0,
        n_jobs=1,
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
            selector_args=selector_args,
        )
        self.n_components = n_components
        self.tol = tol
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
        K_NM_, K_MM_ = self._get_kernel_matrix(X, X_sparse)

        vmm, Umm = eig_solver(K_MM_, n_components=self.n_active, tol=self.tol)
        v_invsqrt = np.sqrt(np.linalg.pinv(np.diagflat(vmm)))
        U_active = Umm[:, : self.n_active] @ v_invsqrt

        phi_active = K_NM_ @ U_active
        C = phi_active.T @ phi_active
        v_C, U_C = eig_solver(C, n_components=self.n_components, tol=self.tol)
        self.pkt_ = U_active @ U_C[:, : self.n_components]
        if X is not None and self._fit_inverse_transform:
            T = K_NM_ @ self.pkt_
            v_C_inv = np.linalg.pinv(np.diagflat(v_C[: self.n_components]))
            self.ptx_ = v_C_inv @ T.T @ X
        return self

    def transform(self, X):
        """
        Projecting  feature matrix into the latent space
        :param X: feature matrix or kernel matrix between X and X_sparse
        :return: T, projection into the latent space
        """
        X = check_array(X)
        check_is_fitted(self, ["pkt_", "X_sparse_"])

        K_NM_ = self._get_kernel(X, self.X_sparse_)
        if self.center:
            K_NM_ = self.kfc.transform(K_NM_)

        return K_NM_ @ self.pkt_

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
        self.transform(X)

    def inverse_transform(self, T):
        """Transform T back to original space.


        :param T : matrix for inverse transform
        :type T: ndarray of shape (n_samples, n_features)

        :return X_new: result of inverse transformation
        :type X_new: ndarray of shape (n_samples, n_features)

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

        check_is_fitted(self, ["ptx_"])

        return T @ self.ptx_
