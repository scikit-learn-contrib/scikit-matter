import numpy as np
from functools import partial

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import pairwise_kernels

from ..utils import eig_solver
from ..selection.FPS import SampleFPS
from ..preprocessing.flexible_scaler import SparseKernelCenterer


class SparseKPCA(TransformerMixin, BaseEstimator):
    """SparseKernel Principal component analysis (SKPCA).

    Non-linear dimensionality reduction through the use of sparse kernels
    determined via the Nystrom approximation.

    Parameters
    ----------

    n_active : int
        Number of active samples to use within the sparse kernel.

    n_components : int, default=n_active
        Number of components.

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

    alpha : float, default=1.0
        Hyperparameter of the ridge regression that learns the
        inverse transform (when fit_inverse_transform=True).

    fit_inverse_transform : bool, default=False
        Learn the inverse transform for non-precomputed kernels.
        (i.e. learn to find the pre-image of a point)

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
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    """

    def __init__(
        self,
        n_active,
        n_components=None,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        selector=SampleFPS,
        alpha=1.0,
        fit_inverse_transform=False,
        tol=1e-12,
        copy_X=True,
        n_jobs=None,
    ):
        if fit_inverse_transform and kernel == "precomputed":
            raise ValueError("Cannot fit_inverse_transform with a precomputed kernel.")

        self.n_active = n_active

        if n_components is not None:
            self.n_components = n_components
        else:
            self.n_components = self.n_active

        self.kernel = kernel
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.tol = tol
        self.n_jobs = n_jobs
        self.copy_X = copy_X
        self.center = False

        self.pkt_ = None
        self.K_sparse_ = None
        self.K_fit_ = None
        self.lambdas_, self.alphas_ = None, None

        self._fit_inverse_transform = fit_inverse_transform
        self._centerer = SparseKernelCenterer(rcond=self.tol)
        self._selector = selector
        self._eig_solver = partial(
            eig_solver, n_components=self.n_components, tol=self.tol, add_null=True
        )

    def _get_kernel(self, X, Y=None):

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

    def fit(self, X=None, X_sparse=None, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features. This may also be the
            precomputed kernel of shape (n_samples, n_samples)
            in the case that self.kernel == 'precomputed'
        X_sparse : {array-like} of shape (n_active, n_features)
            Active set of samples, where n_features is the number of features.
            This may also be the precomputed active kernel of shape
            (n_active, n_active) in the case that self.kernel == 'precomputed'

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X = check_array(X, copy=self.copy_X)
        if X_sparse is None:

            i_active = self._selector(X).select(self.n_active)

            X_sparse = X[i_active]

            if self.kernel == "precomputed":
                X_sparse = X_sparse[:, i_active]

        self.X_sparse_ = X_sparse
        self.K_sparse_ = self._get_kernel(X_sparse)
        self.K_fit_ = self._get_kernel(X, X_sparse)

        if self.center:
            self._centerer.fit(Knm=self.K_fit_, Kmm=self.K_sparse_)
            self.K_sparse_ = self._centerer.transform(self.K_sparse_)
            self.K_fit_ = self._centerer.transform(self.K_fit_)
        else:
            self._centerer.scale_ = 1
            self._centerer.n_active_ = self.n_active
            self._centerer.K_fit_rows_ = np.zeros(self.K_fit_.shape[1])

        self.lambdas_, self.alphas_ = np.linalg.eig(self.K_sparse_)
        v_invsqrt = np.linalg.pinv(np.diagflat(np.sqrt(self.lambdas_)))

        phi_active = self.K_fit_ @ self.alphas_ @ v_invsqrt
        U_active = self.alphas_[:, : self.n_active]

        C = phi_active.T @ phi_active

        v_C, U_C = self._eig_solver(C)

        self.pkt_ = U_active @ U_C[:, : self.n_components]

        if X is not None and self._fit_inverse_transform:
            T = self.K_fit_ @ self.pkt_
            v_C_inv = np.linalg.pinv(np.diagflat(v_C[: self.n_components]))
            self.ptx_ = v_C_inv @ T.T @ X

        return self

    def transform(self, X=None):
        """Transform X.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features). This may also be the
            precomputed kernel of shape (n_samples, n_samples)
            in the case that self.kernel == 'precomputed'

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
        """

        check_is_fitted(self, ["pkt_", "X_sparse_"])
        X = check_array(X, copy=self.copy_X)

        # Compute centered gram matrix between X and training data X_fit_
        K = self._centerer.transform(self._get_kernel(X, self.X_sparse_))

        return K @ self.pkt_

    def fit_transform(self, X, y=None, **params):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features. This may also be the
            precomputed kernel of shape (n_samples, n_samples)
            in the case that self.kernel == 'precomputed'
        **params : {dictionary} of parameters to pass to self.fit()

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
        """
        self.fit(X, y=y, **params)
        return self.K_fit_ @ self.pkt_

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
