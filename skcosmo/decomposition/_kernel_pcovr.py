import numbers

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import svds
from sklearn.decomposition._base import _BasePCA
from sklearn.decomposition._pca import _infer_dimension
from sklearn.exceptions import NotFittedError
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model._base import LinearModel
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import (
    check_array,
    check_random_state,
)
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils.extmath import (
    randomized_svd,
    stable_cumsum,
    svd_flip,
)
from sklearn.utils.validation import (
    check_is_fitted,
    check_X_y,
)

from ..preprocessing import KernelNormalizer
from ..utils import (
    check_krr_fit,
    pcovr_kernel,
)


class KernelPCovR(_BasePCA, LinearModel):
    r"""
    Kernel Principal Covariates Regression, as described in [Helfrecht2020]_
    determines a latent-space projection :math:`\mathbf{T}` which
    minimizes a combined loss in supervised and unsupervised tasks in the
    reproducing kernel Hilbert space (RKHS).

    This projection is determined by the eigendecomposition of a modified gram
    matrix :math:`\mathbf{\tilde{K}}`

    .. math::

      \mathbf{\tilde{K}} = \alpha \mathbf{K} +
            (1 - \alpha) \mathbf{\hat{Y}}\mathbf{\hat{Y}}^T

    where :math:`\alpha` is a mixing parameter,
    :math:`\mathbf{K}` is the input kernel of shape :math:`(n_{samples}, n_{samples})`
    and :math:`\mathbf{\hat{Y}}` is the target matrix of shape
    :math:`(n_{samples}, n_{properties})`.

    Parameters
    ----------
    mixing: float, default=0.5
        mixing parameter, as described in PCovR as :math:`{\\alpha}`

    n_components: int, float or str, default=None
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == n_samples

    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        If auto :
            The solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        If full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        If arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < min(X.shape)
        If randomized :
            run randomized SVD by the method of Halko et al.

    regressor : instance of `sklearn.kernel_ridge.KernelRidge`, default=None
        The regressor to use for computing
        the property predictions :math:`\\hat{\\mathbf{Y}}`.
        A pre-fitted regressor may be provided. The kernel parameters
        for the unsupervised task are inherited from the regressor
        in order to ensure that they are identical. Consequently,
        even if you are performing a fully unsupervised analysis,
        a regressor is required. The relevant kernel parameters are:
            kernel: "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
                Kernel. Default="linear".

            gamma: float, default=None
                Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
                kernels.

            degree: int, default=3
                Degree for poly kernels. Ignored by other kernels.

            coef0: float, default=1
                Independent term in poly and sigmoid kernels.
                Ignored by other kernels.

            kernel_params: mapping of str to any, default=None
                Parameters (keyword arguments) and values for kernel passed as
                callable object. Ignored by other kernels.
        The regularization `alpha` is also set through the regressor
        and is used in all regression operations. If None,
        `KernelRidge(alpha=1.0e-6, kernel="linear")` is used as the regressor.
        Note that any pre-fitting of the regressor will be lost if `KernelPCovR` is
        within a composite estimator that enforces cloning, e.g.,
        `sklearn.compose.TransformedTargetRegressor` or
        `sklearn.pipeline.Pipeline` with model caching.
        In such cases, the regressor will be re-fitted on the same
        training data as the composite estimator.

    center: bool, default=False
            Whether to center any computed kernels

    fit_inverse_transform: bool, default=False
        Learn the inverse transform for non-precomputed kernels.
        (i.e. learn to find the pre-image of a point)

    tol: float, default=1e-12
        Tolerance for singular values computed by svd_solver == 'arpack'
        and for matrix inversions.
        Must be of range [0.0, infinity).

    n_jobs: int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    iterated_power : int or 'auto', default='auto'
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.
        Must be of range [0, infinity).

    random_state : int, RandomState instance or None, default=None
        Used when the 'arpack' or 'randomized' solvers are used. Pass an int
        for reproducible results across multiple function calls.


    Attributes
    ----------

    pt__: ndarray of size :math:`({n_{components}, n_{components}})`
           pseudo-inverse of the latent-space projection, which
           can be used to contruct projectors from latent-space

    pkt_: ndarray of size :math:`({n_{samples}, n_{components}})`
           the projector, or weights, from the input kernel :math:`\\mathbf{K}`
           to the latent-space projection :math:`\\mathbf{T}`

    pky_: ndarray of size :math:`({n_{samples}, n_{properties}})`
           the projector, or weights, from the input kernel :math:`\\mathbf{K}`
           to the properties :math:`\\mathbf{Y}`

    pty_: ndarray of size :math:`({n_{components}, n_{properties}})`
          the projector, or weights, from the latent-space projection
          :math:`\\mathbf{T}` to the properties :math:`\\mathbf{Y}`

    ptx_: ndarray of size :math:`({n_{components}, n_{features}})`
         the projector, or weights, from the latent-space projection
         :math:`\\mathbf{T}` to the feature matrix :math:`\\mathbf{X}`

    X_fit_: ndarray of shape (n_samples, n_features)
        The data used to fit the model. This attribute is used to build kernels
        from new data.

    Examples
    --------
    >>> import numpy as np
    >>> from skcosmo.decomposition import KernelPCovR
    >>> from skcosmo.preprocessing import StandardFlexibleScaler as SFS
    >>>
    >>> X = np.array([[-1, 1, -3, 1], [1, -2, 1, 2], [-2, 0, -2, -2], [1, 0, 2, -1]])
    >>> X = SFS().fit_transform(X)
    >>> Y = np.array([[ 0, -5], [-1, 1], [1, -5], [-3, 2]])
    >>> Y = SFS(column_wise=True).fit_transform(Y)
    >>>
    >>> kpcovr = KernelPCovR(mixing=0.1, n_components=2, regressor=KernelRidge(kernel='rbf', gamma=2))
    >>> kpcovr.fit(X, Y)
        KernelPCovR(mixing=0.1, n_components=2,
                    regressor=KernelRidge(gamma=2, kernel='rbf'))
    >>> T = kpcovr.transform(X)
        [[-0.55119827, -0.21793572],
         [ 0.3768726 ,  0.31208068],
         [-0.76898956,  0.08511876],
         [ 0.92488574, -0.18627707]]
    >>> Yp = kpcovr.predict(X)
        [[ 0.51713163, -0.99453229],
         [-0.16083953,  0.8378709 ],
         [ 1.18143489, -1.01072628],
         [-1.52011339,  1.13439986]]
    >>> kpcovr.score(X, Y)
        1.0000774522028972
    """

    def __init__(
        self,
        mixing=0.5,
        n_components=None,
        svd_solver="auto",
        regressor=None,
        center=False,
        fit_inverse_transform=False,
        tol=1e-12,
        n_jobs=None,
        iterated_power="auto",
        random_state=None,
    ):

        self.mixing = mixing
        self.n_components = n_components

        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.center = center

        if regressor is None:
            regressor = KernelRidge(
                alpha=1e-6,
                kernel="linear",
                gamma=None,
                degree=3,
                coef0=1,
                kernel_params=None,
            )

        self.regressor = regressor

        self.n_jobs = n_jobs
        self.n_samples_ = None

        self.fit_inverse_transform = fit_inverse_transform

    def _get_kernel(self, X, Y=None):

        try:
            check_is_fitted(self, "regressor_")
            regressor_params = self.regressor_.get_params()

        except NotFittedError:
            regressor_params = self.regressor.get_params()

        if callable(regressor_params["kernel"]):
            params = regressor_params["kernel_params"] or {}
        else:
            default_params = {"gamma": None, "degree": 3, "coef0": 1}
            params = {
                param: regressor_params.get(param, default)
                for param, default in default_params.items()
            }
        return pairwise_kernels(
            X,
            Y,
            metric=regressor_params.get("kernel", "linear"),
            filter_params=True,
            n_jobs=self.n_jobs,
            **params
        )

    def _fit(self, K, Yhat, W):
        """
        Fit the model with the computed kernel and approximated properties.
        """

        K_tilde = pcovr_kernel(mixing=self.mixing, X=K, Y=Yhat, kernel="precomputed")

        if self._fit_svd_solver == "full":
            _, S, Vt = self._decompose_full(K_tilde)
        elif self._fit_svd_solver in ["arpack", "randomized"]:
            _, S, Vt = self._decompose_truncated(K_tilde)
        else:
            raise ValueError(
                "Unrecognized svd_solver='{0}'" "".format(self._fit_svd_solver)
            )

        U = Vt.T

        P = (self.mixing * np.eye(K.shape[0])) + (1.0 - self.mixing) * (W @ Yhat.T)

        S_inv = np.array([1.0 / s if s > self.tol else 0.0 for s in S])

        self.pkt_ = P @ U @ np.sqrt(np.diagflat(S_inv))

        T = K @ self.pkt_
        self.pt__ = np.linalg.lstsq(T, np.eye(T.shape[0]), rcond=self.tol)[0]

    def fit(self, X, Y):
        """

        Fit the model with X and Y.

        Parameters
        ----------
        X:  ndarray, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

            It is suggested that :math:`\\mathbf{X}` be centered by its column-
            means and scaled. If features are related, the matrix should be scaled
            to have unit variance, otherwise :math:`\\mathbf{X}` should be
            scaled so that each feature has a variance of 1 / n_features.

        Y:  ndarray, shape (n_samples, n_properties)
            Training data, where n_samples is the number of samples and
            n_properties is the number of properties

            It is suggested that :math:`\\mathbf{X}` be centered by its column-
            means and scaled. If features are related, the matrix should be scaled
            to have unit variance, otherwise :math:`\\mathbf{Y}` should be
            scaled so that each feature has a variance of 1 / n_features.

        Returns
        -------
        self: object
            Returns the instance itself.

        """

        if not isinstance(self.regressor, KernelRidge):
            raise ValueError("Regressor must be an instance of `KernelRidge`")

        X, Y = check_X_y(X, Y, y_numeric=True, multi_output=True)
        self.X_fit_ = X.copy()

        if self.n_components is None:
            if self.svd_solver != "arpack":
                self.n_components = X.shape[0]
            else:
                self.n_components = X.shape[0] - 1

        K = self._get_kernel(X)

        if self.center:
            self.centerer_ = KernelNormalizer()
            K = self.centerer_.fit_transform(K)

        self.n_samples_ = X.shape[0]

        # Check if regressor is fitted; if not, fit with precomputed K
        # to avoid needing to compute the kernel a second time
        self.regressor_ = check_krr_fit(self.regressor, K, Y)

        W = self.regressor_.dual_coef_.reshape(X.shape[0], -1)

        # Use this instead of `self.regressor_.predict(K)`
        # so that we can handle the case of the pre-fitted regressor
        Yhat = K @ self.regressor_.dual_coef_

        # When we have an unfitted regressor,
        # we fit it with a precomputed K so,
        # we must subsequently "reset" it so that
        # it will work on the particular X
        # of the KPCovR call. The dual coefficients are kept.
        # Can be bypassed if the regressor is pre-fitted.
        try:
            check_is_fitted(self.regressor)

        except NotFittedError:
            self.regressor_.set_params(**self.regressor.get_params())
            self.regressor_.X_fit_ = self.X_fit_
            self.regressor_._check_n_features(self.X_fit_, reset=True)

        # Handle svd_solver
        self._fit_svd_solver = self.svd_solver
        if self._fit_svd_solver == "auto":
            # Small problem or self.n_components == 'mle', just call full PCA
            if max(X.shape) <= 500 or self.n_components == "mle":
                self._fit_svd_solver = "full"
            elif self.n_components >= 1 and self.n_components < 0.8 * min(X.shape):
                self._fit_svd_solver = "randomized"
            # This is also the case of self.n_components in (0,1)
            else:
                self._fit_svd_solver = "full"

        self._fit(K, Yhat, W)

        self.ptk_ = self.pt__ @ K
        self.pty_ = self.pt__ @ Y

        if self.fit_inverse_transform:
            self.ptx_ = self.pt__ @ X

        self.pky_ = self.pkt_ @ self.pty_

        self.components_ = self.pkt_.T  # for sklearn compatibility
        return self

    def predict(self, X=None):
        """Predicts the property values"""

        check_is_fitted(self, ["pky_", "pty_"])

        X = check_array(X)
        K = self._get_kernel(X, self.X_fit_)
        if self.center:
            K = self.centerer_.transform(K)

        return K @ self.pky_

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components as determined by the
        modified Kernel PCovR distances.

        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        """

        check_is_fitted(self, ["pkt_", "X_fit_"])

        X = check_array(X)
        K = self._get_kernel(X, self.X_fit_)

        if self.center:
            K = self.centerer_.transform(K)

        return K @ self.pkt_

    def inverse_transform(self, T):
        """Transform input data back to its original space.

        .. math::

            \\mathbf{\\hat{X}} = \\mathbf{T} \\mathbf{P}_{TX}
                              = \\mathbf{K} \\mathbf{P}_{KT} \\mathbf{P}_{TX}


        Similar to KPCA, the original features are not always recoverable,
        as the projection is computed from the kernel features, not the original
        features, and the mapping between the original and kernel features
        is not one-to-one.

        Parameters
        ----------
        T: ndarray, shape (n_samples, n_components)
            Projected data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original ndarray, shape (n_samples, n_features)
        """

        return T @ self.ptx_

    def score(self, X, Y):
        r"""
        Computes the loss values for KernelPCovR on the given predictor and
        response variables. The loss in :math:`\mathbf{K}`, as explained in
        [Helfrecht2020]_ does not correspond to a traditional Gram loss
        :math:`\mathbf{K} - \mathbf{TT}^T`. Indicating the kernel between set
        A and B as :math:`\mathbf{K}_{AB}`,
        the projection of set A as :math:`\mathbf{T}_A`, and with N and V as the
        train and validation/test set, one obtains

        .. math::

            \ell=\frac{\operatorname{Tr}\left[\mathbf{K}_{VV} - 2
            \mathbf{K}_{VN} \mathbf{T}_N (\mathbf{T}_N^T \mathbf{T}_N)^{-1}  \mathbf{T}_V^T
            +\mathbf{T}_V(\mathbf{T}_N^T \mathbf{T}_N)^{-1}  \mathbf{T}_N^T
            \mathbf{K}_{NN} \mathbf{T}_N (\mathbf{T}_N^T \mathbf{T}_N)^{-1}
            \mathbf{T}_V^T\right]}{\operatorname{Tr}(\mathbf{K}_{VV})}

        Arguments
        ---------
        X:              independent (predictor) variable
        Y:              dependent (response) variable

        Returns
        -------
        Lk:             KPCA loss, determined by the reconstruction of the kernel
        Ly:             KR loss

        """

        check_is_fitted(self, ["pkt_", "X_fit_"])

        X = check_array(X)

        K_NN = self._get_kernel(self.X_fit_, self.X_fit_)
        K_VN = self._get_kernel(X, self.X_fit_)
        K_VV = self._get_kernel(X)

        if self.center:
            K_NN = self.centerer_.transform(K_NN)
            K_VN = self.centerer_.transform(K_VN)
            K_VV = self.centerer_.transform(K_VV)

        y = K_VN @ self.pky_
        Lkrr = np.linalg.norm(Y - y) ** 2 / np.linalg.norm(Y) ** 2

        t_n = K_NN @ self.pkt_
        t_v = K_VN @ self.pkt_

        w = (
            t_n
            @ np.linalg.lstsq(t_n.T @ t_n, np.eye(t_n.shape[1]), rcond=self.tol)[0]
            @ t_v.T
        )
        Lkpca = np.trace(K_VV - 2 * K_VN @ w + w.T @ K_VV @ w) / np.trace(K_VV)

        return sum([Lkpca, Lkrr])

    def _decompose_truncated(self, mat):

        if not 1 <= self.n_components <= self.n_samples_:
            raise ValueError(
                "n_components=%r must be between 1 and "
                "n_samples=%r with "
                "svd_solver='%s'"
                % (
                    self.n_components,
                    self.n_samples_,
                    self.svd_solver,
                )
            )
        elif not isinstance(self.n_components, numbers.Integral):
            raise ValueError(
                "n_components=%r must be of type int "
                "when greater than or equal to 1, was of type=%r"
                % (self.n_components, type(self.n_components))
            )
        elif self.svd_solver == "arpack" and self.n_components == self.n_samples_:
            raise ValueError(
                "n_components=%r must be strictly less than "
                "n_samples=%r with "
                "svd_solver='%s'"
                % (
                    self.n_components,
                    self.n_samples_,
                    self.svd_solver,
                )
            )

        random_state = check_random_state(self.random_state)

        if self._fit_svd_solver == "arpack":
            v0 = _init_arpack_v0(min(mat.shape), random_state)
            U, S, Vt = svds(mat, k=self.n_components, tol=self.tol, v0=v0)
            # svds doesn't abide by scipy.linalg.svd/randomized_svd
            # conventions, so reverse its outputs.
            S = S[::-1]
            # flip eigenvectors' sign to enforce deterministic output
            U, Vt = svd_flip(U[:, ::-1], Vt[::-1])

        # We have already eliminated all other solvers, so this must be "randomized"
        else:
            # sign flipping is done inside
            U, S, Vt = randomized_svd(
                mat,
                n_components=self.n_components,
                n_iter=self.iterated_power,
                flip_sign=True,
                random_state=random_state,
            )

        U[:, S < self.tol] = 0.0
        Vt[S < self.tol] = 0.0
        S[S < self.tol] = 0.0

        return U, S, Vt

    def _decompose_full(self, mat):

        if self.n_components != "mle":
            if not 0 <= self.n_components <= self.n_samples_:
                raise ValueError(
                    "n_components=%r must be between 1 and "
                    "n_samples=%r with "
                    "svd_solver='%s'"
                    % (
                        self.n_components,
                        self.n_samples_,
                        self.svd_solver,
                    )
                )
            elif self.n_components >= 1:
                if not isinstance(self.n_components, numbers.Integral):
                    raise ValueError(
                        "n_components=%r must be of type int "
                        "when greater than or equal to 1, "
                        "was of type=%r" % (self.n_components, type(self.n_components))
                    )

        U, S, Vt = linalg.svd(mat, full_matrices=False)
        U[:, S < self.tol] = 0.0
        Vt[S < self.tol] = 0.0
        S[S < self.tol] = 0.0

        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        # Get variance explained by singular values
        explained_variance_ = (S ** 2) / (self.n_samples_ - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var

        # Postprocess the number of components required
        if self.n_components == "mle":
            self.n_components = _infer_dimension(explained_variance_, self.n_samples_)
        elif 0 < self.n_components < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            # side='right' ensures that number of features selected
            # their variance is always greater than self.n_components float
            # passed. More discussion in issue: #15669
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            self.n_components = (
                np.searchsorted(ratio_cumsum, self.n_components, side="right") + 1
            )
        self.n_components = self.n_components
        return (
            U[:, : self.n_components],
            S[: self.n_components],
            Vt[: self.n_components],
        )
