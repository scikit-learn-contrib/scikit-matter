import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.validation import _check_n_features, check_is_fitted, validate_data

from skmatter.utils import check_krr_fit
from skmatter.decomposition import _BaseKPCov


class KernelPCovR(_BaseKPCov):
    r"""Kernel Principal Covariates Regression, as described in [Helfrecht2020]_
    determines a latent-space projection :math:`\mathbf{T}` which minimizes a combined
    loss in supervised and unsupervised tasks in the reproducing kernel Hilbert space
    (RKHS).

    This projection is determined by the eigendecomposition of a modified gram matrix
    :math:`\mathbf{\tilde{K}}`

    .. math::
      \mathbf{\tilde{K}} = \alpha \mathbf{K} +
            (1 - \alpha) \mathbf{\hat{Y}}\mathbf{\hat{Y}}^T

    where :math:`\alpha` is a mixing parameter,
    :math:`\mathbf{K}` is the input kernel of shape :math:`(n_{samples}, n_{samples})`
    and :math:`\mathbf{\hat{Y}}` is the target matrix of shape
    :math:`(n_{samples}, n_{properties})`.

    Parameters
    ----------
    mixing : float, default=0.5
        mixing parameter, as described in PCovR as :math:`{\alpha}`
        
    n_components : int, float or str, default=None
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

    regressor : {instance of `sklearn.kernel_ridge.KernelRidge`, `precomputed`, None}, default=None
        The regressor to use for computing
        the property predictions :math:`\hat{\mathbf{Y}}`.
        A pre-fitted regressor may be provided.
        If the regressor is not `None`, its kernel parameters
        (`kernel`, `gamma`, `degree`, `coef0`, and `kernel_params`)
        must be identical to those passed directly to `KernelPCovR`.

        If `precomputed`, we assume that the `y` passed to the `fit` function
        is the regressed form of the targets :math:`{\mathbf{\hat{Y}}}`.

    kernel : {"linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"}, default="linear"
        Kernel.

    gamma : float, default=None
        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
        kernels.

    degree : int, default=3
        Degree for poly kernels. Ignored by other kernels.

    coef0 : float, default=1
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of str to any, default=None
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.

    center : bool, default=False
        Whether to center any computed kernels

    fit_inverse_transform : bool, default=False
        Learn the inverse transform for non-precomputed kernels.
        (i.e. learn to find the pre-image of a point)

    tol : float, default=1e-12
        Tolerance for singular values computed by svd_solver == 'arpack'
        and for matrix inversions.
        Must be of range [0.0, infinity).

    n_jobs : int, default=None
        The number of parallel jobs to run.
        :obj:`None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    iterated_power : int or 'auto', default='auto'
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.
        Must be of range [0, infinity).

    random_state : int, :class:`numpy.random.RandomState` instance or None, default=None
        Used when the 'arpack' or 'randomized' solvers are used. Pass an int
        for reproducible results across multiple function calls.

    Attributes
    ----------
    pt__: numpy.darray of size :math:`({n_{components}, n_{components}})`
        pseudo-inverse of the latent-space projection, which
        can be used to contruct projectors from latent-space

    pkt_: numpy.ndarray of size :math:`({n_{samples}, n_{components}})`
        the projector, or weights, from the input kernel :math:`\mathbf{K}`
        to the latent-space projection :math:`\mathbf{T}`

    pky_: numpy.ndarray of size :math:`({n_{samples}, n_{properties}})`
        the projector, or weights, from the input kernel :math:`\mathbf{K}`
        to the properties :math:`\mathbf{Y}`

    pty_: numpy.ndarray of size :math:`({n_{components}, n_{properties}})`
        the projector, or weights, from the latent-space projection
        :math:`\mathbf{T}` to the properties :math:`\mathbf{Y}`

    ptx_: numpy.ndarray of size :math:`({n_{components}, n_{features}})`
        the projector, or weights, from the latent-space projection
        :math:`\mathbf{T}` to the feature matrix :math:`\mathbf{X}`

    X_fit_: numpy.ndarray of shape (n_samples, n_features)
        The data used to fit the model. This attribute is used to build kernels
        from new data.

    Examples
    --------
    >>> import numpy as np
    >>> from skmatter.decomposition import KernelPCovR
    >>> from skmatter.preprocessing import StandardFlexibleScaler as SFS
    >>> from sklearn.kernel_ridge import KernelRidge
    >>>
    >>> X = np.array([[-1, 1, -3, 1], [1, -2, 1, 2], [-2, 0, -2, -2], [1, 0, 2, -1]])
    >>> X = SFS().fit_transform(X)
    >>> Y = np.array([[0, -5], [-1, 1], [1, -5], [-3, 2]])
    >>> Y = SFS(column_wise=True).fit_transform(Y)
    >>>
    >>> kpcovr = KernelPCovR(
    ...     mixing=0.1,
    ...     n_components=2,
    ...     regressor=KernelRidge(kernel="rbf", gamma=1),
    ...     kernel="rbf",
    ...     gamma=1,
    ... )
    >>> kpcovr.fit(X, Y)
    KernelPCovR(gamma=1, kernel='rbf', mixing=0.1, n_components=2,
                regressor=KernelRidge(gamma=1, kernel='rbf'))
    >>> kpcovr.transform(X)
    array([[-0.61261285, -0.18937908],
           [ 0.45242098,  0.25453465],
           [-0.77871824,  0.04847559],
           [ 0.91186937, -0.21211816]])
    >>> kpcovr.predict(X)
    array([[ 0.5100212 , -0.99488463],
           [-0.18992219,  0.82064368],
           [ 1.11923584, -1.04798016],
           [-1.5635827 ,  1.11078662]])
    >>> round(kpcovr.score(X, Y), 5)
    np.float64(-0.52039)
    """  # NoQa: E501

    def __init__(
        self,
        mixing=0.5,
        n_components=None,
        svd_solver="auto",
        regressor=None,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        center=False,
        fit_inverse_transform=False,
        tol=1e-12,
        n_jobs=None,
        iterated_power="auto",
        random_state=None,
    ):
        super().__init__(
            mixing=mixing,
            n_components=n_components,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
            center=center,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            n_jobs=n_jobs,
            fit_inverse_transform=fit_inverse_transform,
        )
        self.regressor = regressor

    def fit(self, X, Y, W=None):
        r"""Fit the model with X and Y.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

            It is suggested that :math:`\mathbf{X}` be centered by its column-
            means and scaled. If features are related, the matrix should be scaled
            to have unit variance, otherwise :math:`\mathbf{X}` should be
            scaled so that each feature has a variance of 1 / n_features.
        Y : numpy.ndarray, shape (n_samples, n_properties)
            Training data, where n_samples is the number of samples and
            n_properties is the number of properties

            It is suggested that :math:`\mathbf{X}` be centered by its column-
            means and scaled. If features are related, the matrix should be scaled
            to have unit variance, otherwise :math:`\mathbf{Y}` should be
            scaled so that each feature has a variance of 1 / n_features.
        W : numpy.ndarray, shape (n_samples, n_properties)
            Regression weights, optional when regressor=`precomputed`. If not
            passed, it is assumed that `W = np.linalg.lstsq(K, Y, self.tol)[0]`

        Returns
        -------
        self: object
            Returns the instance itself.
        """
        X, Y = validate_data(self, X, Y, y_numeric=True, multi_output=True)

        K = super()._fit_utils(X)

        if self.regressor not in ["precomputed", None] and not isinstance(
            self.regressor, KernelRidge
        ):
            raise ValueError("Regressor must be an instance of `KernelRidge`")

        if self.regressor != "precomputed":
            if self.regressor is None:
                regressor = KernelRidge(
                    kernel=self.kernel,
                    gamma=self.gamma,
                    degree=self.degree,
                    coef0=self.coef0,
                    kernel_params=self.kernel_params,
                )
            else:
                regressor = self.regressor
                kernel_attrs = ["kernel", "gamma", "degree", "coef0", "kernel_params"]
                if not all(
                    [
                        getattr(self, attr) == getattr(regressor, attr)
                        for attr in kernel_attrs
                    ]
                ):
                    raise ValueError(
                        "Kernel parameter mismatch: the regressor has kernel "
                        "parameters {%s} and KernelPCovR was initialized with kernel "
                        "parameters {%s}"
                        % (
                            ", ".join(
                                [
                                    "%s: %r" % (attr, getattr(regressor, attr))
                                    for attr in kernel_attrs
                                ]
                            ),
                            ", ".join(
                                [
                                    "%s: %r" % (attr, getattr(self, attr))
                                    for attr in kernel_attrs
                                ]
                            ),
                        )
                    )

            # Check if regressor is fitted; if not, fit with precomputed K
            # to avoid needing to compute the kernel a second time
            self.regressor_ = check_krr_fit(regressor, K, X, Y)
            W = self.regressor_.dual_coef_.reshape(self.n_samples_in_, -1)

            # Use this instead of `self.regressor_.predict(K)`
            # so that we can handle the case of the pre-fitted regressor
            Yhat = K @ W

            # When we have an unfitted regressor,
            # we fit it with a precomputed K
            # so we must subsequently "reset" it so that
            # it will work on the particular X
            # of the KPCovR call. The dual coefficients are kept.
            # Can be bypassed if the regressor is pre-fitted.
            try:
                check_is_fitted(regressor)
            except NotFittedError:
                self.regressor_.set_params(**regressor.get_params())
                self.regressor_.X_fit_ = self.X_fit_
                _check_n_features(self.regressor_, self.X_fit_, reset=True)
        else:
            Yhat = Y.copy()
            if W is None:
                W = np.linalg.lstsq(K, Yhat, self.tol)[0]

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

        X = validate_data(self, X, reset=False)
        K = self._get_kernel(X, self.X_fit_)
        if self.center:
            K = self.centerer_.transform(K)

        return K @ self.pky_

    def transform(self, X):
        """Apply dimensionality reduction to X.

        ``X`` is projected on the first principal components as determined by the
        modified Kernel PCovR distances.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        """
        return super().transform(X)

    def inverse_transform(self, T):
        r"""Transform input data back to its original space.

        .. math::
            \mathbf{\hat{X}} = \mathbf{T} \mathbf{P}_{TX}
                              = \mathbf{K} \mathbf{P}_{KT} \mathbf{P}_{TX}

        Similar to KPCA, the original features are not always recoverable,
        as the projection is computed from the kernel features, not the original
        features, and the mapping between the original and kernel features
        is not one-to-one.

        Parameters
        ----------
        T : numpy.ndarray, shape (n_samples, n_components)
            Projected data, where n_samples is the number of samples and n_components is
            the number of components.

        Returns
        -------
        X_original : numpy.ndarray, shape (n_samples, n_features)
        """
        return super().inverse_transform(T)

    def score(self, X, y):
        r"""Computes the (negative) loss values for KernelPCovR on the given predictor
        and response variables. The loss in :math:`\mathbf{K}`, as explained in
        [Helfrecht2020]_ does not correspond to a traditional Gram loss
        :math:`\mathbf{K} - \mathbf{TT}^T`. Indicating the kernel between set A and B as
        :math:`\mathbf{K}_{AB}`, the projection of set A as :math:`\mathbf{T}_A`, and
        with N and V as the train and validation/test set, one obtains

        .. math::
            \ell=\frac{\operatorname{Tr}\left[\mathbf{K}_{VV} - 2
            \mathbf{K}_{VN} \mathbf{T}_N
                (\mathbf{T}_N^T \mathbf{T}_N)^{-1} \mathbf{T}_V^T
            +\mathbf{T}_V(\mathbf{T}_N^T \mathbf{T}_N)^{-1}  \mathbf{T}_N^T
            \mathbf{K}_{NN} \mathbf{T}_N (\mathbf{T}_N^T \mathbf{T}_N)^{-1}
            \mathbf{T}_V^T\right]}{\operatorname{Tr}(\mathbf{K}_{VV})}

        The negative loss is returned for easier use in sklearn pipelines, e.g., a grid
        search, where methods named 'score' are meant to be maximized.

        Parameters
        ----------
        X : numpy.ndarray
            independent (predictor) variable
        Y : numpy.ndarray
            dependent (response) variable

        Returns
        -------
        L : float
            Negative sum of the KPCA and KRR losses, with the KPCA loss determined by
            the reconstruction of the kernel
        """
        check_is_fitted(self, ["pkt_", "X_fit_"])

        X = validate_data(self, X, reset=False)

        K_NN = self._get_kernel(self.X_fit_, self.X_fit_)
        K_VN = self._get_kernel(X, self.X_fit_)
        K_VV = self._get_kernel(X)

        if self.center:
            K_NN = self.centerer_.transform(K_NN)
            K_VN = self.centerer_.transform(K_VN)
            K_VV = self.centerer_.transform(K_VV)

        ypred = K_VN @ self.pky_
        Lkrr = np.linalg.norm(y - ypred) ** 2 / np.linalg.norm(y) ** 2

        t_n = K_NN @ self.pkt_
        t_v = K_VN @ self.pkt_

        w = (
            t_n
            @ np.linalg.lstsq(t_n.T @ t_n, np.eye(t_n.shape[1]), rcond=self.tol)[0]
            @ t_v.T
        )
        Lkpca = np.trace(K_VV - 2 * K_VN @ w + w.T @ K_VV @ w) / np.trace(K_VV)

        return -sum([Lkpca, Lkrr])
