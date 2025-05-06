import numpy as np

from sklearn.base import check_array
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.utils.validation import check_is_fitted, validate_data

from skmatter.decomposition import _BasePCov
from skmatter.utils import check_lr_fit

class PCovR(_BasePCov):
    r"""Principal Covariates Regression, as described in [deJong1992]_
    determines a latent-space projection :math:`\mathbf{T}` which
    minimizes a combined loss in supervised and unsupervised tasks.

    This projection is determined by the eigendecomposition of a modified gram
    matrix :math:`\mathbf{\tilde{K}}`
    .. math::
      \mathbf{\tilde{K}} = \alpha \mathbf{X} \mathbf{X}^T +
            (1 - \alpha) \mathbf{\hat{Y}}\mathbf{\hat{Y}}^T

    where :math:`\alpha` is a mixing parameter and
    :math:`\mathbf{X}` and :math:`\mathbf{\hat{Y}}` are matrices of shapes
    :math:`(n_{samples}, n_{features})` and :math:`(n_{samples}, n_{properties})`,
    respectively, which contain the input and approximate targets. For
    :math:`(n_{samples} < n_{features})`, this can be more efficiently computed
    using the eigendecomposition of a modified covariance matrix
    :math:`\mathbf{\tilde{C}}`

    .. math::
      \mathbf{\tilde{C}} = \alpha \mathbf{X}^T \mathbf{X} +
            (1 - \alpha) \left(\left(\mathbf{X}^T
            \mathbf{X}\right)^{-\frac{1}{2}} \mathbf{X}^T
            \mathbf{\hat{Y}}\mathbf{\hat{Y}}^T \mathbf{X} \left(\mathbf{X}^T
            \mathbf{X}\right)^{-\frac{1}{2}}\right)

    For all PCovR methods, it is strongly suggested that :math:`\mathbf{X}` and
    :math:`\mathbf{Y}` are centered and scaled to unit variance, otherwise the
    results will change drastically near :math:`\alpha \to 0` and :math:`\alpha \to 1`.
    This can be done with the companion preprocessing classes, where

    >>> from skmatter.preprocessing import StandardFlexibleScaler as SFS
    >>> import numpy as np
    >>>
    >>> # Set column_wise to True when the columns are relative to one another,
    >>> # False otherwise.
    >>> scaler = SFS(column_wise=True)
    >>>
    >>> A = np.array([[1, 2], [2, 1]])  # replace with your matrix
    >>> scaler.fit(A)
    StandardFlexibleScaler(column_wise=True)
    >>> A = scaler.transform(A)

    Parameters
    ----------
    mixing: float, default=0.5
        mixing parameter, as described in PCovR as :math:`{\alpha}`, here named to avoid
        confusion with regularization parameter `alpha`

    n_components : int, float or str, default=None
        Number of components to keep.
        if n_components is not set all components are kept::
            n_components == min(n_samples, n_features)

    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        If auto :
            The solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the number of
            components to extract is lower than 80% of the smallest dimension of the
            data, then the more efficient 'randomized' method is enabled. Otherwise the
            exact full SVD is computed and optionally truncated afterwards.
        If full :
            run exact full SVD calling the standard LAPACK solver via `scipy.linalg.svd`
            and select the components by postprocessing
        If arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly 0 < n_components <
            min(X.shape)
        If randomized :
            run randomized SVD by the method of Halko et al.

    tol : float, default=1e-12
        Tolerance for singular values computed by svd_solver == 'arpack'. Must be of
        range [0.0, infinity).

    space: {'feature', 'sample', 'auto'}, default='auto'
        whether to compute the PCovR in `sample` or `feature` space default=`sample`
        when :math:`{n_{samples} < n_{features}}` and `feature` when
        :math:`{n_{features} < n_{samples}}`

    regressor: {`Ridge`, `RidgeCV`, `LinearRegression`, `precomputed`}, default=None
        regressor for computing approximated :math:`{\mathbf{\hat{Y}}}`. The regressor
        should be one `sklearn.linear_model.Ridge`, `sklearn.linear_model.RidgeCV`, or
        `sklearn.linear_model.LinearRegression`. If a pre-fitted regressor is provided,
        it is used to compute :math:`{\mathbf{\hat{Y}}}`. Note that any pre-fitting of
        the regressor will be lost if `PCovR` is within a composite estimator that
        enforces cloning, e.g., `sklearn.compose.TransformedTargetRegressor` or
        `sklearn.pipeline.Pipeline` with model caching. In such cases, the regressor
        will be re-fitted on the same training data as the composite estimator. If
        `precomputed`, we assume that the `y` passed to the `fit` function is the
        regressed form of the targets :math:`{\mathbf{\hat{Y}}}`. If None,
        ``sklearn.linear_model.Ridge('alpha':1e-6, 'fit_intercept':False, 'tol':1e-12)``
        is used as the regressor.

    iterated_power : int or 'auto', default='auto'
         Number of iterations for the power method computed by svd_solver ==
         'randomized'. Must be of range [0, infinity).

    random_state : int, :class:`numpy.random.RandomState` instance or None, default=None
         Used when the 'arpack' or 'randomized' solvers are used. Pass an int for
         reproducible results across multiple function calls.

    whiten : boolean, deprecated

    Attributes
    ----------
    mixing: float, default=0.5
        mixing parameter, as described in PCovR as :math:`{\alpha}`

    tol: float, default=1e-12
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

    space: {'feature', 'sample', 'auto'}, default='auto'
        whether to compute the PCovR in `sample` or `feature` space default=`sample`
        when :math:`{n_{samples} < n_{features}}` and `feature` when
        :math:`{n_{features} < n_{samples}}`

    n_components_ : int
        The estimated number of components, which equals the parameter n_components, or
        the lesser value of n_features and n_samples if n_components is None.

    pxt_ : numpy.ndarray of size :math:`({n_{samples}, n_{components}})`
        the projector, or weights, from the input space :math:`\mathbf{X}` to the
        latent-space projection :math:`\mathbf{T}`

    pty_ : numpy.ndarray of size :math:`({n_{components}, n_{properties}})`
        the projector, or weights, from the latent-space projection :math:`\mathbf{T}`
        to the properties :math:`\mathbf{Y}`

    pxy_ : numpy.ndarray of size :math:`({n_{samples}, n_{properties}})`
        the projector, or weights, from the input space :math:`\mathbf{X}` to the
        properties :math:`\mathbf{Y}`

    explained_variance_ : numpy.ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.
        Equal to n_components largest eigenvalues
        of the PCovR-modified covariance matrix of :math:`\mathbf{X}`.

    singular_values_ : numpy.ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.

    Examples
    --------
    >>> import numpy as np
    >>> from skmatter.decomposition import PCovR
    >>> X = np.array([[-1, 1, -3, 1], [1, -2, 1, 2], [-2, 0, -2, -2], [1, 0, 2, -1]])
    >>> Y = np.array([[0, -5], [-1, 1], [1, -5], [-3, 2]])
    >>> pcovr = PCovR(mixing=0.1, n_components=2)
    >>> pcovr.fit(X, Y)
    PCovR(mixing=0.1, n_components=2)
    >>> pcovr.transform(X)
    array([[ 3.2630561 ,  0.06663787],
           [-2.69395511, -0.41582771],
           [ 3.48683147, -0.83164387],
           [-4.05593245,  1.18083371]])
    >>> pcovr.predict(X)
    array([[ 0.01371776, -5.00945512],
           [-1.02805338,  1.06736871],
           [ 0.98166504, -4.98307078],
           [-2.9963189 ,  1.98238856]])
    """

    def __init__(
        self,
        mixing=0.5,
        n_components=None,
        svd_solver="auto",
        tol=1e-12,
        space="auto",
        regressor=None,
        iterated_power="auto",
        random_state=None,
        whiten=False,
    ):
        super().__init__(
            mixing=mixing,
            n_components=n_components,
            svd_solver=svd_solver,
            tol=tol,
            space=space,
            iterated_power=iterated_power,
            random_state=random_state,
            whiten=whiten,
        )
        self.regressor = regressor

    def fit(self, X, Y, W=None):
        r"""Fit the model with X and Y. Depending on the dimensions of X, calls either
        `_fit_feature_space` or `_fit_sample_space`

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is
            the number of features.

            It is suggested that :math:`\mathbf{X}` be centered by its column-
            means and scaled. If features are related, the matrix should be scaled
            to have unit variance, otherwise :math:`\mathbf{X}` should be
            scaled so that each feature has a variance of 1 / n_features.

        Y : numpy.ndarray, shape (n_samples, n_properties)
            Training data, where n_samples is the number of samples and n_properties is
            the number of properties

            It is suggested that :math:`\mathbf{X}` be centered by its column- means and
            scaled. If features are related, the matrix should be scaled to have unit
            variance, otherwise :math:`\mathbf{Y}` should be scaled so that each feature
            has a variance of 1 / n_features.

            If the passed regressor = `precomputed`, it is assumed that Y is the
            regressed form of the properties, :math:`{\mathbf{\hat{Y}}}`.

        W : numpy.ndarray, shape (n_features, n_properties)
            Regression weights, optional when regressor=`precomputed`. If not
            passed, it is assumed that `W = np.linalg.lstsq(X, Y, self.tol)[0]`
        """
        X, Y = validate_data(self, X, Y, y_numeric=True, multi_output=True)
        super()._fit_utils(X, Y)

        compatible_regressors = (LinearRegression, Ridge, RidgeCV)

        if self.regressor not in ["precomputed", None] and not isinstance(
            self.regressor, compatible_regressors
        ):
            raise ValueError(
                "Regressor must be an instance of `"
                f"{'`, `'.join(r.__name__ for r in compatible_regressors)}`"
                ", or `precomputed`"
            )

        # Assign the default regressor
        if self.regressor != "precomputed":
            if self.regressor is None:
                regressor = Ridge(
                    alpha=1e-6,
                    fit_intercept=False,
                    tol=1e-12,
                )
            else:
                regressor = self.regressor

            self.regressor_ = check_lr_fit(regressor, X, Y)

            W = self.regressor_.coef_.T.reshape(X.shape[1], -1)
            Yhat = self.regressor_.predict(X).reshape(X.shape[0], -1)
        else:
            Yhat = Y.copy()
            if W is None:
                W = np.linalg.lstsq(X, Yhat, self.tol)[0]

        if self.space_ == "feature":
            self._fit_feature_space(X, Y.reshape(Yhat.shape), Yhat)
        else:
            self._fit_sample_space(X, Y.reshape(Yhat.shape), Yhat, W)

        self.pxy_ = self.pxt_ @ self.pty_
        if len(Y.shape) == 1:
            self.pxy_ = self.pxy_.reshape(
                X.shape[1],
            )
            self.pty_ = self.pty_.reshape(
                self.n_components_,
            )

        self.components_ = self.pxt_.T  # for sklearn compatibility
        return self

    def _fit_feature_space(self, X, Y, Yhat):
        r"""In feature-space PCovR, the projectors are determined by:

        .. math::
            \mathbf{\tilde{C}} = \alpha \mathbf{X}^T \mathbf{X} +
            (1 - \alpha) \left(\left(\mathbf{X}^T
            \mathbf{X}\right)^{-\frac{1}{2}} \mathbf{X}^T
            \mathbf{\hat{Y}}\mathbf{\hat{Y}}^T \mathbf{X} \left(\mathbf{X}^T
            \mathbf{X}\right)^{-\frac{1}{2}}\right)

        where

        .. math::
            \mathbf{P}_{XT} = (\mathbf{X}^T \mathbf{X})^{-\frac{1}{2}}
                                \mathbf{U}_\mathbf{\tilde{C}}^T
                                \mathbf{\Lambda}_\mathbf{\tilde{C}}^{\frac{1}{2}}

        .. math::
            \mathbf{P}_{TX} = \mathbf{\Lambda}_\mathbf{\tilde{C}}^{-\frac{1}{2}}
                                \mathbf{U}_\mathbf{\tilde{C}}^T
                                (\mathbf{X}^T \mathbf{X})^{\frac{1}{2}}

        .. math::
            \mathbf{P}_{TY} = \mathbf{\Lambda}_\mathbf{\tilde{C}}^{-\frac{1}{2}}
                               \mathbf{U}_\mathbf{\tilde{C}}^T (\mathbf{X}^T
                               \mathbf{X})^{-\frac{1}{2}} \mathbf{X}^T
                               \mathbf{Y}
        """
        return super()._fit_feature_space(X, Y, Yhat)

    def _fit_sample_space(self, X, Y, Yhat, W):
        r"""In sample-space PCovR, the projectors are determined by:

        .. math::
            \mathbf{\tilde{K}} = \alpha \mathbf{X} \mathbf{X}^T +
            (1 - \alpha) \mathbf{\hat{Y}}\mathbf{\hat{Y}}^T

        where

        .. math::
            \mathbf{P}_{XT} = \left(\alpha \mathbf{X}^T + (1 - \alpha)
                               \mathbf{W} \mathbf{\hat{Y}}^T\right)
                               \mathbf{U}_\mathbf{\tilde{K}}
                               \mathbf{\Lambda}_\mathbf{\tilde{K}}^{-\frac{1}{2}}

        .. math::
            \mathbf{P}_{TX} = \mathbf{\Lambda}_\mathbf{\tilde{K}}^{-\frac{1}{2}}
                                \mathbf{U}_\mathbf{\tilde{K}}^T \mathbf{X}

        .. math::
            \mathbf{P}_{TY} = \mathbf{\Lambda}_\mathbf{\tilde{K}}^{-\frac{1}{2}}
                               \mathbf{U}_\mathbf{\tilde{K}}^T \mathbf{Y}
        """
        return super()._fit_sample_space(X, Y, Yhat, W)

    def inverse_transform(self, T):
        r"""Transform data back to its original space.

        .. math::
            \mathbf{\hat{X}} = \mathbf{T} \mathbf{P}_{TX}
                              = \mathbf{X} \mathbf{P}_{XT} \mathbf{P}_{TX}

        Parameters
        ----------
        T : ndarray, shape (n_samples, n_components)
            Projected data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original ndarray, shape (n_samples, n_features)
        """
        return super().inverse_transform(T)

    def predict(self, X=None, T=None):
        """Predicts the property values using regression on X or T."""
        check_is_fitted(self, ["pxy_", "pty_"])

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            X = validate_data(self, X, reset=False)
            return X @ self.pxy_
        else:
            T = check_array(T)
            return T @ self.pty_

    def transform(self, X=None):
        """Apply dimensionality reduction to X.

        ``X`` is projected on the first principal components as determined by the
        modified PCovR distances.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        """
        return super().transform(X)

    def score(self, X, y, T=None):
        r"""Return the (negative) total reconstruction error for X and Y,
        defined as:

        .. math::
            \ell_{X} = \frac{\lVert \mathbf{X} - \mathbf{T}\mathbf{P}_{TX} \rVert ^ 2}
                            {\lVert \mathbf{X}\rVert ^ 2}

        and

        .. math::
            \ell_{Y} = \frac{\lVert \mathbf{Y} - \mathbf{T}\mathbf{P}_{TY} \rVert ^ 2}
                            {\lVert \mathbf{Y}\rVert ^ 2}

        The negative loss :math:`-\ell = -(\ell_{X} + \ell{Y})` is returned for easier
        use in sklearn pipelines, e.g., a grid search, where methods named 'score' are
        meant to be maximized.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The data.
        Y : numpy.ndarray of shape (n_samples, n_properties)
            The target.

        Returns
        -------
        loss : float
            Negative sum of the loss in reconstructing X from the latent-space
            projection T and the loss in predicting Y from the latent-space projection T
        """
        X, y = validate_data(self, X, y, reset=False)

        if T is None:
            T = self.transform(X)

        Xrec = self.inverse_transform(T)
        ypred = self.predict(T=T)

        return -(
            np.linalg.norm(X - Xrec) ** 2.0 / np.linalg.norm(X) ** 2.0
            + np.linalg.norm(y - ypred) ** 2.0 / np.linalg.norm(y) ** 2.0
        )
