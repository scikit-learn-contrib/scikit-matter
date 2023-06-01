import numbers
import warnings

import numpy as np
from numpy.linalg import LinAlgError
from scipy import linalg
from scipy.linalg import sqrtm as MatrixSqrt
from scipy.sparse.linalg import svds
from sklearn.decomposition._base import _BasePCA
from sklearn.decomposition._pca import _infer_dimension
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.linear_model._base import LinearModel
from sklearn.utils import check_array, check_random_state
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils.extmath import randomized_svd, stable_cumsum, svd_flip
from sklearn.utils.validation import check_is_fitted, check_X_y

from ..utils import check_lr_fit, pcovr_covariance, pcovr_kernel


class PCovR(_BasePCA, LinearModel):
    r"""

    Principal Covariates Regression, as described in [deJong1992]_
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
        mixing parameter, as described in PCovR as :math:`{\alpha}`, here named
        to avoid confusion with regularization parameter `alpha`

    n_components : int, float or str, default=None
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

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

    tol : float, default=1e-12
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

    space: {'feature', 'sample', 'auto'}, default='auto'
            whether to compute the PCovR in `sample` or `feature` space
            default=`sample` when :math:`{n_{samples} < n_{features}}` and
            `feature` when :math:`{n_{features} < n_{samples}}`

    regressor: {`Ridge`, `RidgeCV`, `LinearRegression`, `precomputed`}, default=None
             regressor for computing approximated :math:`{\mathbf{\hat{Y}}}`.
             The regressor should be one `sklearn.linear_model.Ridge`,
             `sklearn.linear_model.RidgeCV`, or `sklearn.linear_model.LinearRegression`.
             If a pre-fitted regressor is provided, it is used to compute
             :math:`{\mathbf{\hat{Y}}}`.
             Note that any pre-fitting of the regressor will be lost if `PCovR` is
             within a composite estimator that enforces cloning, e.g.,
             `sklearn.compose.TransformedTargetRegressor` or
             `sklearn.pipeline.Pipeline` with model caching.
             In such cases, the regressor will be re-fitted on the same
             training data as the composite estimator.
             If `precomputed`, we assume that the `y` passed to the `fit` function
             is the regressed form of the targets :math:`{\mathbf{\hat{Y}}}`.
             If None, ``sklearn.linear_model.Ridge('alpha':1e-6, 'fit_intercept':False, 'tol':1e-12)``
             is used as the regressor.

    iterated_power : int or 'auto', default='auto'
         Number of iterations for the power method computed by
         svd_solver == 'randomized'.
         Must be of range [0, infinity).

    random_state : int, RandomState instance or None, default=None
         Used when the 'arpack' or 'randomized' solvers are used. Pass an int
         for reproducible results across multiple function calls.

    Attributes
    ----------

    mixing: float, default=0.5
        mixing parameter, as described in PCovR as :math:`{\alpha}`

    tol: float, default=1e-12
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

    space: {'feature', 'sample', 'auto'}, default='auto'
            whether to compute the PCovR in `sample` or `feature` space
            default=`sample` when :math:`{n_{samples} < n_{features}}` and
            `feature` when :math:`{n_{features} < n_{samples}}`

    n_components_ : int
        The estimated number of components, which equals the parameter
        n_components, or the lesser value of n_features and n_samples
        if n_components is None.

    pxt_ : ndarray of size :math:`({n_{samples}, n_{components}})`
           the projector, or weights, from the input space :math:`\mathbf{X}`
           to the latent-space projection :math:`\mathbf{T}`

    pty_ : ndarray of size :math:`({n_{components}, n_{properties}})`
          the projector, or weights, from the latent-space projection
          :math:`\mathbf{T}` to the properties :math:`\mathbf{Y}`

    pxy_ : ndarray of size :math:`({n_{samples}, n_{properties}})`
           the projector, or weights, from the input space :math:`\mathbf{X}`
           to the properties :math:`\mathbf{Y}`

    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.

        Equal to n_components largest eigenvalues
        of the PCovR-modified covariance matrix of :math:`\mathbf{X}`.

    singular_values_ : ndarray of shape (n_components,)
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
    """  # NoQa: E501

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
    ):
        self.mixing = mixing
        self.n_components = n_components
        self.space = space

        self.whiten = False
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

        self.regressor = regressor

    def fit(self, X, Y, W=None):
        r"""

        Fit the model with X and Y. Depending on the dimensions of X,
        calls either `_fit_feature_space` or `_fit_sample_space`

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

            It is suggested that :math:`\mathbf{X}` be centered by its column-
            means and scaled. If features are related, the matrix should be scaled
            to have unit variance, otherwise :math:`\mathbf{X}` should be
            scaled so that each feature has a variance of 1 / n_features.

        Y : ndarray, shape (n_samples, n_properties)
            Training data, where n_samples is the number of samples and
            n_properties is the number of properties

            It is suggested that :math:`\mathbf{X}` be centered by its column-
            means and scaled. If features are related, the matrix should be scaled
            to have unit variance, otherwise :math:`\mathbf{Y}` should be
            scaled so that each feature has a variance of 1 / n_features.

            If the passed regressor = `precomputed`, it is assumed that Y is the
            regressed form of the properties, :math:`{\mathbf{\hat{Y}}}`.

        W : ndarray, shape (n_features, n_properties)
            Regression weights, optional when regressor=`precomputed`. If not
            passed, it is assumed that `W = np.linalg.lstsq(X, Y, self.tol)[0]`

        """

        X, Y = check_X_y(X, Y, y_numeric=True, multi_output=True)

        # saved for inverse transformations from the latent space,
        # should be zero in the case that the features have been properly centered
        self.mean_ = np.mean(X, axis=0)

        if np.max(np.abs(self.mean_)) > self.tol:
            warnings.warn(
                "This class does not automatically center data, and your data mean is"
                " greater than the supplied tolerance.",
                stacklevel=1,
            )

        if self.space is not None and self.space not in [
            "feature",
            "sample",
            "auto",
        ]:
            raise ValueError("Only feature and sample space are supported.")

        # Handle self.n_components==None
        if self.n_components is None:
            if self.svd_solver != "arpack":
                self.n_components_ = min(X.shape)
            else:
                self.n_components_ = min(X.shape) - 1
        else:
            self.n_components_ = self.n_components

        if not any(
            [
                self.regressor is None,
                self.regressor == "precomputed",
                isinstance(self.regressor, LinearRegression),
                isinstance(self.regressor, Ridge),
                isinstance(self.regressor, RidgeCV),
            ]
        ):
            raise ValueError(
                "Regressor must be an instance of "
                "`LinearRegression`, `Ridge`, `RidgeCV`, or `precomputed`"
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

            self.regressor_ = check_lr_fit(regressor, X, y=Y)

            W = self.regressor_.coef_.T.reshape(X.shape[1], -1)
            Yhat = self.regressor_.predict(X).reshape(X.shape[0], -1)
        else:
            Yhat = Y.copy()
            if W is None:
                W = np.linalg.lstsq(X, Yhat, self.tol)[0]

        # Handle svd_solver
        self.fit_svd_solver_ = self.svd_solver
        if self.fit_svd_solver_ == "auto":
            # Small problem or self.n_components_ == 'mle', just call full PCA
            if max(X.shape) <= 500 or self.n_components_ == "mle":
                self.fit_svd_solver_ = "full"
            elif self.n_components_ >= 1 and self.n_components_ < 0.8 * min(X.shape):
                self.fit_svd_solver_ = "randomized"
            # This is also the case of self.n_components_ in (0,1)
            else:
                self.fit_svd_solver_ = "full"

        self.n_samples_in_, self.n_features_in_ = X.shape
        self.space_ = self.space
        if self.space_ is None or self.space_ == "auto":
            if self.n_samples_in_ > self.n_features_in_:
                self.space_ = "feature"
            else:
                self.space_ = "sample"

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
        r"""
        In feature-space PCovR, the projectors are determined by:

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

        Ct, iCsqrt = pcovr_covariance(
            mixing=self.mixing,
            X=X,
            Y=Yhat,
            rcond=self.tol,
            return_isqrt=True,
        )
        try:
            Csqrt = np.linalg.lstsq(iCsqrt, np.eye(len(iCsqrt)), rcond=None)[0]

        # if we can avoid recomputing Csqrt, we should, but sometimes we
        # run into a singular matrix, which is what we do here
        except LinAlgError:
            Csqrt = np.real(MatrixSqrt(X.T @ X))

        if self.fit_svd_solver_ == "full":
            U, S, Vt = self._decompose_full(Ct)
        elif self.fit_svd_solver_ in ["arpack", "randomized"]:
            U, S, Vt = self._decompose_truncated(Ct)
        else:
            raise ValueError(
                "Unrecognized svd_solver='{0}'" "".format(self.fit_svd_solver_)
            )

        self.singular_values_ = np.sqrt(S.copy())
        self.explained_variance_ = S / (X.shape[0] - 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / self.explained_variance_.sum()
        )

        S_sqrt = np.diagflat([np.sqrt(s) if s > self.tol else 0.0 for s in S])
        S_sqrt_inv = np.diagflat([1.0 / np.sqrt(s) if s > self.tol else 0.0 for s in S])
        self.pxt_ = np.linalg.multi_dot([iCsqrt, Vt.T, S_sqrt])
        self.ptx_ = np.linalg.multi_dot([S_sqrt_inv, Vt, Csqrt])
        self.pty_ = np.linalg.multi_dot([S_sqrt_inv, Vt, iCsqrt, X.T, Y])

    def _fit_sample_space(self, X, Y, Yhat, W):
        r"""
        In sample-space PCovR, the projectors are determined by:

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

        Kt = pcovr_kernel(mixing=self.mixing, X=X, Y=Yhat)

        if self.fit_svd_solver_ == "full":
            U, S, Vt = self._decompose_full(Kt)
        elif self.fit_svd_solver_ in ["arpack", "randomized"]:
            U, S, Vt = self._decompose_truncated(Kt)
        else:
            raise ValueError(
                "Unrecognized svd_solver='{0}'" "".format(self.fit_svd_solver_)
            )

        self.singular_values_ = np.sqrt(S.copy())
        self.explained_variance_ = S / (X.shape[0] - 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / self.explained_variance_.sum()
        )

        P = (self.mixing * X.T) + (1.0 - self.mixing) * W @ Yhat.T
        S_sqrt_inv = np.diagflat([1.0 / np.sqrt(s) if s > self.tol else 0.0 for s in S])
        T = Vt.T @ S_sqrt_inv

        self.pxt_ = P @ T
        self.pty_ = T.T @ Y
        self.ptx_ = T.T @ X

    def _decompose_truncated(self, mat):
        if not 1 <= self.n_components_ <= min(self.n_samples_in_, self.n_features_in_):
            raise ValueError(
                "n_components=%r must be between 1 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    self.n_components_,
                    min(self.n_samples_in_, self.n_features_in_),
                    self.svd_solver,
                )
            )
        elif not isinstance(self.n_components_, numbers.Integral):
            raise ValueError(
                "n_components=%r must be of type int "
                "when greater than or equal to 1, was of type=%r"
                % (self.n_components_, type(self.n_components_))
            )
        elif self.svd_solver == "arpack" and self.n_components_ == min(
            self.n_samples_in_, self.n_features_in_
        ):
            raise ValueError(
                "n_components=%r must be strictly less than "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    self.n_components_,
                    min(self.n_samples_in_, self.n_features_in_),
                    self.svd_solver,
                )
            )

        random_state = check_random_state(self.random_state)

        if self.fit_svd_solver_ == "arpack":
            v0 = _init_arpack_v0(min(mat.shape), random_state)
            U, S, Vt = svds(mat, k=self.n_components_, tol=self.tol, v0=v0)
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
                n_components=self.n_components_,
                n_iter=self.iterated_power,
                flip_sign=True,
                random_state=random_state,
            )

        return U, S, Vt

    def _decompose_full(self, mat):
        if self.n_components_ == "mle":
            if self.n_samples_in_ < self.n_features_in_:
                raise ValueError(
                    "n_components='mle' is only supported " "if n_samples >= n_features"
                )
        elif (
            not 0 <= self.n_components_ <= min(self.n_samples_in_, self.n_features_in_)
        ):
            raise ValueError(
                "n_components=%r must be between 1 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    self.n_components_,
                    min(self.n_samples_in_, self.n_features_in_),
                    self.svd_solver,
                )
            )
        elif self.n_components_ >= 1:
            if not isinstance(self.n_components_, numbers.Integral):
                raise ValueError(
                    "n_components=%r must be of type int "
                    "when greater than or equal to 1, "
                    "was of type=%r" % (self.n_components_, type(self.n_components_))
                )

        U, S, Vt = linalg.svd(mat, full_matrices=False)

        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        # Get variance explained by singular values
        explained_variance_ = S / (self.n_samples_in_ - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var

        # Postprocess the number of components required
        if self.n_components_ == "mle":
            self.n_components_ = _infer_dimension(
                explained_variance_, self.n_samples_in_
            )
        elif 0 < self.n_components_ < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            # side='right' ensures that number of features selected
            # their variance is always greater than self.n_components_ float
            # passed. More discussion in issue: #15669
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            self.n_components_ = (
                np.searchsorted(ratio_cumsum, self.n_components_, side="right") + 1
            )
        return (
            U[:, : self.n_components_],
            S[: self.n_components_],
            Vt[: self.n_components_],
        )

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

        if np.max(np.abs(self.mean_)) > self.tol:
            warnings.warn(
                "This class does not automatically un-center data, and your data mean "
                "is greater than the supplied tolerance, so the inverse transformation "
                "will be off by the original data mean.",
                stacklevel=1,
            )

        return T @ self.ptx_

    def predict(self, X=None, T=None):
        """Predicts the property values using regression on X or T"""

        check_is_fitted(self, ["pxy_", "pty_"])

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            X = check_array(X)
            return X @ self.pxy_
        else:
            T = check_array(T)
            return T @ self.pty_

    def transform(self, X=None):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components as determined by the
        modified PCovR distances.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        """

        check_is_fitted(self, ["pxt_", "mean_"])

        return super().transform(X)

    def score(self, X, Y, T=None):
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
        X : ndarray of shape (n_samples, n_features)
            The data.

        Y : ndarray of shape (n_samples, n_properties)
            The target.

        Returns
        -------
        loss : float
             Negative sum of the loss in reconstructing X from the latent-space
             projection T and the loss in predicting Y from the latent-space
             projection T
        """

        if T is None:
            T = self.transform(X)

        x = self.inverse_transform(T)
        y = self.predict(T=T)

        return -(
            np.linalg.norm(X - x) ** 2.0 / np.linalg.norm(X) ** 2.0
            + np.linalg.norm(Y - y) ** 2.0 / np.linalg.norm(Y) ** 2.0
        )
