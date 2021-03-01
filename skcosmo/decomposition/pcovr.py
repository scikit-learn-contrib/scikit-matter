import numpy as np
from numpy.linalg import LinAlgError
import numbers

from scipy import linalg
from scipy.sparse.linalg import svds
from scipy.linalg import sqrtm as MatrixSqrt

from sklearn.linear_model import Ridge as LR
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd, svd_flip
from sklearn.utils.extmath import stable_cumsum
from sklearn.decomposition._base import _BasePCA
from sklearn.linear_model._base import LinearModel

from sklearn.decomposition._pca import _infer_dimension
from sklearn.utils._arpack import _init_arpack_v0
from skcosmo.utils import pcovr_covariance, pcovr_kernel


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
    results will change drastically near :math:`\alpha \to 0` and :math:`\alpha \to 0`.
    This can be done with the companion preprocessing classes, where

    >>> from skcosmo.preprocessing import StandardFlexibleScaler as SFS
    >>>
    >>> # Set column_wise to True when the columns are relative to one another,
    >>> # False otherwise.
    >>> scaler = SFS(column_wise=True)
    >>>
    >>> scaler.fit(A) # replace with your matrix
    >>> A = scaler.transform(A)

    Parameters
    ----------
    mixing: float, defaults to 1
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

    tol : float, default=0.0
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

    space: {'feature', 'sample', 'auto'}, default='auto'
            whether to compute the PCovR in `sample` or `feature` space
            defaults to `sample` when :math:`{n_{samples} < n_{features}}` and
            `feature` when :math:`{n_{features} < n_{samples}}`

    alpha: float, default=1E-6
            Regularization parameter to use in all regression operations.
            Defaults to alpha included in `lr_args`, or if none is specified, 1E-6.

    estimator:
             estimator for computing approximated :math:`{\mathbf{\hat{Y}}}`,
             default = `sklearn.linear_model.Ridge('alpha':1e-6, 'fit_intercept':False, 'tol':1e-12`)

    iterated_power : int or 'auto', default='auto'
         Number of iterations for the power method computed by
         svd_solver == 'randomized'.
         Must be of range [0, infinity).

    random_state : int, RandomState instance or None, default=None
         Used when the 'arpack' or 'randomized' solvers are used. Pass an int
         for reproducible results across multiple function calls.

    Attributes
    ----------

    mixing: float, defaults to 1
        mixing parameter, as described in PCovR as :math:`{\alpha}`

    alpha: float, default=1E-6
            Regularization parameter to use in all regression operations.

    tol: float, default=0.0
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

    space: {'feature', 'sample', 'auto'}, default='auto'
            whether to compute the PCovR in `sample` or `feature` space
            defaults to `sample` when :math:`{n_{samples} < n_{features}}` and
            `feature` when :math:`{n_{features} < n_{samples}}`

    n_components : int
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
    >>> from skcosmo.decomposition import PCovR
    >>> X = np.array([[-1, 1, -3, 1], [1, -2, 1, 2], [-2, 0, -2, -2], [1, 0, 2, -1]])
    >>> Y = np.array([[ 0, -5], [-1, 1], [1, -5], [-3, 2]])
    >>> pcovr = PCovR(mixing=0.1, n_components=2)
    >>> pcovr.fit(X, Y)
    PCovR(alpha=1e-06, mixing=0.1, n_components=2, space='sample')
    >>> T = pcovr.transform(X)
        [[ 3.2630561 ,  0.06663787],
         [-2.69395511, -0.41582771],
         [ 3.48683147, -0.83164387],
         [-4.05593245,  1.18083371]]
    >>> Yp = pcovr.predict(X)
        [[ 0.01374656, -5.00943466],
         [-1.02804032,  1.06737777],
         [ 0.98167556, -4.9830631 ],
         [-2.99627428,  1.98241962]]
    """

    def __init__(
        self,
        mixing=0.0,
        n_components=None,
        svd_solver="auto",
        alpha=None,
        tol=1e-12,
        space="auto",
        estimator=LR(alpha=1e-6, fit_intercept=False, tol=1e-12),
        iterated_power="auto",
        random_state=None,
    ):

        self.mixing = mixing
        self.n_components = n_components
        self.alpha = alpha
        self.space = space

        self.whiten = False
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

        self.estimator = estimator

        if alpha is None:
            self.alpha = getattr(self.estimator, "alpha", 1e-6)

    def fit(self, X, Y, Yhat=None, W=None):
        r"""

        Fit the model with X and Y. Depending on the dimensions of X,
        calls either `_fit_feature_space` or `_fit_sample_space`

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

            It is suggested that :math:`\mathbf{X}` be centered by its column-
            means and scaled. If features are related, the matrix should be scaled
            to have unit variance, otherwise :math:`\mathbf{X}` should be
            scaled so that each feature has a variance of 1 / n_features.

        Y : array-like, shape (n_samples, n_properties)
            Training data, where n_samples is the number of samples and
            n_properties is the number of properties

            It is suggested that :math:`\mathbf{X}` be centered by its column-
            means and scaled. If features are related, the matrix should be scaled
            to have unit variance, otherwise :math:`\mathbf{Y}` should be
            scaled so that each feature has a variance of 1 / n_features.

        Yhat : array-like, shape (n_samples, n_properties), optional
            Regressed training data, where n_samples is the number of samples and
            n_properties is the number of properties. If not supplied, computed
            by ridge regression.
        W : array-like, shape (n_features, n_properties), optional
            Weights of regressed training data. If not supplied, computed
            by ridge regression.

        """

        X, Y = check_X_y(X, Y, y_numeric=True, multi_output=True)

        # saved for inverse transformations from the latent space,
        # should be zero in the case that the features have been properly centered
        self.mean_ = np.mean(X, axis=0)

        if self.space is not None and self.space not in [
            "feature",
            "sample",
            "auto",
        ]:
            raise ValueError("Only feature and sample space are supported.")

        # Handle self.n_components==None
        if self.n_components is None:
            if self.svd_solver != "arpack":
                self.n_components = min(X.shape)
            else:
                self.n_components = min(X.shape) - 1

        if W is None:
            self.estimator.fit(X, Y)
            W = self.estimator.coef_.T.reshape(X.shape[1], -1)

        if Yhat is None:
            Yhat = self.estimator.predict(X).reshape(X.shape[0], -1)

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

        self.n_samples, self.n_features = X.shape
        if self.space is None or self.space == "auto":
            if self.n_samples > self.n_features:
                self.space = "feature"
            else:
                self.space = "sample"

        if self.space == "feature":
            self._fit_feature_space(X, Y.reshape(Yhat.shape), Yhat, W)
        else:
            self._fit_sample_space(X, Y.reshape(Yhat.shape), Yhat, W)

        self.pxy_ = self.pxt_ @ self.pty_
        if len(Y.shape) == 1:
            self.pxy_ = self.pxy_.reshape(
                X.shape[1],
            )
            self.pty_ = self.pty_.reshape(
                self.n_components,
            )

        self.components_ = self.pxt_.T  # for sklearn compatibility
        return self

    def _fit_feature_space(self, X, Y, Yhat, W=None):
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
            Csqrt = np.linalg.inv(iCsqrt)

        # if we can avoid recomputing Csqrt, we should, but sometimes we
        # run into a singular matrix, which is what we do here
        except LinAlgError:
            Csqrt = np.real(MatrixSqrt(X.T @ X))

        if self._fit_svd_solver == "full":
            U, S, Vt = self._decompose_full(Ct)
        elif self._fit_svd_solver in ["arpack", "randomized"]:
            U, S, Vt = self._decompose_truncated(Ct)
        else:
            raise ValueError(
                "Unrecognized svd_solver='{0}'" "".format(self._fit_svd_solver)
            )

        self.singular_values_ = S.copy()
        self.explained_variance_ = (S ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / self.explained_variance_.sum()
        )

        S_inv = np.diagflat(
            [1.0 / s if s > self.tol else 0.0 for s in S]
        )
        self.pxt_ = np.linalg.multi_dot([iCsqrt, Vt.T, np.diagflat(S)])
        self.ptx_ = np.linalg.multi_dot([S_inv, Vt, Csqrt])
        self.pty_ = np.linalg.multi_dot([S_inv, Vt, iCsqrt, X.T, Y])

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

        if self._fit_svd_solver == "full":
            U, S, Vt = self._decompose_full(Kt)
        elif self._fit_svd_solver in ["arpack", "randomized"]:
            U, S, Vt = self._decompose_truncated(Kt)
        else:
            raise ValueError(
                "Unrecognized svd_solver='{0}'" "".format(self._fit_svd_solver)
            )

        self.singular_values_ = S.copy()
        self.explained_variance_ = (S ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / self.explained_variance_.sum()
        )

        P = (self.mixing * X.T) + (1.0 - self.mixing) * np.dot(W, Yhat.T)
        T = Vt.T @ np.diagflat(1 / np.sqrt(S))

        self.pxt_ = P @ T
        self.pty_ = T.T @ Y
        self.ptx_ = T.T @ X

    def _decompose_truncated(self, mat):

        if not 1 <= self.n_components <= min(self.n_samples, self.n_features):
            raise ValueError(
                "n_components=%r must be between 1 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    self.n_components,
                    min(self.n_samples, self.n_features),
                    self.svd_solver,
                )
            )
        elif not isinstance(self.n_components, numbers.Integral):
            raise ValueError(
                "n_components=%r must be of type int "
                "when greater than or equal to 1, was of type=%r"
                % (self.n_components, type(self.n_components))
            )
        elif self.svd_solver == "arpack" and self.n_components == min(
            self.n_samples, self.n_features
        ):
            raise ValueError(
                "n_components=%r must be strictly less than "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    self.n_components,
                    min(self.n_samples, self.n_features),
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

        return U, S, Vt

    def _decompose_full(self, mat):
        if self.n_components == "mle":
            if self.n_samples < self.n_features:
                raise ValueError(
                    "n_components='mle' is only supported " "if n_samples >= n_features"
                )
        elif not 0 <= self.n_components <= min(self.n_samples, self.n_features):
            raise ValueError(
                "n_components=%r must be between 1 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    self.n_components,
                    min(self.n_samples, self.n_features),
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

        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        # Get variance explained by singular values
        explained_variance_ = (S ** 2) / (self.n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var

        # Postprocess the number of components required
        if self.n_components == "mle":
            self.n_components = _infer_dimension(explained_variance_, self.n_samples)
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

    def inverse_transform(self, T):
        r"""Transform data back to its original space.

        .. math::

            \mathbf{\hat{X}} = \mathbf{T} \mathbf{P}_{TX}
                              = \mathbf{X} \mathbf{P}_{XT} \mathbf{P}_{TX}


        Parameters
        ----------
        T : array-like, shape (n_samples, n_components)
            Projected data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)
        """

        return T @ self.ptx_ + self.mean_

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
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        """

        check_is_fitted(self, ["pxt_", "mean_"])

        return super().transform(X)

    def score(self, X, Y, T=None):
        r"""Return the total reconstruction error for X and Y,
        defined as:

        .. math::

            \ell_{X} = \frac{\lVert \mathbf{X} - \mathbf{T}\mathbf{P}_{TX} \rVert ^ 2}{\lVert \mathbf{X}\rVert ^ 2}

        and

        .. math::

            \ell_{Y} = \frac{\lVert \mathbf{Y} - \mathbf{T}\mathbf{P}_{TY} \rVert ^ 2}{\lVert \mathbf{Y}\rVert ^ 2}


        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.

        Y : array-like of shape (n_samples, n_properties)
            The target.

        Returns
        -------
        lx : float
             Loss in reconstructing X from the latent-space projection T
        ly : float
             Loss in predicting Y from the latent-space projection T
        """

        if T is None:
            T = self.transform(X)

        x = self.inverse_transform(T)
        y = self.predict(T=T)

        return (
            np.linalg.norm(X - x) ** 2.0 / np.linalg.norm(X) ** 2.0
            + np.linalg.norm(Y - y) ** 2.0 / np.linalg.norm(Y) ** 2.0
        )
