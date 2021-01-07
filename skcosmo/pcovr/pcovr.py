import numpy as np
from numpy.linalg import LinAlgError
import scipy
from scipy.linalg import sqrtm as MatrixSqrt
from sklearn.linear_model import Ridge as LR
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils import check_array
from sklearn.decomposition._base import _BasePCA
from sklearn.linear_model._base import LinearModel

from skcosmo.utils import eig_solver


def pcovr_covariance(mixing, X, Y, rcond=1e-12, return_isqrt=False):
    r"""
    Creates the PCovR modified covariance

    .. math::

        \mathbf{\tilde{C}} = \alpha \mathbf{X}^T \mathbf{X} +
        (1 - \alpha) \left(\left(\mathbf{X}^T
        \mathbf{X}\right)^{-\frac{1}{2}} \mathbf{X}^T
        \mathbf{\hat{Y}}\mathbf{\hat{Y}}^T \mathbf{X} \left(\mathbf{X}^T
        \mathbf{X}\right)^{-\frac{1}{2}}\right)

    where :math:`\mathbf{\hat{Y}}`` are the properties obtained by linear regression.

    :param mixing: mixing parameter,
                   as described in PCovR as :math:`{\alpha}`, defaults to 1
    :type mixing: float

    :param X: Data matrix :math:`\mathbf{X}`
    :type X: array of shape (n x m)

    :param Y: array to include in biased selection when mixing < 1
    :type Y: array of shape (n x p)

    :param rcond: threshold below which eigenvalues will be considered 0,
                      defaults to 1E-12
    :type rcond: float

    """

    C = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)

    cov = X.T @ X

    if mixing < 1 or return_isqrt:
        # Do not try to approximate C_inv, it will affect results
        C_inv = np.linalg.pinv(cov, rcond=rcond)
        C_isqrt = np.real(scipy.linalg.sqrtm(C_inv))

        # parentheses speed up calculation greatly
        C_Y = C_isqrt @ (X.T @ Y)
        C_Y = C_Y.reshape((C.shape[0], -1))
        C_Y = np.real(C_Y)

        C += (1 - mixing) * C_Y @ C_Y.T

    if mixing > 0:
        C += (mixing) * cov

    if return_isqrt:
        return C, C_isqrt
    else:
        return C


def pcovr_kernel(mixing, X, Y):
    r"""
    Creates the PCovR modified kernel distances

    .. math::

        \mathbf{\tilde{K}} = \alpha \mathbf{X} \mathbf{X}^T +
        (1 - \alpha) \mathbf{\hat{Y}}\mathbf{\hat{Y}}^T

    :param mixing: mixing parameter,
                   as described in PCovR as :math:`{\alpha}`, defaults to 1
    :type mixing: float

    :param X: Data matrix :math:`\mathbf{X}`
    :type X: array of shape (n x m)

    :param Y: array to include in biased selection when mixing < 1
    :type Y: array of shape (n x p)

    """

    K = np.zeros((X.shape[0], X.shape[0]))
    if mixing < 1:
        K += (1 - mixing) * Y @ Y.T
    if mixing > 0:
        K += (mixing) * X @ X.T

    return K


class PCovR(_BasePCA, LinearModel):
    r"""
    Performs Principal Covariates Regression, as described in `[S. de Jong and
    H. A. L. Kiers, 1992] <https://doi.org/10.1016/0169-7439(92)80100-I>`_,
    implemented according to the notations and strategies covered in
    `[Helfrecht, et al., 2020]
    <https://iopscience.iop.org/article/10.1088/2632-2153/aba9ef>`_.

    Parameters
    ----------
    mixing: float, defaults to 1
        mixing parameter, as described in PCovR as :math:`{\alpha}`

    n_components : int, float or str, default=None
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

    tol : float, default=0.0
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

    space: {'feature', 'structure', 'auto'}, default='auto'
            whether to compute the PCovR in `structure` or `feature` space
            defaults to `structure` when :math:`{n_{samples} < n_{features}}` and
            `feature` when :math:`{n_{features} < n_{samples}}`

    regularization: float, default=1E-6
            Regularization parameter to use in all regression operations.
            Defaults to regularization included in `lr_args`, or if none is specified, 1E-6.

    estimator:
             estimator for computing approximated :math:`{\mathbf{\hat{Y}}}`,
             default = `sklearn.linear_model.Ridge('alpha':1e-6, 'fit_intercept':False, 'tol':1e-12`)

    Attributes
    ----------

    mixing: float, defaults to 1
        mixing parameter, as described in PCovR as :math:`{\alpha}`

    regularization: float, default=1E-6
            Regularization parameter to use in all regression operations.

    tol: float, default=0.0
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

    space: {'feature', 'structure', 'auto'}, default='auto'
            whether to compute the PCovR in `structure` or `feature` space
            defaults to `structure` when :math:`{n_{samples} < n_{features}}` and
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

    References
    ----------
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015
        3.  B. A. Helfrecht, R. K. Cersonsky, G. Fraux, and M. Ceriotti,
            'Structure-property maps with Kernel principal covariates regression',
            Machine Learning: Science and Technology 1(4):045021, 2020

    Examples
    --------
    >>> import numpy as np
    >>> from skcosmo.pcovr import PCovR
    >>> X = np.array([[-1, 1, -3, 1], [1, -2, 1, 2], [-2, 0, -2, -2], [1, 0, 2, -1]])
    >>> Y = np.array([[ 0, -5], [-1, 1], [1, -5], [-3, 2]])
    >>> pcovr = PCovR(mixing=0.1, n_components=2)
    >>> pcovr.fit(X)
    PCovR(lr_args=None, mixing=0.1, n_components=2, space=None, tol=None)
    >>> T = pcovr.transform(X)
        [[-2.48017109 -1.54378072]
         [ 2.74724894 -1.67904456]
         [-2.56778727  1.44091806]
         [ 2.30070942  1.78190722]]
    >>> Yp = pcovr.predict(X)
        [[ 0.03200724, -5.01754987],
         [-1.05781286,  1.14547488],
         [ 0.95713937, -4.96456897],
         [-2.99288563,  1.95975794]]
    """

    def __init__(
        self,
        mixing=0.0,
        n_components=None,
        *,
        regularization=None,
        tol=1e-12,
        space="auto",
        estimator=LR(alpha=1e-6, fit_intercept=False, tol=1e-12),
    ):

        self.mixing = mixing
        self.n_components = n_components
        self.regularization = regularization
        self.tol = tol
        self.space = space

        self.whiten = False
        self.estimator = estimator

        if regularization is None:
            self.regularization = getattr(self.estimator, "alpha", 1e-6)

    def fit(self, X, Y, Yhat=None, W=None):
        r"""

        Fit the model with X and Y. Depending on the dimensions of X,
        calls either `_fit_feature_space` or `_fit_structure_space`

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
            "structure",
            "auto",
        ]:
            raise ValueError("Only feature and structure space are supported.")

        if self.n_components is None:
            self.n_components = min(X.shape)

        if W is None:
            self.estimator.fit(X, Y)
            W = self.estimator.coef_.T.reshape(X.shape[1], -1)

        if Yhat is None:
            Yhat = self.estimator.predict(X).reshape(X.shape[0], -1)

        if self.space is None or self.space == "auto":
            if X.shape[0] > X.shape[1]:
                self.space = "feature"
            else:
                self.space = "structure"

        if self.space == "feature":
            self._fit_feature_space(X, Yhat, W)
        else:
            self._fit_structure_space(X, Yhat, W)

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

    def _fit_feature_space(self, X, Yhat, W=None):
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
        except LinAlgError:
            Csqrt = np.real(MatrixSqrt(X.T @ X))

        v, U = eig_solver(
            Ct, n_components=self.n_components, tol=self.tol, add_null=True
        )
        S = v ** 0.5
        S_inv = np.linalg.pinv(np.diagflat(S))

        self.singular_values_ = S.copy()
        self.explained_variance_ = (S ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / self.explained_variance_.sum()
        )

        self.pxt_ = np.linalg.multi_dot([iCsqrt, U, np.diagflat(S)])
        self.ptx_ = np.linalg.multi_dot([S_inv, U.T, Csqrt])
        self.pty_ = np.linalg.multi_dot([S_inv, U.T, iCsqrt, X.T, Yhat])

    def _fit_structure_space(self, X, Yhat, W):
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

        v, U = eig_solver(
            Kt, n_components=self.n_components, tol=self.tol, add_null=True
        )
        S = v ** 0.5

        P = (self.mixing * X.T) + (1.0 - self.mixing) * np.dot(W, Yhat.T)

        self.singular_values_ = S.copy()
        self.explained_variance_ = (S ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / self.explained_variance_.sum()
        )

        T = U @ np.diagflat(1 / S)

        self.pxt_ = P @ T
        self.pty_ = T.T @ Yhat
        self.ptx_ = T.T @ X

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
        r"""Return the normalized, squared reconstruction error for X and Y,
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
            np.linalg.norm(X - x) ** 2.0 / np.linalg.norm(X) ** 2.0,
            np.linalg.norm(Y - y) ** 2.0 / np.linalg.norm(Y) ** 2.0,
        )
