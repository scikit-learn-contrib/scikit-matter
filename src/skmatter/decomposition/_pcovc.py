'''
Option 1:
Base PCov Class (contains all shared methods (same name) between PCovR and PCovC)
- contains options for implementation depending on sub class type
1. PCovR extends PCov 
2. PCovC extends PCov (will contain some unique methods such as decision_function)

This would prevent us from having to update all PCovR instances in examples, docs, etc 
(since external method names and variables would remain the same).

Bse KPCov Class (contains all shared methods (same name)) between KPCovR and KPCovC) 
- contains options for implementation depending on sub class type
1. KPCovR extends PCov 
2. KPCovC extends PCov 

This would prevent us from having to update all KPCovR instances in examples, docs, etc.
Benefit of doing this would be that users can clearly see the differences between PCovR and PCovC 
(how implementation differs just so slightly in base class)

sklearn RidgeRegression / RidgeClassifier implementation has _BaseRidge as a private class.
They have _BaseRidge
1. Ridge Regression extends _BaseRidge
2. Ridge Classifier extends _BaseRidge

They have _BaseRidgeCV (uses grid search CV)
1. Ridge RegressionCV extends _BaseRidgeCV
2. Ridge ClassifierCV extends _BaseRidgeCV

Kernel Ridge Regression is separate.

Option 2:
Simply have PCovC extend PCovR and override several methods (might lead to some redundancy)
'''

import numbers
import warnings

import numpy as np
from numpy.linalg import LinAlgError
from scipy import linalg
from scipy.linalg import sqrtm as MatrixSqrt
from scipy.sparse.linalg import svds
from sklearn.decomposition._base import _BasePCA
from sklearn.decomposition._pca import _infer_dimension
from sklearn.linear_model import (
    RidgeClassifier,
    RidgeClassifierCV,
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier,
)
from sklearn.linear_model._base import LinearModel
from sklearn.utils import check_array, check_random_state, column_or_1d
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils.extmath import randomized_svd, stable_cumsum, svd_flip
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC

from skmatter.utils import pcovr_covariance, pcovr_kernel
from sklearn.utils._array_api import get_namespace, indexing_dtype
from copy import deepcopy

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_is_fitted

from sklearn.multioutput import MultiOutputClassifier

def check_cl_fit(classifier, X, y):
    r"""
    Checks that a (linear) classifier is fitted, and if not,
    fits it with the provided data
    :param regressor: sklearn-style classifier
    :type classifier: object
    :param X: feature matrix with which to fit the classifier
        if it is not already fitted
    :type X: array
    :param y: target values with which to fit the classifier
        if it is not already fitted
    :type y: array
    """
    try:
        check_is_fitted(classifier)
        fitted_classifier = deepcopy(classifier)

        # Check compatibility with X
        fitted_classifier._validate_data(X, y, reset=False, multi_output=True)
        print("X shape "+str(X.shape))
        print("y shape " + str(y.shape))
        # Check compatibility with y

        # changed from if fitted_classifier.coef_.ndim != y.ndim:
        # dimension of classifier coefficients is always 2, hence we don't need to check 
        # for match with Y
        if fitted_classifier.coef_.shape[1] != X.shape[1]:
            raise ValueError(
                "The classifier coefficients have a shape incompatible "
                "with the supplied feature space. "
                "The coefficients have shape %d and the features "
                "have shape %d" % (fitted_classifier.coef_.shape, X.shape)
            )
        # LogisticRegression does not support multioutput, but RidgeClassifier does
        elif y.ndim == 2:
            if fitted_classifier.coef_.shape[0] != y.shape[1]:
                raise ValueError(
                    "The classifier coefficients have a shape incompatible "
                    "with the supplied target space. "
                    "The coefficients have shape %r and the targets "
                    "have shape %r" % (fitted_classifier.coef_.shape, y.shape)
                )

    except NotFittedError:
        fitted_classifier = clone(classifier)
        fitted_classifier.fit(X, y)

    return fitted_classifier


class PCovC(_BasePCA, LinearModel):
    r"""
    Principal Covariates Classification.
    Determines a latent-space projection :math:`\mathbf{T}` which
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
    classifier: {`Ridge`, `RidgeCV`, `LinearRegression`, `precomputed`}, default=None
             classifier for computing approximated :math:`{\mathbf{\hat{Y}}}`.
             The classifier should be one `sklearn.linear_model.Ridge`,
             `sklearn.linear_model.RidgeCV`, or `sklearn.linear_model.LinearRegression`.
             If a pre-fitted classifier is provided, it is used to compute
             :math:`{\mathbf{\hat{Y}}}`.
             Note that any pre-fitting of the classifier will be lost if `PCovR` is
             within a composite estimator that enforces cloning, e.g.,
             `sklearn.compose.TransformedTargetclassifier` or
             `sklearn.pipeline.Pipeline` with model caching.
             In such cases, the classifier will be re-fitted on the same
             training data as the composite estimator.
             If `precomputed`, we assume that the `y` passed to the `fit` function
             is the regressed form of the targets :math:`{\mathbf{\hat{Y}}}`.
             If None, ``sklearn.linear_model.Ridge('alpha':1e-6, 'fit_intercept':False, 'tol':1e-12)``
             is used as the classifier.
    iterated_power : int or 'auto', default='auto'
         Number of iterations for the power method computed by
         svd_solver == 'randomized'.
         Must be of range [0, infinity).
    random_state : int, RandomState instance or None, default=None
         Used when the 'arpack' or 'randomized' solvers are used. Pass an int
         for reproducible results across multiple function calls.
    whiten : boolean, deprecated

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
        classifier=None,
        iterated_power="auto",
        random_state=None,
        whiten=False,
    ):
        self.mixing = mixing
        self.n_components = n_components
        self.space = space

        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

        self.classifier = classifier

    def fit(self, X, y, W=None):
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
            If the passed classifier = `precomputed`, it is assumed that Y is the
            regressed form of the properties, :math:`{\mathbf{\hat{Y}}}`.
        W : ndarray, shape (n_features, n_properties)
            Regression weights, optional when classifier=`precomputed`. If not
            passed, it is assumed that `W = np.linalg.lstsq(X, Y, self.tol)[0]`
        """
        X, y = check_X_y(X, y, multi_output=True)

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
                self.classifier is None,
                self.classifier == "precomputed",
                isinstance(
                    self.classifier,
                    (
                        RidgeClassifier,
                        RidgeClassifierCV,
                        LogisticRegression,
                        LogisticRegressionCV,
                        SGDClassifier,
                        LinearSVC,
                        MultiOutputClassifier,
                    ),
                ),
            ]
        ):
            raise ValueError(
                "classifier must be an instance of "
                "`RidgeClassifier`, `RidgeClassifierCV`, `LogisticRegression`,"
                "`Logistic RegressionCV`, or `precomputed`"
            )
        
        # Assign the default classifier
        if self.classifier != "precomputed":
            if self.classifier is None:
                classifier = LogisticRegression()
            else:
                classifier = self.classifier

            z_classifier_ = check_cl_fit(classifier, X, y=y)  #change to z classifier, fits linear classifier on x and y to get Pxz

            if isinstance(z_classifier_, MultiOutputClassifier):
                W = np.hstack([est_.coef_.T for est_ in z_classifier_.estimators_])
                Z = X @ W #computes Z, basically Z=XPxz

            else:
                W = z_classifier_.coef_.T.reshape(X.shape[1], -1)
                Z = z_classifier_.decision_function(X).reshape(X.shape[0], -1) #computes Z

        else:
            Z = y.copy()
            if W is None:
                W = np.linalg.lstsq(X, Z, self.tol)[0]  #W = weights for Pxz

        self._label_binarizer = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self._label_binarizer.fit_transform(y)
        if not self._label_binarizer.y_type_.startswith("multilabel"):
            y = column_or_1d(y, warn=True)

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
            self._fit_feature_space(X, Y.reshape(Z.shape), Z)
        else:
            self._fit_sample_space(X, Y.reshape(Z.shape), Z, W)

        # instead of using linear regression solution, refit with the classifier
        # and steal weights to get ptz
        # this is failing because self.classifier is never changed from None if None is passed as classifier
        # change self.classifier to classifier and see what happens. if classifier is precomputed, there might be more errors so be careful.
        # if classifier is precomputed, I don't think we need to check if the classifier is fit or not?

        #cases:
        #1. if classifier has been fit with X and Y already, we dont need to perform a check_cl_fit
        #2. if classifier has not been fit with X or Y, we dont need to 
        #3. if classifier has been fit with T and Y, we need to perform check_cl_fit

        # old: self.classifier_ = check_cl_fit(self.classifier, X @ self.pxt_, y=y) #Has Ptz as weights 

        self.classifier_ = check_cl_fit(classifier, X @ self.pxt_, y=y)

        #self.classifier_ = LogisticRegression().fit(X @ self.pxt_, y)
        #check_cl_fit(classifier., X @ self.pxt_, y=y) #Has Ptz as weights 
        print("Self.classifier_ shape "+ str(self.classifier_.coef_.shape))
        print("PCovC Self.pxt_ "+ str((self.pxt_).shape))

        if isinstance(self.classifier_, MultiOutputClassifier):
            self.ptz_ = np.hstack(
                [est_.coef_.T for est_ in self.classifier_.estimators_]
            )
            self.pxz_ = self.pxt_ @ self.ptz_
        else:
            self.ptz_ = self.classifier_.coef_.T #self.ptz_ = self.classifier_.coef.T
            self.pxz_ = self.pxt_ @ self.ptz_ #self.pxz_ = self.pxt_ @ self.ptz_

        if len(Y.shape) == 1:
            self.pxz_ = self.pxz_.reshape(
                X.shape[1],
            )
            self.ptz_ = self.ptz_.reshape(
                self.n_components_,
            )

        self.components_ = self.pxt_.T  # for sklearn compatibility
        return self

    def _fit_feature_space(self, X, Y, Z):
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
            Y=Z,
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
        # self.pty_ = np.linalg.multi_dot([S_sqrt_inv, Vt, iCsqrt, X.T, Y])

    def _fit_sample_space(self, X, Y, Z, W):
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

        Kt = pcovr_kernel(mixing=self.mixing, X=X, Y=Z)

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

        P = (self.mixing * X.T) + (1.0 - self.mixing) * W @ Z.T
        S_sqrt_inv = np.diagflat([1.0 / np.sqrt(s) if s > self.tol else 0.0 for s in S])
        T = Vt.T @ S_sqrt_inv

        self.pxt_ = P @ T
        # self.pty_ = T.T @ Y
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

    def decision_function(self, X=None, T=None):
        """Predicts confidence score from X or T."""

        check_is_fitted(self, attributes=["_label_binarizer", "pxz_", "ptz_"])

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            X = check_array(X)
            return X @ self.pxz_
        else:
            T = check_array(T)
            return T @ self.ptz_

    def predict(self, X=None, T=None):
        """Predicts class labels from X or T."""
        check_is_fitted(self, attributes=["_label_binarizer", "pxz_", "ptz_"])

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        # multiclass = self._label_binarizer.y_type_.startswith("multiclass")

        if X is not None:
            return self.classifier_.predict(X @ self.pxt_) #Ptz(T) -> activation -> Y labels
            # xp, _ = get_namespace(X)
            # scores = self.decision_function(X=X)
            # if multiclass:
            #     indices = xp.argmax(scores, axis=1)
            # else:
            #     indices = xp.astype(scores > 0, indexing_dtype(xp))
            # return xp.take(self.classes_, indices, axis=0)

        else:
            return self.classifier_.predict(T) #Ptz(T) -> activation -> Y labels
            # tp, _ = get_namespace(T)
            # scores = self.decision_function(T=T)
            # if multiclass:
            #     indices = tp.argmax(scores, axis=1)
            # else:
            #     indices = tp.astype(scores > 0, indexing_dtype(tp))
            # return tp.take(self.classes_, indices, axis=0)

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
        y = self.decision_function(T=T)

        return -(
            np.linalg.norm(X - x) ** 2.0 / np.linalg.norm(X) ** 2.0
            + np.linalg.norm(Y - y) ** 2.0 / np.linalg.norm(Y) ** 2.0
        )

    @property
    def classes_(self):
        return self._label_binarizer.classes_