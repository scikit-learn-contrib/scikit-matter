import numpy as np

from sklearn import clone
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import (
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV,
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier,
)
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.utils.multiclass import check_classification_targets, type_of_target

from skmatter.utils import check_cl_fit
from skmatter.decomposition import _BaseKPCov


class KernelPCovC(LinearClassifierMixin, _BaseKPCov):
    r"""Kernel Principal Covariates Classification
    determines a latent-space projection :math:`\mathbf{T}` which minimizes a combined
    loss in supervised and unsupervised tasks in the reproducing kernel Hilbert space
    (RKHS).

    This projection is determined by the eigendecomposition of a modified gram matrix
    :math:`\mathbf{\tilde{K}}`

    .. math::
      \mathbf{\tilde{K}} = \alpha \mathbf{K} +
            (1 - \alpha) \mathbf{Z}\mathbf{Z}^T

    where :math:`\alpha` is a mixing parameter,
    :math:`\mathbf{K}` is the input kernel of shape :math:`(n_{samples}, n_{samples})`
    and :math:`\mathbf{Z}` is a matrix of class confidence scores of shape 
    :math:`(n_{samples}, n_{classes})`

    Parameters
    ----------
    mixing : float, default=0.5
        mixing parameter, as described in PCovC as :math:`{\alpha}`

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

    classifier: {`RidgeClassifier`, `RidgeClassifierCV`, `LogisticRegression`,
        `LogisticRegressionCV`, `SGDClassifier`, `LinearSVC`, `precomputed`}, default=None
        classifier for computing :math:`{\mathbf{Z}}`. The classifier should be one
        `sklearn.linear_model.RidgeClassifier`, `sklearn.linear_model.RidgeClassifierCV`,
        `sklearn.linear_model.LogisticRegression`, `sklearn.linear_model.LogisticRegressionCV`,
        `sklearn.linear_model.SGDClassifier`, or `sklearn.svm.LinearSVC`. If a pre-fitted classifier
        is provided, it is used to compute :math:`{\mathbf{Z}}`.
        Note that any pre-fitting of the classifier will be lost if `PCovC` is
        within a composite estimator that enforces cloning, e.g.,
        `sklearn.compose.TransformedTargetclassifier` or
        `sklearn.pipeline.Pipeline` with model caching.
        In such cases, the classifier will be re-fitted on the same
        training data as the composite estimator.
        If `precomputed`, we assume that the `y` passed to the `fit` function
        is the classified form of the targets :math:`{\mathbf{\hat{Y}}}`.
        If None, ``sklearn.linear_model.LogisticRegression()``
        is used as the classifier.

    kernel : {"linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"}, default="linear
        Kernel.

    gamma : {'scale', 'auto'} or float, default=None
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

    pkz_: numpy.ndarray of size :math:`({n_{samples}, n_{classes}})`
        the projector, or weights, from the input kernel :math:`\mathbf{K}`
        to the class confidence scores :math:`\mathbf{Z}`

    ptz_: numpy.ndarray of size :math:`({n_{components}, n_{classes}})`
        the projector, or weights, from the latent-space projection
        :math:`\mathbf{T}` to the class confidence scores :math:`\mathbf{Z}`

    ptx_: numpy.ndarray of size :math:`({n_{components}, n_{features}})`
        the projector, or weights, from the latent-space projection
        :math:`\mathbf{T}` to the feature matrix :math:`\mathbf{X}`

    X_fit_: numpy.ndarray of shape (n_samples, n_features)
        The data used to fit the model. This attribute is used to build kernels
        from new data.

    Examples
    --------
    >>> import numpy as np
    >>> from skmatter.decomposition import KernelPCovC
    >>> from sklearn.preprocessing import StandardScaler
    >>> X = np.array([[-2, 3, -1, 0], [2, 0, -3, 1], [3, 0, -1, 3], [2, -2, 1, 0]])
    >>> X = scaler.fit_transform(X)
    >>> Y = np.array([[2], [0], [1], [2]])
    >>> kpcovc = KernelPCovC(
    ...     mixing=0.1,
    ...     n_components=2,
    ...     kernel="rbf",
    ...     gamma=1,
    ... )
    >>> kpcovc.fit(X, Y)
    KernelPCovC(gamma=1, kernel='rbf', mixing=0.1, n_components=2)
    >>> kpcovc.transform(X)
    array([[-4.45970689e-01  8.95327566e-06]
           [ 4.52745933e-01  5.54810948e-01]
           [ 4.52881359e-01 -5.54708315e-01]
           [-4.45921092e-01 -7.32157649e-05]])
    >>> kpcovc.predict(X)
    array([2 0 1 2])
    >>> kpcovc.score(X, Y)
    1.0
    """  # NoQa: E501

    def __init__(
        self,
        mixing=0.5,
        n_components=None,
        svd_solver="auto",
        classifier=None,
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
        self.classifier = classifier

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
            Classification weights, optional when classifier=`precomputed`. If not
            passed, it is assumed that `W = np.linalg.lstsq(K, Y, self.tol)[0]`

        Returns
        -------
        self: object
            Returns the instance itself.
        """
        X, Y = validate_data(self, X, Y, y_numeric=False, multi_output=True)
        check_classification_targets(Y)
        self.classes_ = np.unique(Y)

        K = super()._fit_utils(X)

        compatible_classifiers = (
            LinearDiscriminantAnalysis,
            LinearSVC,
            LogisticRegression,
            LogisticRegressionCV,
            MultiOutputClassifier,
            Perceptron,
            RidgeClassifier,
            RidgeClassifierCV,
            SGDClassifier,
        )

        if self.classifier not in ["precomputed", None] and not isinstance(
            self.classifier, compatible_classifiers
        ):
            raise ValueError(
                "Classifier must be an instance of `"
                f"{'`, `'.join(c.__name__ for c in compatible_classifiers)}`"
                ", or `precomputed`"
            )

        if self.classifier != "precomputed":
            if self.classifier is None:
                classifier = LogisticRegression()
            else:
                classifier = self.classifier

            # Check if classifier is fitted; if not, fit with precomputed K
            self.z_classifier_ = check_cl_fit(classifier, K, Y)
            W = self.z_classifier_.coef_.T.reshape(K.shape[1], -1)

            self.classifier_ = clone(classifier)
        else:
            # If precomputed, use default classifier to predict Y from T
            classifier = LogisticRegression()
            if W is None:
                W = LogisticRegression().fit(X, Y).coef_.T
                W = W.reshape(X.shape[1], -1)

        Z = K @ W

        self._fit(K, Z, W)

        self.ptk_ = self.pt__ @ K
        print("KPCovc"+str(self.ptk_[:10][1]))
        if self.fit_inverse_transform:
            self.ptx_ = self.pt__ @ X

        self.classifier_ = clone(classifier).fit(K @ self.pkt_, Y)

        self.ptz_ = self.classifier_.coef_.T
        self.pkz_ = self.pkt_ @ self.ptz_

        if len(Y.shape) == 1 and type_of_target(Y) == "binary":
            self.pkz_ = self.pkz_.reshape(
                K.shape[1],
            )
            self.ptz_ = self.ptz_.reshape(
                self.n_components_,
            )

        self.components_ = self.pkt_.T  # for sklearn compatibility
        return self

    def predict(self, X=None, T=None):
        """Predicts the property labels using classification on T."""
        check_is_fitted(self, ["pkz_", "ptz_"])

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            X = validate_data(self, X, reset=False)
            K = self._get_kernel(X, self.X_fit_)
            if self.center:
                K = self.centerer_.transform(K)

            return self.classifier_.predict(K @ self.pkt_)
        else:
            return self.classifier_.predict(T)

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

    def decision_function(self, X=None, T=None):
        """Predicts confidence scores from X or T."""
        check_is_fitted(self, attributes=["pkz_", "ptz_"])

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            X = validate_data(self, X, reset=False)
            K = self._get_kernel(X, self.X_fit_)
            if self.center:
                K = self.centerer_.transform(K)

            # Or self.classifier_.decision_function(K @ self.pxt_)
            return K @ self.pkz_ + self.classifier_.intercept_

        else:
            T = check_array(T)
            return T @ self.ptz_ + self.classifier_.intercept_
