import numpy as np
from sklearn import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
)
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.utils import check_array
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from skmatter.decomposition import _BasePCov
from skmatter.utils import check_cl_fit


class PCovC(LinearClassifierMixin, _BasePCov):
    r"""Principal Covariates Classification (PCovC).

    As described in [Jorgensen2025]_, PCovC determines a latent-space projection
    :math:`\mathbf{T}` which minimizes a combined loss in supervised and
    unsupervised tasks.

    This projection is determined by the eigendecomposition of a modified gram
    matrix :math:`\mathbf{\tilde{K}}`

    .. math::
      \mathbf{\tilde{K}} = \alpha \mathbf{X} \mathbf{X}^T +
            (1 - \alpha) \mathbf{Z}\mathbf{Z}^T

    where :math:`\alpha` is a mixing parameter, :math:`\mathbf{X}` is an input matrix of shape
    :math:`(n_{samples}, n_{features})`, and :math:`\mathbf{Z}` is a matrix of class confidence scores
    of shape :math:`(n_{samples}, n_{classes})`. For :math:`(n_{samples} < n_{features})`,
    this can be more efficiently computed using the eigendecomposition of a modified covariance matrix
    :math:`\mathbf{\tilde{C}}`

    .. math::
      \mathbf{\tilde{C}} = \alpha \mathbf{X}^T \mathbf{X} +
            (1 - \alpha) \left(\left(\mathbf{X}^T
            \mathbf{X}\right)^{-\frac{1}{2}} \mathbf{X}^T
            \mathbf{Z}\mathbf{Z}^T \mathbf{X} \left(\mathbf{X}^T
            \mathbf{X}\right)^{-\frac{1}{2}}\right)

    For all PCovC methods, it is strongly suggested that :math:`\mathbf{X}` is centered
    and scaled to unit variance, otherwise the results will change drastically near
    :math:`\alpha \to 0` and :math:`\alpha \to 1`.
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
        mixing parameter, as described in PCovC as :math:`{\alpha}`, here named
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
        whether to compute the PCovC in ``sample`` or ``feature`` space.
        The default is equal to ``sample`` when :math:`{n_{samples} < n_{features}}`
        and ``feature`` when :math:`{n_{features} < n_{samples}}`

    classifier: ``estimator object`` or ``precomputed``, default=None
        classifier for computing :math:`{\mathbf{Z}}`. The classifier should be
        one of the following:

        - ``sklearn.linear_model.LogisticRegression()``
        - ``sklearn.linear_model.LogisticRegressionCV()``
        - ``sklearn.svm.LinearSVC()``
        - ``sklearn.discriminant_analysis.LinearDiscriminantAnalysis()``
        - ``sklearn.linear_model.RidgeClassifier()``
        - ``sklearn.linear_model.RidgeClassifierCV()``
        - ``sklearn.linear_model.Perceptron()``

        If a pre-fitted classifier
        is provided, it is used to compute :math:`{\mathbf{Z}}`.
        Note that any pre-fitting of the classifier will be lost if `PCovC` is
        within a composite estimator that enforces cloning, e.g.,
        `sklearn.pipeline.Pipeline` with model caching.
        In such cases, the classifier will be re-fitted on the same
        training data as the composite estimator.
        If None, ``sklearn.linear_model.LogisticRegression()``
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
        mixing parameter, as described in PCovC as :math:`{\alpha}`

    tol: float, default=1e-12
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

    space: {'feature', 'sample', 'auto'}, default='auto'
        whether to compute the PCovC in ``sample`` or ``feature`` space.
        The default is equal to ``sample`` when :math:`{n_{samples} < n_{features}}`
        and ``feature`` when :math:`{n_{features} < n_{samples}}`

    n_components_ : int
        The estimated number of components, which equals the parameter
        n_components, or the lesser value of n_features and n_samples
        if n_components is None.

    classifier : estimator object
        The linear classifier passed for fitting.

    z_classifier_ : estimator object
        The linear classifier fit between :math:`\mathbf{X}` and :math:`\mathbf{Y}`.

    classifier_ : estimator object
        The linear classifier fit between :math:`\mathbf{T}` and :math:`\mathbf{Y}`.

    pxt_ : ndarray of size :math:`({n_{features}, n_{components}})`
        the projector, or weights, from the input space :math:`\mathbf{X}`
        to the latent-space projection :math:`\mathbf{T}`

    pxz_ : ndarray of size :math:`({n_{features}, })` or :math:`({n_{features}, n_{classes}})`
        the projector, or weights, from the input space :math:`\mathbf{X}`
        to the class confidence scores :math:`\mathbf{Z}`

    ptz_ : ndarray of size :math:`({n_{components}, })` or :math:`({n_{components}, n_{classes}})`
        the projector, or weights, from the latent-space projection
        :math:`\mathbf{T}` to the class confidence scores :math:`\mathbf{Z}`

    explained_variance_ : numpy.ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.
        Equal to n_components largest eigenvalues
        of the PCovC-modified covariance matrix of :math:`\mathbf{X}`.

    singular_values_ : numpy.ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.

    Examples
    --------
    >>> import numpy as np
    >>> from skmatter.decomposition import PCovC
    >>> from sklearn.preprocessing import StandardScaler
    >>> X = np.array([[-1, 0, -2, 3], [3, -2, 0, 1], [-3, 0, -1, -1], [1, 3, 0, -2]])
    >>> X = StandardScaler().fit_transform(X)
    >>> Y = np.array([0, 1, 2, 0])
    >>> pcovc = PCovC(mixing=0.1, n_components=2)
    >>> pcovc.fit(X, Y)
    PCovC(mixing=0.1, n_components=2)
    >>> pcovc.transform(X)
    array([[-0.4794854 , -0.46228114],
           [ 1.9416966 ,  0.2532831 ],
           [-1.08744947,  0.89117784],
           [-0.37476173, -0.6821798 ]])
    >>> pcovc.predict(X)
    array([0, 1, 2, 0])
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
        self.classifier = classifier

    def fit(self, X, Y, W=None):
        r"""Fit the model with X and Y.

        Note that W is taken from the coefficients of a linear classifier fit
        between X and Y to compute Z:

        .. math::
            \mathbf{Z} = \mathbf{X} \mathbf{W}

        We then call either `_fit_feature_space` or `_fit_sample_space`,
        using Z as our approximation of Y. Finally, we refit a classifier on
        T and Y to obtain :math:`\mathbf{P}_{TZ}`.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

            It is suggested that :math:`\mathbf{X}` be centered by its column-
            means and scaled. If features are related, the matrix should be
            scaled to have unit variance, otherwise :math:`\mathbf{X}` should
            be scaled so that each feature has a variance of 1 / n_features.

        Y : numpy.ndarray, shape (n_samples,)
            Training data, where n_samples is the number of samples.

        W : numpy.ndarray, shape (n_features, n_classes)
            Classification weights, optional when classifier is ``precomputed``. If
            not passed, it is assumed that the weights will be taken from a
            linear classifier fit between :math:`\mathbf{X}` and :math:`\mathbf{Y}`
        """
        X, Y = validate_data(self, X, Y, multi_output=True, y_numeric=False)
        check_classification_targets(Y)
        self.classes_ = np.unique(Y)

        super()._set_fit_params(X)

        compatible_classifiers = (
            LogisticRegression,
            LogisticRegressionCV,
            LinearSVC,
            LinearDiscriminantAnalysis,
            MultiOutputClassifier,
            RidgeClassifier,
            RidgeClassifierCV,
            SGDClassifier,
            Perceptron,
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
            if self.classifier is None and Y.ndim < 2:
                classifier = LogisticRegression()
            elif self.classifier is None and Y.ndim >= 2:
                classifier = MultiOutputClassifier(estimator=LogisticRegression())
            else:
                classifier = self.classifier

            self.z_classifier_ = check_cl_fit(classifier, X, Y)

            if isinstance(self.z_classifier_, MultiOutputClassifier):
                W = np.hstack([est_.coef_.T for est_ in self.z_classifier_.estimators_])
                print(W.shape)
            else:
                W = self.z_classifier_.coef_.T.reshape(X.shape[1], -1)

        else:
            # If precomputed, use default classifier to predict Y from T
            # check for the case of 2D Y -- we need to make sure that this is MultiOutputClassifier instead
            if Y.ndim >= 2:
                classifier = MultiOutputClassifier(estimator=LogisticRegression)
                if W is None:
                    _ = MultiOutputClassifier(estimator=LogisticRegression).fit(X, Y)
                    W = np.hstack([est_.coef_.T for est_ in _.estimators_])
            else:
                classifier = LogisticRegression()
                if W is None:
                    W = LogisticRegression().fit(X, Y).coef_.T
                    W = W.reshape(X.shape[1], -1)

        # print(f"X {X.shape}")
        # print(f"W {W.shape}")
        Z = X @ W
        # print(f"Z {Z.shape}")
        if self.space_ == "feature":
            self._fit_feature_space(X, Y, Z)
        else:
            self._fit_sample_space(X, Y, Z, W)

        # instead of using linear regression solution, refit with the
        # classifier and steal weights to get pxz and ptz
        self.classifier_ = clone(classifier).fit(X @ self.pxt_, Y)

        if isinstance(self.classifier_, MultiOutputClassifier):
            self.ptz_ = np.hstack(
                [est_.coef_.T for est_ in self.classifier_.estimators_]
            )
            # print(f"pxt {self.pxt_.shape}")
            # print(f"ptz {self.ptz_.shape}")
            self.pxz_ = self.pxt_ @ self.ptz_
            # print(f"pxz {self.pxz_.shape}")

        else:
            self.ptz_ = self.classifier_.coef_.T
            # print(self.ptz_.shape)
            self.pxz_ = self.pxt_ @ self.ptz_

        if len(Y.shape) == 1 and type_of_target(Y) == "binary":
            self.pxz_ = self.pxz_.reshape(
                X.shape[1],
            )
            self.ptz_ = self.ptz_.reshape(
                self.n_components_,
            )

        self.components_ = self.pxt_.T  # for sklearn compatibility
        return self

    def _fit_feature_space(self, X, Y, Z):
        r"""In feature-space PCovC, the projectors are determined by:

        .. math::
            \mathbf{\tilde{C}} = \alpha \mathbf{X}^T \mathbf{X} +
            (1 - \alpha) \left(\left(\mathbf{X}^T
            \mathbf{X}\right)^{-\frac{1}{2}} \mathbf{X}^T
            \mathbf{Z}\mathbf{Z}^T \mathbf{X} \left(\mathbf{X}^T
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
        """
        return super()._fit_feature_space(X, Y, Z, compute_pty_=False)

    def _fit_sample_space(self, X, Y, Z, W):
        r"""In sample-space PCovC, the projectors are determined by:

        .. math::
            \mathbf{\tilde{K}} = \alpha \mathbf{X} \mathbf{X}^T +
            (1 - \alpha) \mathbf{Z}\mathbf{Z}^T

        where

        .. math::
            \mathbf{P}_{XT} = \left(\alpha \mathbf{X}^T + (1 - \alpha)
                               \mathbf{W} \mathbf{Z}^T\right)
                               \mathbf{U}_\mathbf{\tilde{K}}
                               \mathbf{\Lambda}_\mathbf{\tilde{K}}^{-\frac{1}{2}}

        .. math::
            \mathbf{P}_{TX} = \mathbf{\Lambda}_\mathbf{\tilde{K}}^{-\frac{1}{2}}
                                \mathbf{U}_\mathbf{\tilde{K}}^T \mathbf{X}
        """
        return super()._fit_sample_space(X, Y, Z, W, compute_pty_=False)

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
        X_original : numpy.ndarray, shape (n_samples, n_features)
        """
        return super().inverse_transform(T)

    def decision_function(self, X=None, T=None):
        r"""Predicts confidence scores from X or T.

        .. math::
            \mathbf{Z} = \mathbf{T} \mathbf{P}_{TZ}
                       = \mathbf{X} \mathbf{P}_{XT} \mathbf{P}_{TZ}
                       = \mathbf{X} \mathbf{P}_{XZ}

        Parameters
        ----------
        X : ndarray, shape(n_samples, n_features)
            Original data for which we want to get confidence scores,
            where n_samples is the number of samples and n_features is the
            number of features.

        T : ndarray, shape (n_samples, n_components)
            Projected data for which we want to get confidence scores,
            where n_samples is the number of samples and n_components is the
            number of components.

        Returns
        -------
        Z : numpy.ndarray, shape (n_samples,) or (n_samples, n_classes)
            Confidence scores. For binary classification, has shape `(n_samples,)`,
            for multiclass classification, has shape `(n_samples, n_classes)`
        """
        check_is_fitted(self, attributes=["pxz_", "ptz_"])

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            X = validate_data(self, X, reset=False)
            # Or self.classifier_.decision_function(X @ self.pxt_)
            Z = X @ self.pxz_
        else:
            T = check_array(T)
            Z = T @ self.ptz_

        if isinstance(self.classifier_, MultiOutputClassifier):

            n_outputs = len(self.classifier_.estimators_)
            n_classes = Z.shape[1] // n_outputs
            print(Z.shape)
            # unpack to 3d
            Z = Z.reshape(Z.shape[0], n_outputs, n_classes)
            print(Z.shape)

            # add the intercept for estimator in MultiOutputClassifier
            for i, est_ in enumerate(self.classifier_.estimators_):
                # print(Z[:, i, :][0, :])
                Z[:, i, :] += est_.intercept_
                # print(Z)
                # print()
                print(est_.intercept_)
                # print()
                # print(Z[:, i, :][0, :])
            # swap order of Z axesfrom (n_samples, n_outputs, n_classes) to (n_samples, n_classes, n_outputs) as in paper

            return Z.transpose(0, 2, 1)

        print(self.classifier_.intercept_)
        return Z + self.classifier_.intercept_

    def predict(self, X=None, T=None):
        """Predicts the property labels using classification on T."""
        check_is_fitted(self, attributes=["pxz_", "ptz_"])

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            X = validate_data(self, X, reset=False)
            return self.classifier_.predict(X @ self.pxt_)
        else:
            T = check_array(T)
            return self.classifier_.predict(T)

    def transform(self, X=None):
        """Apply dimensionality reduction to X.

        ``X`` is projected on the first principal components as determined by
        the modified PCovC distances.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        """
        return super().transform(X)
