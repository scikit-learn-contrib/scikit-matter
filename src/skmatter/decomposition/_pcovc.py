import numpy as np
from sklearn import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.linear_model import (
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV,
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier,
)
from sklearn.svm import LinearSVC

from sklearn.calibration import column_or_1d
from sklearn.naive_bayes import LabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, validate_data

from skmatter.decomposition import _BasePCov
from skmatter.utils import check_cl_fit


class PCovC(_BasePCov):
    r"""Principal Covariates Classification determines a latent-space projection :math:`\mathbf{T}`
    which minimizes a combined loss in supervised and unsupervised tasks.

    This projection is determined by the eigendecomposition of a modified gram
    matrix :math:`\mathbf{\tilde{K}}`

    .. math::
      \mathbf{\tilde{K}} = \alpha \mathbf{X} \mathbf{X}^T +
            (1 - \alpha) \mathbf{Z}\mathbf{Z}^T

    where :math:`\alpha` is a mixing parameter, :math:`\mathbf{X}` is an input matrix of shape
    :math:`(n_{samples}, n_{features})`, and :math:`\mathbf{Z}` is an evidence tensor of shape
    :math:`(n_{samples}, n_{classes}, n_{labels})`. For :math:`(n_{samples} < n_{features})`,
    this can be more efficiently computed using the eigendecomposition of a modified covariance matrix
    :math:`\mathbf{\tilde{C}}`

    .. math::
      \mathbf{\tilde{C}} = \alpha \mathbf{X}^T \mathbf{X} +
            (1 - \alpha) \left(\left(\mathbf{X}^T
            \mathbf{X}\right)^{-\frac{1}{2}} \mathbf{X}^T
            \mathbf{Z}\mathbf{Z}^T \mathbf{X} \left(\mathbf{X}^T
            \mathbf{X}\right)^{-\frac{1}{2}}\right)

    For all PCovC methods, it is strongly suggested that :math:`\mathbf{X}` and
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
            whether to compute the PCovC in `sample` or `feature` space
            default=`sample` when :math:`{n_{samples} < n_{features}}` and
            `feature` when :math:`{n_{features} < n_{samples}}`

    classifier: {`RidgeClassifier`, `RidgeClassifierCV`, `LogisticRegression`,
            `LogisticRegressionCV`, `SGDClassifier`, `LinearSVC`, `precomputed`}, default=None
            classifier for computing :math:`{\mathbf{Z}}`. The classifier should be one
            `sklearn.linear_model.RidgeClassifier`, `sklearn.linear_model.RidgeClassifierCV`,
            `sklearn.linear_model.LogisticRegression`, `sklearn.linear_model.LogisticRegressionCV`,
            `sklearn.linear_model.SGDClassifier`, or `sklearn.svm.LinearSVC`. If a pre-fitted classifier
            is provided, it is used to compute :math:`{\mathbf{Y}}`.
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
            whether to compute the PCovC in `sample` or `feature` space
            default=`sample` when :math:`{n_{samples} < n_{features}}` and
            `feature` when :math:`{n_{features} < n_{samples}}`

    n_components_ : int
        The estimated number of components, which equals the parameter
        n_components, or the lesser value of n_features and n_samples
        if n_components is None.

    pxt_ : ndarray of size :math:`({n_{features}, n_{components}})`
           the projector, or weights, from the input space :math:`\mathbf{X}`
           to the latent-space projection :math:`\mathbf{T}`

    ptz_ : ndarray of size :math:`({n_{components}, n_{classes}})`
          the projector, or weights, from the latent-space projection
          :math:`\mathbf{T}` to the class likelihoods :math:`\mathbf{Z}`

    pxz_ : ndarray of size :math:`({n_{features}, n_{classes}})`
           the projector, or weights, from the input space :math:`\mathbf{X}`
           to the class likelihoods :math:`\mathbf{Z}`

    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.
        Equal to n_components largest eigenvalues
        of the PCovC-modified covariance matrix of :math:`\mathbf{X}`.

    singular_values_ : ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.

    Examples
    --------
    >>> import numpy as np
    >>> from skmatter.decomposition import PCovC
    >>> X = np.array([[-1, 0, -2, 3], [3, -2, 0, 1], [-3, 0, -1, -1], [1, 3, 0, -2]])
    >>> Y = np.array([[0], [1], [2], [0]])
    >>> pcovc = PCovC(mixing=0.1, n_components=2)
    >>> pcovc.fit(X, Y)
    PCovC(mixing=0.1, n_components=2)
    >>> pcovc.transform(X)
    array([[-0.32189393  0.81738389]
           [ 3.13455213 -0.40636372]
           [-2.2883084  -1.51562597]
           [-0.5243498   1.1046058 ]])
    >>> pcovc.predict(X)
    array([[0], [1], [2], [0]])
    """

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

    def fit(self, X, y, W=None):
        r"""Fit the model with X and y. Depending on the dimensions of X, calls either
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

        y : numpy.ndarray, shape (n_samples, n_properties)
            Training data, where n_samples is the number of samples and n_properties is
            the number of properties

            It is suggested that :math:`\mathbf{X}` be centered by its column-means and
            scaled. If features are related, the matrix should be scaled to have unit
            variance, otherwise :math:`\mathbf{Y}` should be scaled so that each feature
            has a variance of 1 / n_features.

            If the passed classifier = `precomputed`, it is assumed that Y is the
            classified form of the properties, :math:`{\mathbf{\hat{Y}}}`.

        W : numpy.ndarray, shape (n_features, n_properties)
            Classification weights, optional when classifier=`precomputed`. If not
            passed, it is assumed that `W = np.linalg.lstsq(X, Z, self.tol)[0]`
        """
        X, y = validate_data(self, X, y, multi_output=True)
        super()._fit_utils(X, y)

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

            self.z_classifier_ = check_cl_fit(
                classifier, X, y
            )  # its linear classifier on x and y to get Pxz

            if isinstance(self.z_classifier_, MultiOutputClassifier):
                W = np.hstack([est_.coef_.T for est_ in self.z_classifier_.estimators_])
                Z = X @ W  # computes Z, basically Z=XPxz

            else:
                W = self.z_classifier_.coef_.T.reshape(X.shape[1], -1)
                Z = self.z_classifier_.decision_function(X).reshape(X.shape[0], -1)

        else:
            Z = X @ W
            if W is None:
                W = np.linalg.lstsq(X, Z, self.tol)[0]  # W = weights for Pxz

        self._label_binarizer = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self._label_binarizer.fit_transform(y)  # check if we need this
        if not self._label_binarizer.y_type_.startswith("multilabel"):
            y = column_or_1d(y, warn=True)

        if self.space_ == "feature":
            self._fit_feature_space(X, Y.reshape(Z.shape), Z)
        else:
            self._fit_sample_space(X, Y.reshape(Z.shape), Z, W)

        # instead of using linear regression solution, refit with the classifier
        # and steal weights to get ptz
        # what to do when classifier = precomputed?

        # original: self.classifier_ = check_cl_fit(classifier, X @ self.pxt_, y=y)
        # we don't want to copy ALl parameters of classifier, such as n_features_in, since we are re-fitting it on T, y
        if self.classifier != "precomputed":
            self.classifier_ = clone(classifier).fit(X @ self.pxt_, y)
        else:
            # if precomputed, use default classifier to predict y from T
            self.classifier_ = LogisticRegression().fit(X @ self.pxt_, y)
        print(self.classifier_)

        # self.classifier_ = LogisticRegression().fit(X @ self.pxt_, y)
        # check_cl_fit(classifier., X @ self.pxt_, y=y) #Has Ptz as weights

        if isinstance(self.classifier_, MultiOutputClassifier):
            self.ptz_ = np.hstack(
                [est_.coef_.T for est_ in self.classifier_.estimators_]
            )
            self.pxz_ = self.pxt_ @ self.ptz_
        else:
            self.ptz_ = self.classifier_.coef_.T
            self.pxz_ = self.pxt_ @ self.ptz_

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
        r"""In feature-space PCovC, the projectors are determined by:

        .. math::
            \mathbf{\tilde{C}} = \alpha \mathbf{X}^T \mathbf{X} +
            (1 - \alpha) \left(\left(\mathbf{X}^T
            \mathbf{X}\right)^{-\frac{1}{2}} \mathbf{X}^T
            \mathbf{Z}\mathbf{Z}}^T \mathbf{X} \left(\mathbf{X}^T
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
        return super()._fit_feature_space(X, Y, Z)

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
        return super()._fit_sample_space(X, Y, Z, W)

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

    def decision_function(self, X=None, T=None):
        """Predicts confidence scores from X or T."""
        check_is_fitted(self, attributes=["_label_binarizer", "pxz_", "ptz_"])

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            X = check_array(X)
            scores = X @ self.pxz_
        else:
            T = check_array(T)
            scores = T @ self.ptz_

        return (
            np.reshape(scores, (-1,))
            if (scores.ndim > 1 and scores.shape[1] == 1)
            else scores
        )

    def predict(self, X=None, T=None):
        """Predicts the property labels using classification on T."""
        check_is_fitted(self, attributes=["_label_binarizer", "pxz_", "ptz_"])

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            return self.classifier_.predict(X @ self.pxt_)
        else:
            return self.classifier_.predict(T)

    def transform(self, X=None):
        """Apply dimensionality reduction to X.

        ``X`` is projected on the first principal components as determined by the
        modified PCovC distances.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        """
        return super().transform(X)

    def score(self, X, Y, sample_weight=None):
        r"""Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` w.r.t. `Y`.
        """
        return accuracy_score(Y, self.predict(X), sample_weight=sample_weight)
