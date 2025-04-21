from sklearn.linear_model import (
    RidgeClassifier,
    RidgeClassifierCV,
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier
)
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

import sys
sys.path.append('scikit-matter')
from src.skmatter.decomposition._pcov import _BasePCov

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
             classifier for computing :math:`{\mathbf{Z}}`.
             The classifier should be one `sklearn.linear_model.RidgeClassifier`,
             `sklearn.linear_model.RidgeClassifierCV`, `sklearn.linear_model.LogisticRegression`, 
             `sklearn.linear_model.LogisticRegressionCV`, `sklearn.linear_model.SGDClassifier`, 
             or `sklearn.svm.LinearSVC`. If a pre-fitted classifier is provided, it is used to compute
             :math:`{\mathbf{Y}}`.
             Note that any pre-fitting of the classifier will be lost if `PCovC` is
             within a composite estimator that enforces cloning, e.g.,
             `sklearn.compose.TransformedTargetclassifier` or
             `sklearn.pipeline.Pipeline` with model caching.
             In such cases, the classifier will be re-fitted on the same
             training data as the composite estimator.
             If `precomputed`, we assume that the `y` passed to the `fit` function
             is the class likelihoods :math:`{\mathbf{Z}}`.
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

    pxt_ : ndarray of size :math:`({n_{samples}, n_{components}})`
           the projector, or weights, from the input space :math:`\mathbf{X}`
           to the latent-space projection :math:`\mathbf{T}`

    ptz_ : ndarray of size :math:`({n_{components}, n_{properties}})`
          the projector, or weights, from the latent-space projection
          :math:`\mathbf{T}` to the class likelihoods :math:`\mathbf{Z}`

    pxz_ : ndarray of size :math:`({n_{samples}, n_{properties}})`
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
    >>> from skmatter.decomposition import PCovc
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
            classifier=classifier,
            iterated_power=iterated_power,
            random_state=random_state,
            whiten=whiten,
            subclass="PCovC")

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

            If the passed classifier = `precomputed`, it is assumed that Y is the
            class likelihoods, :math:`{\mathbf{Z}}`.

        W : numpy.ndarray, shape (n_features, n_properties)
            Classification weights, optional when classifier=`precomputed`. If not
            passed, it is assumed that `W = np.linalg.lstsq(X, Z, self.tol)[0]`
        """
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
                "`Logistic RegressionCV`, `SGDClassifier`, `LinearSVC`,"
                "`MultiOutputClassifier`, or `precomputed`"
            )
        return super().fit(X, Y, W)
    
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
    
    def _decompose_truncated(self, mat):
        return super()._decompose_truncated(mat)

    def _decompose_full(self, mat):
        return super()._decompose_full(mat)

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
            return X @ self.pxz_
        else:
            T = check_array(T)
            return T @ self.ptz_
        
    def predict(self, X=None, T=None):
        """Predicts the property values using classification on X or T."""
        check_is_fitted(self, attributes=["_label_binarizer", "pxz_", "ptz_"])
        return super().predict(X, T)
        
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
        return super().score(X, Y, T)
