import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.base import MultiOutputMixin, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted


class OrthogonalRegression(MultiOutputMixin, RegressorMixin):
    r"""Orthogonal regression by solving the Procrustes problem

    Linear regression with the additional constraint that the weight matrix
    must be an orthogonal matrix/projection. It minimizes the Procrustes
    problem:

    .. math::

        \min_\Omega ||y - X\Omega\||_F \quad\mathrm{subject\ to}\quad \Omega^T\Omega=I

    Parameters
    ----------
    use_orthogonal_projector : bool, default=True
        Controls if orthogonal projectors are used to predict y fitting on X.
        If this parameter is set to False X and y are padded with zeros to the larger
        number of features of X and y. The projection method is similar
        to the procedure in the computation GFRD in the first version of
        Ref. [Goscinski2021]_. The method has been adapted obtain a full weight matrix.

        The projection can introduce nonanalytic behavior with respect to
        changes in dimensions of X for cases where X n_features > y n_targets.
        See ``examples/OrthogonalRegressionNonAnalytic_no-doc.ipynb``

    linear_estimator : object implementing fit/predict, default=None
        The linear estimator is used when `use_orthogonal_projector`
        is set to True, to compute the projection matrix

    Attributes
    ----------
    max_components_ : int
        The source X and target y are padded with zeros to match in feature/target
        dimension, when `use_orthogonal_projector` is set to False. This attribute
        is set to the maximum of the feature and target dimension.

    coef_ : numpy.ndarray of shape (n_features,) or (n_targets, n_features) or (max_components, max_components)
        Weight matrix. The shape (max_components, max_components) is used if
        `use_orthogonal_projector` is set to False.
    """  # NoQa: E501

    def __init__(self, use_orthogonal_projector=True, linear_estimator=None):
        self.use_orthogonal_projector = use_orthogonal_projector
        self.linear_estimator = linear_estimator

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        y : numpy.ndarray of shape (n_samples, n_targets)
            Training data, where ``n_samples`` is the number of samples and
            ``n_targets`` is the number of target properties.
        """
        X, y = check_X_y(
            X,
            y,
            y_numeric=True,
            ensure_min_features=1,
            ensure_min_samples=1,
            multi_output=True,
        )

        self.n_samples_in_, self.n_features_in_ = X.shape
        if self.use_orthogonal_projector:
            # check estimator
            linear_estimator = (
                LinearRegression()
                if self.linear_estimator is None
                else self.linear_estimator
            )
            # compute orthogonal projectors
            linear_estimator.fit(X, y)
            coef = np.reshape(linear_estimator.coef_.T, (X.shape[1], -1))
            U, _, Vt = np.linalg.svd(coef, full_matrices=False)

            # compute weights by solving the Procrustes problem
            self.coef_ = (
                U
                @ orthogonal_procrustes(X @ U, y.reshape(X.shape[0], -1) @ Vt.T)[0]
                @ Vt
            ).T
        else:
            self.max_components_ = max(X.shape[1], y.shape[1])
            X = np.pad(X, [(0, 0), (0, self.max_components_ - X.shape[1])])
            y = np.pad(y, [(0, 0), (0, self.max_components_ - y.shape[1])])
            self.coef_ = orthogonal_procrustes(X, y)[0].T

        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is
            the number of features.
        """
        X = check_array(X, ensure_min_features=1, ensure_min_samples=1)
        check_is_fitted(self, ["coef_"])

        if not (self.use_orthogonal_projector):
            X = np.pad(X, [(0, 0), (0, self.max_components_ - X.shape[1])])
        return X @ self.coef_.T
