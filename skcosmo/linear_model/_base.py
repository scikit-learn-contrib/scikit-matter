import numpy as np
from scipy.linalg import orthogonal_procrustes

from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.linear_model import LinearRegression


class OrthogonalRegression(MultiOutputMixin, RegressorMixin):
    """Orthogonal regression by solving the Procrustes problem

    Linear regression with the additional constraint that the weight matrix
    must be an orthogonal matrix/projection. It minimizes the Procrustes
    problem:

    .. math:: \min_\Omega ||y - X\Omega\||_F \quad\mathrm{subject\ to}\quad \Omega^T\Omega=I % # noqa: W605

    Parameters
    ----------
    use_orthogonal_projector : bool, default=True
        Controls if orthogonal projectors are used to predict y fitting on X.
        If this parameter is set to False X and y are padded with zeros to the larger
        number of features of X and y. The projection method is similar
        to the procedure in the computation GFRD in the first version of
        Ref. [frm]_. The method has been adapted obtain a full weight matrix.

        The projection can introduce nonanalytic behavior with respect to
        changes in dimensions of X for cases where X n_features > y n_targets.
        See ``examples/linear_model/plot_orthogonal_regression_nonanalytic_behavior.py``

    linear_estimator : object implementing fit/predict, default=None
        The linear estimator is used when `use_orthogonal_projector`
        is set to True, to compute the projection matrix

    Attributes
    ----------
    max_components_ : int
        The source X and target y are padded with zeros to match in feature/target
        dimension, when `use_orthogonal_projector` is set to False. This attribute
        is set to the maximum of the feature and target dimension.

    coef_ : ndarray of shape (n_features,) or (n_targets, n_features) or (max_components_, max_components_)
        Weight matrix. The shape (max_components_, max_components_) is used if
        `use_orthogonal_projector` is set to False.

    References
    ----------
    .. [frm]_ Goscinski, Fraux, Imbalzano and Ceriotti. "The role of feature space in
             atomistic learning." arXiv preprint arXiv:2009.02741 (2020).
    """

    def __init__(self, use_orthogonal_projector=True, linear_estimator=None):
        self.use_orthogonal_projector = use_orthogonal_projector
        self.linear_estimator = linear_estimator

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array_like of shape (n_samples, n_targets)
            Training data, where n_samples is the number of samples
            and n_targets is the number of target properties.
        """
        if self.use_orthogonal_projector:
            # check estimator
            linear_estimator = (
                LinearRegression()
                if self.linear_estimator is None
                else self.linear_estimator
            )
            # compute orthogonal projectors
            linear_estimator.fit(X, y)
            U, _, Vt = np.linalg.svd(linear_estimator.coef_.T, full_matrices=False)
            # project X and y to same dimension
            X = X @ U
            y = y @ Vt.T
            # compute weights by solving the Procrustes problem
            self.coef_ = (U @ orthogonal_procrustes(X, y)[0] @ Vt).T
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
        X : array_like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        """
        if not (self.use_orthogonal_projector):
            X = np.pad(X, [(0, 0), (0, self.max_components_ - X.shape[1])])
        return X @ self.coef_.T
