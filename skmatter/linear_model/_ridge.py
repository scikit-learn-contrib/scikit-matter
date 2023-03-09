import numpy as np
from joblib import (
    Parallel,
    delayed,
)
from sklearn.base import (
    MultiOutputMixin,
    RegressorMixin,
)
from sklearn.metrics import check_scoring
from sklearn.model_selection import KFold


class RidgeRegression2FoldCV(MultiOutputMixin, RegressorMixin):
    r"""Ridge regression with an efficient 2-fold cross-validation method using the SVD solver.
    Minimizes the objective function:

    .. math::

        \|y - Xw\|^2_2 + \alpha \|w\|^2_2,

    while the alpha value is determined with a 2-fold cross-validation from a list of
    alpha values. It is more efficient than doing a 2-fold cross-validation using
    :obj:`sklearn.linear_model.RidgeCV`.
    The advantage over :obj:`sklearn.linear_model.RidgeCV` using leave-one-out cross-validation
    (LOOCV) [loocv]_ needs to be analyzed more in detail. Internal benchmarks suggest that it
    is more efficient than the LOOCV in :obj:`sklearn.linear_model.RidgeCV` for feature sizes < 600
    and in general more accurate, see issue #40. However, it is constraint to a svd
    solver for the matrix inversion.
    It offers additional functionalities in comparison to :obj:`sklearn.linear_model.Ridge`:
    The regularaization parameters can be chosen relative to the largest eigenvalue of the feature matrix
    as well as regularization method. Details are explained in the `Parameters` section.

    Parameters
    ----------
    alphas : ndarray of shape (n_alphas,), default=(0.1, 1.0, 10.0)
        Array of alpha values to try.
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
    alpha_type : str, default="absolute"
        "absolute" assumes that `alphas` are absolute values and does not scale them,
        "relative" assumes that `alphas` as relative values in the range [0,1] and
        scales them with respect to the largest eigenvalue.
    regularization_method : str, default="tikhonov"
        "tikhonov" uses the ``alphas`` with standard Tihhonov regularization,
        "cutoff" uses the ``alphas as a cutoff for the eigenvalues of the kernel,
        The case "cutoff" with "relative" ``alphas`` has the same effect as the `rcond`
        parameter in e.g. :obj:`numpy.linalg.lstsq`. Be aware that for every case
        we always apply a small default cutoff dependend on the numerical
        accuracy of the data type of ``X`` in the fitting function.
    random_state : int or RandomState instance, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See
        `random_state glossary from sklearn (external link) <https://scikit-learn.org/stable/glossary.html#term-random-state>`_
    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    scoring : str, callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If None, the negative mean squared error is used.
    n_jobs : int, default=None
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See
        `n_jobs glossary from sklearn (external link) <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    Attributes
    ----------
    cv_values_ : ndarray of shape (n_samples, n_alphas) or \
        shape (n_samples, n_targets, n_alphas), optional
        Cross-validation values for each alpha (only available if \
        ``store_cv_values=True`` and ``cv=None``). After ``fit()`` has been \
        called, this attribute will contain the mean squared errors \
        (by default) or the values of the ``{loss,score}_func`` function \
        (if provided in the constructor).
    coef_ : ndarray of shape (n_features) or (n_targets, n_features)
        Weight vector(s).
    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.
    alpha_ : float
        Estimated regularization parameter.
    best_score_ : float
        Score of base estimator with best alpha.

    References
    ----------
    .. [loocv] Rifkin "Regularized Least Squares."
            https://www.mit.edu/~9.520/spring07/Classes/rlsslides.pdf
    """

    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        alpha_type="absolute",
        regularization_method="tikhonov",
        scoring=None,
        random_state=None,
        shuffle=True,
        n_jobs=None,
    ):
        self.alphas = np.asarray(alphas)
        self.alpha_type = alpha_type
        self.regularization_method = regularization_method
        self.scoring = scoring
        self.random_state = random_state
        self.shuffle = shuffle
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : ndarray of shape (n_samples, n_targets)
            Training data, where n_samples is the number of samples
            and n_targets is the number of target properties.
        """
        # check input parameters, can be moved at some point to a sklearn-like check function
        if self.regularization_method not in ["tikhonov", "cutoff"]:
            raise ValueError(
                f"regularization method {self.regularization_method} is not known."
            )
        if self.alpha_type not in ["absolute", "relative"]:
            raise ValueError(f"alpha type {self.alpha_type} is not known.")
        if self.alpha_type == "relative" and (
            np.any(self.alphas < 0) or np.any(self.alphas >= 1)
        ):
            raise ValueError(
                "relative alphas type used, but the alphas are not within the range [0,1)"
            )

        # check_scoring uses estimators scoring function if the scorer is None, this is intercepted here
        if self.scoring is None:
            scorer = check_scoring(
                self, scoring="neg_root_mean_squared_error", allow_none=False
            )
        else:
            scorer = check_scoring(self, scoring=self.scoring, allow_none=False)
        fold1_idx, fold2_idx = next(
            KFold(
                n_splits=2, shuffle=self.shuffle, random_state=self.random_state
            ).split(X)
        )
        self.coef_ = self._2fold_cv(X, y, fold1_idx, fold2_idx, scorer)
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        """
        return X @ self.coef_.T

    def _2fold_cv(self, X, y, fold1_idx, fold2_idx, scorer):
        # two-way split of the dataset
        X_fold1 = X[fold1_idx]
        X_fold2 = X[fold2_idx]
        y_fold1 = y[fold1_idx]
        y_fold2 = y[fold2_idx]

        U_fold1, s_fold1, Vt_fold1 = np.linalg.svd(X_fold1, full_matrices=False)
        U_fold2, s_fold2, Vt_fold2 = np.linalg.svd(X_fold2, full_matrices=False)

        # scipy.linalg.pinv default rcond value
        rcond = max(X.shape) * np.spacing(X.real.dtype.type(1))

        # cutoff for singular values
        n_fold1 = sum(s_fold1 > rcond)
        n_fold2 = sum(s_fold2 > rcond)

        # computes intermediates in the least squares solution by SVD,
        # y2 ~ X_fold2 @ (V_fold1@S_fold1@U_fold1.T)@y_fold1
        Ut_fold1_y_fold1 = U_fold1.T[:n_fold1] @ y_fold1
        Ut_fold2_y_fold2 = U_fold2.T[:n_fold2] @ y_fold2
        X_fold2_V_fold1 = X_fold2 @ Vt_fold1.T[:, :n_fold1]
        X_fold1_V_fold2 = X_fold1 @ Vt_fold2.T[:, :n_fold2]

        # The scorer want an object that will make the predictions but
        # they are already computed efficiently. This
        # identity_estimator will just return them
        identity_estimator = _IdentityRegressor()

        # we use the the maximum eigenvalue of both fold, to simplify the code
        scaled_alphas = np.copy(self.alphas)
        if self.alpha_type == "relative":
            scaled_alphas *= max(np.max(s_fold1), np.max(s_fold2))

        def _2fold_loss_cutoff(alpha):
            # error approximating X2 a-fitted model and vice versa
            n_alpha_fold1 = min(n_fold1, sum(s_fold1 > alpha))
            loss_1_to_2 = scorer(
                identity_estimator,
                y_fold2,
                (X_fold2_V_fold1[:, :n_alpha_fold1] / s_fold1[:n_alpha_fold1])
                @ Ut_fold1_y_fold1[:n_alpha_fold1],
            )
            n_alpha_fold2 = min(n_fold2, sum(s_fold2 > alpha))
            loss_2_to_1 = scorer(
                identity_estimator,
                y_fold1,
                (X_fold1_V_fold2[:, :n_alpha_fold2] / s_fold2[:n_alpha_fold2])
                @ Ut_fold2_y_fold2[:n_alpha_fold2],
            )
            return (loss_1_to_2 + loss_2_to_1) / 2

        def _2fold_loss_tikhonov(alpha):
            # error approximating X2 a-fitted model and vice versa
            loss_1_to_2 = scorer(
                identity_estimator,
                y_fold2,
                (
                    X_fold2_V_fold1
                    * (s_fold1[:n_fold1] / (s_fold1[:n_fold1] ** 2 + alpha))
                )
                @ Ut_fold1_y_fold1,
            )
            loss_2_to_1 = scorer(
                identity_estimator,
                y_fold1,
                (
                    X_fold1_V_fold2
                    * (s_fold2[:n_fold2] / (s_fold2[:n_fold2] ** 2 + alpha))
                )
                @ Ut_fold2_y_fold2,
            )
            return (loss_1_to_2 + loss_2_to_1) / 2

        if self.regularization_method == "tikhonov":
            _2fold_loss_function = _2fold_loss_tikhonov
        elif self.regularization_method == "cutoff":
            _2fold_loss_function = _2fold_loss_cutoff

        self.cv_values_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_2fold_loss_function)(alpha) for alpha in scaled_alphas
        )
        self.best_score_ = np.max(self.cv_values_)
        best_alpha_idx = np.argmax(self.cv_values_)
        self.alpha_ = self.alphas[best_alpha_idx]
        best_scaled_alpha = scaled_alphas[best_alpha_idx]

        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        n = len(s > rcond)
        if self.regularization_method == "tikhonov":
            return (
                (Vt.T[:, :n] * s[:n] / (s[:n] ** 2 + best_scaled_alpha)) @ (U.T[:n] @ y)
            ).T
        elif self.regularization_method == "cutoff":
            n_alpha = min(n, sum(s > best_scaled_alpha))
            return ((Vt.T[:, :n_alpha] / s[:n_alpha]) @ (U.T[:n_alpha] @ y)).T


class _IdentityRegressor:
    """Fake regressor which will directly output the prediction."""

    def predict(self, y_predict):
        return y_predict
