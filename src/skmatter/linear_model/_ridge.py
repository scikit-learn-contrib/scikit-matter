import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.metrics import check_scoring
from sklearn.model_selection import KFold, check_cv
from sklearn.utils.validation import check_is_fitted, validate_data


class Ridge2FoldCV(RegressorMixin, MultiOutputMixin, BaseEstimator):
    r"""Ridge regression with an efficient 2-fold cross-validation method using the SVD
    solver.

    Minimizes the objective function:

    .. math::

        \|y - Xw\|^2_2 + \alpha \|w\|^2_2,

    while the alpha value is determined with a 2-fold cross-validation from a list of
    alpha values. It is more efficient version than doing 2-fold cross-validation
    naively The algorithmic trick is to reuse the matrices obtained by SVD for each
    regularization paramater :param alpha: The 2-fold CV can be broken down to

    .. math::

         \begin{align}
             &\mathbf{X}_1 = \mathbf{U}_1\mathbf{S}_1\mathbf{V}_1^T,
                   \qquad\qquad\qquad\quad
                   \textrm{feature matrix }\mathbf{X}\textrm{ for fold 1} \\
             &\mathbf{W}_1(\lambda) = \mathbf{V}_1
                    \tilde{\mathbf{S}}_1(\lambda)^{-1} \mathbf{U}_1^T y_1,
                    \qquad
                    \textrm{weight matrix fitted on fold 1}\\
             &\tilde{y}_2 = \mathbf{X}_2 \mathbf{W}_1,
                    \qquad\qquad\qquad\qquad
                    \textrm{ prediction of } y\textrm{ for fold 2}
         \end{align}

    where the matrices

     .. math::

          \begin{align}
              &\mathbf{A}_1 = \mathbf{X}_2 \mathbf{V}_1, \quad
               \mathbf{B}_1 = \mathbf{U}_1^T y_1.
          \end{align}

    are stored to not recompute the SVD.

    It offers additional functionalities in comparison to
    :obj:`sklearn.linear_model.RidgeCV`: The regularization parameters can be chosen
    relative to the largest eigenvalue of the feature matrix using :param alpha_type:
    as well as type of regularization using :param regularization_method:.
    Details are explained in the `Parameters` section.

    It does not offer :param fit_intercept: as sklearn linear models do. It only
    can fit with no intercept.

    Parameters
    ----------
    alphas : numpy.ndarray of shape (n_alphas,), default=(0.1, 1.0, 10.0)
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
    cv: cross-validation generator or an iterable, default=None
        The first yield of the generator is used do determine the two folds.
        If None, a 0.5 split of the two folds is used using the arguments
        :param shuffle: and :param random_state:
    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.
        If :param cv: is not None, this parameter is ignored.
    random_state : int or :class:`numpy.random.`RandomState` instance, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See
        `random_state glossary from sklearn (external link) <https://scikit-learn.org/stable/glossary.html#term-random-state>`_
        parameter is ignored.
        If :param cv: is not None, this parameter is ignored.
    scoring : str, callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If None, the negative mean squared error is used.
    n_jobs : int, default=None
        The number of CPUs to use to do the computation.
        :obj:`None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See
        `n_jobs glossary from sklearn (external link) <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    Attributes
    ----------
    cv_values_ : numpy.ndarray of shape (n_alphas)
        2-fold cross-validation values for each alpha. After :meth:`fit` has
        been called, this attribute will contain the values out of score
        function
    coef_ : numpy.ndarray of shape (n_features) or (n_targets, n_features)
        Weight vector(s).
    alpha_ : float
        Estimated regularization parameter.
    best_score_ : float
        Score of base estimator with best alpha.

    """  # NoQa: E501

    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        alpha_type="absolute",
        regularization_method="tikhonov",
        cv=None,
        scoring=None,
        random_state=None,
        shuffle=True,
        n_jobs=None,
    ):
        self.alphas = np.asarray(alphas)
        self.alpha_type = alpha_type
        self.regularization_method = regularization_method
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.shuffle = shuffle
        self.n_jobs = n_jobs

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.single_output = False
        return tags

    def _more_tags(self):
        return {"multioutput_only": True}

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : numpy.ndarray of shape (n_samples, n_targets)
            Training data, where n_samples is the number of samples
            and n_targets is the number of target properties.
        """
        # check input parameters, can be moved at some point to a sklearn-like check
        # function
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
                "relative alphas type used, but the alphas are not within the range "
                "[0,1)"
            )

        X, y = validate_data(self, X, y, y_numeric=True, multi_output=True)
        self.n_samples_in_, self.n_features_in_ = X.shape

        # check_scoring uses estimators scoring function if the scorer is None, this is
        # intercepted here
        if self.scoring is None:
            scorer = check_scoring(
                self, scoring="neg_mean_squared_error", allow_none=False
            )
        else:
            scorer = check_scoring(self, scoring=self.scoring, allow_none=False)

        if self.cv is None:
            cv = KFold(n_splits=2, shuffle=self.shuffle, random_state=self.random_state)
        else:
            cv = check_cv(self.cv)

        fold1_idx, fold2_idx = next(cv.split(X))
        self.coef_ = self._2fold_cv(X, y, fold1_idx, fold2_idx, scorer)
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        """
        X = validate_data(self, X, reset=False)

        check_is_fitted(self, ["coef_"])

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


class _IdentityRegressor(BaseEstimator):
    """Fake regressor which will directly output the prediction."""

    def predict(self, y_predict):
        return y_predict
