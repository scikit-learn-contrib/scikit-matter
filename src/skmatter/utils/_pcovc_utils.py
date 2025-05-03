from copy import deepcopy
from sklearn import clone
from sklearn.base import check_is_fitted
from sklearn.exceptions import NotFittedError
import numpy as np

def check_cl_fit(classifier, X, y):
        try:
            check_is_fitted(classifier)
            fitted_classifier = deepcopy(classifier)

            # Check compatibility with X
            fitted_classifier._validate_data(X, y, reset=False, multi_output=True)
   
            n_classes = len(np.unique(y))
            # Check compatibility with y
            # dimension of classifier coefficients is always 2, hence we don't 
            # need to check dimension for match with Y 
            # We need to double check this...
            if n_classes == 2:
                if fitted_classifier.coef_.shape[0] != 1:
                    raise ValueError(
                        "For binary classification, expected classifier coefficients "
                        "to have shape (1, %d) but got shape %r"
                        % (fitted_classifier.n_features_in_, fitted_classifier.coef_.shape)
                    )
            else:
                if fitted_classifier.coef_.shape[0] != n_classes:
                    raise ValueError(
                        "For multiclass classification, expected classifier coefficients "
                        "to have shape (%d, %d) but got shape %r" 
                        % (n_classes, fitted_classifier.n_features_in_, fitted_classifier.coef_.shape)
                    )
                
        except NotFittedError:
            fitted_classifier = clone(classifier)
            fitted_classifier.fit(X, y)

        return fitted_classifier

def check_svc_fit(classifier, K, X, y):
    r"""
    Checks that a (SVC) classifier is fitted, and if not,
    fits it with the provided data

    :param classifier: sklearn-style classifier
    :type classifier: object
    :param K: kernel matrix with which to fit the classifier
        if it is not already fitted
    :type K: array
    :param X: feature matrix with which to check the classifier
    :type X: array
    :param y: target values with which to fit the classifier
        if it is not already fitted
    :type y: array
    """
    try:
        check_is_fitted(classifier)
        fitted_classifier = deepcopy(classifier)

        # Check compatibility with K
        fitted_classifier._validate_data(X, y, reset=False, multi_output=True)
        print("Pass")
        # Check compatibility with y
        # if fitted_regressor.dual_coef_.ndim != y.ndim:
        #     raise ValueError(
        #         "The regressor coefficients have a dimension incompatible "
        #         "with the supplied target space. "
        #         "The coefficients have dimension %d and the targets "
        #         "have dimension %d" % (fitted_regressor.dual_coef_.ndim, y.ndim)
        #     )
        # elif y.ndim == 2:
        #     if fitted_regressor.dual_coef_.shape[1] != y.shape[1]:
        #         raise ValueError(
        #             "The regressor coefficients have a shape incompatible "
        #             "with the supplied target space. "
        #             "The coefficients have shape %r and the targets "
        #             "have shape %r" % (fitted_regressor.dual_coef_.shape, y.shape)
        #        )

    except NotFittedError:
        fitted_classifier = clone(classifier)

        # Use a precomputed kernel
        # to avoid re-computing K
        fitted_classifier.set_params(kernel="precomputed")
        fitted_classifier.fit(K, y=y)

    return fitted_classifier