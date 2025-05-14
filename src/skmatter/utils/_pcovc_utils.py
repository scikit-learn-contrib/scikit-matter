from copy import deepcopy

import numpy as np
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted, validate_data


def check_cl_fit(classifier, X, y):
    """
    Checks that a (linear) classifier is fitted, and if not,
    fits it with the provided data.

    Parameters
    ----------
    classifier : object
        sklearn-style classifier
    X : array-like
        Feature matrix with which to fit the classifier if it is not already fitted
    y : array-like
        Target values with which to fit the classifier if it is not already fitted

    Returns
    -------
    fitted_classifier : object
        The fitted classifier. If input classifier was already fitted and compatible
        with the data, returns a deep copy. Otherwise returns a newly fitted classifier.

    Raises
    ------
    ValueError
        If the fitted classifiers's coefficients have a shape incompatible with the
        number of classes or number of features.
    """
    try:
        check_is_fitted(classifier)
        fitted_classifier = deepcopy(classifier)

        # Check compatibility with X
        validate_data(fitted_classifier, X, y, reset=False, multi_output=True)

        # Check compatibility with y
        # dimension of classifier coefficients is always 2, hence we don't
        # need to check dimension for match with Y
        # We need to double check this...
        n_classes = len(np.unique(y))

        if n_classes == 2:
            if fitted_classifier.coef_.shape[0] != 1:
                raise ValueError(
                    "For binary classification, expected classifier coefficients "
                    "to have shape (1, "
                    f"{X.shape[1]}) but got shape "
                    f"{fitted_classifier.coef_.shape}"
                )
        else:
            if fitted_classifier.coef_.shape[0] != n_classes:
                raise ValueError(
                    "For multiclass classification, expected classifier coefficients "
                    "to have shape "
                    f"({n_classes}, {X.shape[1]}) but got shape "
                    f"{fitted_classifier.coef_.shape}"
                )

    except NotFittedError:
        fitted_classifier = clone(classifier)
        fitted_classifier.fit(X, y)

    return fitted_classifier
