from copy import deepcopy

import numpy as np
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted, validate_data

from sklearn.multioutput import MultiOutputClassifier


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
        number of features in X or the number of classes in y.
    """
    try:
        check_is_fitted(classifier)
        fitted_classifier = deepcopy(classifier)

        # Check compatibility with X
        validate_data(fitted_classifier, X, y, reset=False, multi_output=True)

        # Check coefficent compatibility with the number of features in X and the
        # number of classes in y
        if isinstance(fitted_classifier, MultiOutputClassifier):
            for est_ in fitted_classifier.estimators_:
                check_cl_coef(X, est_.coef_, len(est_.classes_))
        else:
            check_cl_coef(X, fitted_classifier.coef_, len(np.unique(y)))

    except NotFittedError:
        fitted_classifier = clone(classifier)
        fitted_classifier.fit(X, y)

    return fitted_classifier


def check_cl_coef(X, classifier_coef_, n_classes):
    if n_classes == 2:
        if classifier_coef_.shape[0] != 1:
            raise ValueError(
                "For binary classification, expected classifier coefficients "
                "to have shape (1, "
                f"{X.shape[1]}) but got shape "
                f"{classifier_coef_.shape}"
            )
    else:
        if classifier_coef_.shape[0] != n_classes:
            raise ValueError(
                "For multiclass classification, expected classifier coefficients "
                "to have shape "
                f"({n_classes}, {X.shape[1]}) but got shape "
                f"{classifier_coef_.shape}"
            )
