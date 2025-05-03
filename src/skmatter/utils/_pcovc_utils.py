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
   
            # Check compatibility with y
            # dimension of classifier coefficients is always 2, hence we don't 
            # need to check dimension for match with Y 
            # We need to double check this...
            n_classes = len(np.unique(y))

            if n_classes == 2:
                if fitted_classifier.coef_.shape[0] != 1:
                    raise ValueError(
                        "For binary classification, expected classifier coefficients "
                        "to have shape (1, %d) but got shape %r"
                        % (X.shape[1], fitted_classifier.coef_.shape)
                    )
            else:
                if fitted_classifier.coef_.shape[0] != n_classes:
                    raise ValueError(
                        "For multiclass classification, expected classifier coefficients "
                        "to have shape (%d, %d) but got shape %r" 
                        % (n_classes, X.shape[1], fitted_classifier.coef_.shape)
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

        # Check compatibility with X
        fitted_classifier._validate_data(X, y, reset=False, multi_output=True)
        print("Pass")

        #Check compatibility with y
        n_classes = len(np.unique(y))
        n_sv = len(fitted_classifier.support_)

        if fitted_classifier.coef_.shape[0] != n_classes - 1:
            raise ValueError(
                "Expected classifier coefficients "
                "to have shape (%d, %d) but got shape %r" 
                % (n_classes, n_sv, fitted_classifier.coef_.shape)
            )

    except NotFittedError:
        fitted_classifier = clone(classifier)

        # Use a precomputed kernel
        # to avoid re-computing K
        fitted_classifier.set_params(kernel="precomputed")
        fitted_classifier.fit(K, y=y)

    return fitted_classifier