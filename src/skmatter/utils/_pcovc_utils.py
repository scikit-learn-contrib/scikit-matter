from copy import deepcopy
from sklearn import clone
from sklearn.base import check_is_fitted
from sklearn.exceptions import NotFittedError

def check_cl_fit(classifier, X, y):
        try:
            check_is_fitted(classifier)
            fitted_classifier = deepcopy(classifier)

            # Check compatibility with X
            fitted_classifier._validate_data(X, y, reset=False, multi_output=True)
   
            # # Check compatibility with y
            # # changed from if fitted_classifier.coef_.ndim != y.ndim:
            # # dimension of classifier coefficients is always 2, hence we don't need to check 
            # # dimension
            # # for match with Y
            # # LogisticRegression does not support multioutput, but RidgeClassifier does.
            # # We need to check this...
            # # if fitted_classifier.coef_.shape[0] != y.shape[1]:
            # #     raise ValueError(
            # #         "The classifier coefficients have a shape incompatible "
            # #         "with the supplied target space. "
            # #         "The coefficients have shape %r and the targets "
            # #         "have shape %r" % (fitted_classifier.coef_.shape, y.shape)
            # #     )

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