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
            print("X shape "+str(X.shape))
            print("y shape " + str(y.shape))
            # Check compatibility with y

            # changed from if fitted_classifier.coef_.ndim != y.ndim:
            # dimension of classifier coefficients is always 2, hence we don't need to check 
            # for match with Y
            if fitted_classifier.coef_.shape[1] != X.shape[1]:
                raise ValueError(
                    "The classifier coefficients have a shape incompatible "
                    "with the supplied feature space. "
                    "The coefficients have shape %d and the features "
                    "have shape %d" % (fitted_classifier.coef_.shape, X.shape)
                )
            # LogisticRegression does not support multioutput, but RidgeClassifier does
            elif y.ndim == 2:
                if fitted_classifier.coef_.shape[0] != y.shape[1]:
                    raise ValueError(
                        "The classifier coefficients have a shape incompatible "
                        "with the supplied target space. "
                        "The coefficients have shape %r and the targets "
                        "have shape %r" % (fitted_classifier.coef_.shape, y.shape)
                    )

        except NotFittedError:
            fitted_classifier = clone(classifier)
            fitted_classifier.fit(X, y)

        return fitted_classifier
