import numpy as np
import sklearn as sk
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted


class StandardFlexibleScaler(TransformerMixin, BaseEstimator):
    """Standardize features by removing the mean and scaling to unit variance.
    Make the mean of the columns equal to zero and the
    variance of each column (`column_wise==True`) equal to one.

    :param with_mean: If True, center the data before scaling. If False, keep the mean intact
    :type with_mean: boolean
    :param with_std: If True, scale the data to unit variance. If False, keep the variance intact
    :type with_std: boolean
    :param column_wise: If True, normalize each column separately. If False, normalize the whole matrix, divided it by variaton.
    :type column_wise: boolean
    :param tol: The tolerance for the optimization: if the variance are smaller than tol, it is considered zero.
    """

    def __init__(self, with_mean=True, with_std=True, column_wise=False, tol=1e-15):
        """Initialize StandardFlexibleScaler."""
        self.with_mean = with_mean
        self.with_std = with_std
        self.column_wise = column_wise
        self.n_samples_seen_ = 0
        self.tol = tol

    def fit(self, X, y=None):
        """Compute mean and scaling to be applied for subsequent normalization.

        :param X: Matrix
        :type X: ndarray
        :param y: ignored

        :return: itself
        """

        self.n_samples_seen_, self.n_features_ = X.shape
        if self.with_mean:
            self.mean_ = X.mean(axis=0)
        else:
            self.mean_ = np.zeros(self.n_features_)

        self.scale_ = 1.0
        if self.with_std:
            var = ((X - X.mean(axis=0)) ** 2).mean(axis=0)

            if self.column_wise:
                if np.any(var < self.tol):
                    raise ValueError("Cannot normalize a feature with zero variance")
                self.scale_ = np.sqrt(var)
            else:
                var_sum = var.sum()
                if var_sum < self.tol:
                    raise ValueError("Cannot normalize a matrix with zero variance")
                self.scale_ = np.sqrt(var_sum)

        return self

    def transform(self, X, y=None):
        """Normalize a vector based on previously computed mean and scaling.

        :param X: Matrix
        :type X: ndarray
        :param y: ignored

        :return: transformed matrix X
        """

        if self.n_samples_seen_ == 0:
            raise sk.exceptions.NotFittedError(
                "This "
                + type(self).__name__
                + " instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )
        if self.n_features_ != X.shape[1]:
            raise ValueError("X shape does not match training shape")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None, **fit_params):
        r"""Fit to data, then transform it.

        :param X: Matrix
        :type X: ndarray
        :param y: ignored
        :param \**fit_params: necessary for compatibility with the functions of the TransformerMixin class

        :return: itself
        """
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform(self, X_tr):
        """Scale back the data to the original representation

        :param X_tr: Matrix
        :type X_tr: ndarray

        :return: original matrix X
        """

        check_is_fitted(self, attributes=["n_samples_seen_", "n_features_"])

        if self.n_features_ != X_tr.shape[1]:
            raise ValueError("X shape does not match training shape")
        return X_tr * self.scale_ + self.mean_


class KernelFlexibleCenterer(TransformerMixin, BaseEstimator):
    """Kernel centering method, similar to KernelCenterer,
    but with additional parameters, relative to which centering
    is carried out:

    """

    def __init__(self):
        """Initialize KernelFlexibleCenterer."""
        pass

    def fit(self, K=None, y=None, K_fit_rows=None, K_fit_all=None):
        """Fit KernelFlexibleCenterer

        :param K: Kernel matrix
        :type K: ndarray of shape (n_samples, n_samples)
        :param y: ignored
        :param K_fit_rows: an array with means for each column.
        :type K_fit_rows: array of shape (1, n_features)
        :param K_fit_all: an average for the whole kernel matrix
        :type K_fit_all: array

        :return: itself
        """
        if K is not None:
            if K.shape[0] != K.shape[1]:
                raise ValueError(
                    "The reference kernel is not square, and does not define a RKHS"
                )

            self.reference_shape_ = K.shape

            if K_fit_rows is not None:
                if K.shape[0] != len(K_fit_rows):
                    raise ValueError(
                        "The supplied column mean does not match the supplied kernel."
                    )
            else:
                K_fit_rows = K.mean(axis=0)

            if K_fit_all is None:
                K_fit_all = K.mean()

        else:
            assert K_fit_rows is not None and K_fit_all is not None
            self.reference_shape_ = [None, len(K_fit_rows)]

        self.K_fit_rows_ = K_fit_rows
        self.K_fit_all_ = K_fit_all

        Kc = (
            K
            - np.broadcast_arrays(K, self.K_fit_rows_)[1]
            - np.mean(K, axis=1).reshape((K.shape[0], 1))
            + np.broadcast_arrays(K, self.K_fit_all_)[1]
        )

        self.scale_ = np.trace(Kc) / K.shape[0]

        return self

    def transform(self, K, y=None):
        """Centering our Kernel. Previously you should fit data.

        :param K: Kernel matrix
        :type K: ndarray of shape (n_samples, n_samples)
        :param y: ignored

        :return: tranformed matrix Kc

        check each of the parameters self.reference_shape_, self.scale_, self.K_fit_all_,
        and self.K_fit_rows_, which must all be defined
        """

        check_is_fitted(
            self, attributes=["K_fit_rows_", "K_fit_all_", "scale_", "reference_shape_"]
        )

        if K.shape[1] != self.reference_shape_[1]:
            raise ValueError(
                "The reference kernel and received kernel have different shape"
            )
        rmean = K.mean(axis=1)

        Kc = (
            K
            - np.broadcast_arrays(K, self.K_fit_rows_)[1]
            - rmean.reshape((K.shape[0], 1))
            + np.broadcast_arrays(K, self.K_fit_all_)[1]
        ) / self.scale_

        return Kc

    def fit_transform(self, K, y=None, K_fit_rows=None, K_fit_all=None, **fit_params):
        r"""Fit to data, then transform it.

        :param K: Kernel matrix
        :type K: ndarray of shape (n_samples, n_samples)
        :param y: ignored
        :param K_fit_rows: an array with means for each column.
        :type K_fit_rows: array of shape (1, n_features)
        :param K_fit_all: an average for the whole kernel matrix
        :type K_fit_all: array
        :param \**fit_params: necessary for compatibility with the functions of the TransformerMixin class

        :return: tranformed matrix Kc
        """
        self.fit(K, y, K_fit_rows=K_fit_rows, K_fit_all=K_fit_all)
        return self.transform(K, y)
