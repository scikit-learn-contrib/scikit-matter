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
        if self.n_samples_seen_ == 0:
            raise sk.exceptions.NotFittedError(
                "This "
                + type(self).__name__
                + " instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )
        if self.n_features_ != X_tr.shape[1]:
            raise ValueError("X shape does not match training shape")
        return X_tr * self.scale_ + self.mean_


class KernelFlexibleCenterer(TransformerMixin, BaseEstimator):
    """Kernel centering method, similar to KernelCenterer,
    but with additional parameters, relative to which centering
    is carried out:


    :param ref_cmean: an array with means for each column.
    :type ref_cmean: array of shape (1, n_features)

    :param ref_mean: an average for the whole kernel matrix
    :type ref_mean: array
    """

    def __init__(self, ref_cmean=None, ref_mean=None):
        """Initialize KernelFlexibleCenterer."""
        pass

    def fit(self, K=None, y=None, ref_cmean=None, ref_mean=None):
        """Fit KernelFlexibleCenterer

        :param K: Kernel matrix
        :type K: ndarray of shape (n_samples, n_samples)
        :param y: ignored
        :param ref_cmean: an array with means for each column.
        :type ref_cmean: array of shape (1, n_features)
        :param ref_mean: an average for the whole kernel matrix
        :type ref_mean: array

        :return: itself
        """
        if K is not None:
            if K.shape[0] != K.shape[1]:
                raise ValueError(
                    "The reference kernel is not square, and does not define a RKHS"
                )

            self.reference_shape = K.shape

            if ref_cmean is not None:
                if K.shape[0] != len(ref_cmean):
                    raise ValueError(
                        "The supplied column mean does not match the supplied kernel."
                    )
            else:
                ref_cmean = K.mean(axis=0)

            if ref_mean is None:
                ref_mean = K.mean()

        else:
            assert ref_cmean is not None and ref_mean is not None
            self.reference_shape = [None, len(ref_cmean)]

        self.ref_cmean = ref_cmean
        self.ref_mean = ref_mean

        return self

    def transform(self, K, y=None):
        """Centering our Kernel. Previously you should fit data.

        :param K: Kernel matrix
        :type K: ndarray of shape (n_samples, n_samples)
        :param y: ignored

        :return: tranformed matrix Kc
        """
        parameters = self.get_params()
        """check each of the parameters self.reference_shape, self.ref_mean,
        and self.ref_cmean, which must all be defined
        """
        for key in parameters.keys():
            if parameters[key] is None:
                raise sk.exceptions.NotFittedError(
                    "This "
                    + type(self).__name__
                    + " instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
                )
        if K.shape[1] != self.reference_shape[1]:
            raise ValueError(
                "The reference kernel and received kernel have different shape"
            )
        rmean = K.mean(axis=1)

        Kc = (
            K
            - np.broadcast_arrays(K, self.ref_cmean)[1]
            - rmean.reshape((K.shape[0], 1))
            + np.broadcast_arrays(K, self.ref_mean)[1]
        )

        return Kc

    def fit_transform(self, K, y=None, ref_cmean=None, ref_mean=None, **fit_params):
        r"""Fit to data, then transform it.

        :param K: Kernel matrix
        :type K: ndarray of shape (n_samples, n_samples)
        :param y: ignored
        :param ref_cmean: an array with means for each column.
        :type ref_cmean: array of shape (1, n_features)
        :param ref_mean: an average for the whole kernel matrix
        :type ref_mean: array
        :param \**fit_params: necessary for compatibility with the functions of
                              the TransformerMixin class

        :return: tranformed matrix Kc
        """
        self.fit(K, y, ref_cmean=ref_cmean, ref_mean=ref_mean)
        return self.transform(K, y)


class SparseKernelCenterer(TransformerMixin, BaseEstimator):
    """Kernel centering method for sparse kernels, similar to
    KernelFlexibleCenterer
    """

    def __init__(self, rcond=1e-12):
        """
        Initialize SparseKernelCenterer.

        :param rcond: conditioning parameter to use when computing the
                      Nystrom-approximated kernel for scaling
        :type rcond: float, default 1E-12
        """

        self.rcond = rcond

    def fit(self, Knm, Kmm, y=None):
        """Fit KernelFlexibleCenterer

        :param Knm: Kernel matrix between the reference data set and the active
                    set
        :type Knm: ndarray of shape (n_samples, n_active)

        :param Kmm: Kernel matrix between the active set and itself
        :type Kmm: ndarray of shape (n_active, n_active)

        :param y: ignored

        :return: itself
        """

        if Knm.shape[1] != Kmm.shape[0]:
            raise ValueError(
                "The reference kernel is not commensurate shape with the"
                "active kernel."
            )

        if Kmm.shape[0] != Kmm.shape[1]:
            raise ValueError("The active kernel is not square.")

        self.n_active_ = Kmm.shape[0]

        self.K_fit_rows_ = Knm.mean(axis=0)

        Knm_centered = Knm - np.broadcast_arrays(Knm, self.K_fit_rows_)[1]
        Kmm_centered = Kmm - np.broadcast_arrays(Kmm, self.K_fit_rows_)[1]

        Khat = Knm_centered @ np.linalg.pinv(Kmm_centered, self.rcond) @ Knm_centered.T

        self.scale_ = np.sqrt(np.trace(Khat) / Knm.shape[0])

        return self

    def transform(self, Knm, y=None):
        """Centering our Kernel. Previously you should fit data.

        :param Knm: Kernel matrix between the reference data set and the active
                    set
        :type Knm: ndarray of shape (n_samples, n_active)
        :param y: ignored

        :return: tranformed matrix Kc

        check each of the parameters self.n_active_, self.scale_
        and self.K_fit_rows_, which must all be defined
        """
        check_is_fitted(self, attributes=["scale_", "K_fit_rows_", "n_active_"])

        if Knm.shape[1] != self.n_active_:
            raise ValueError(
                "The reference kernel and received kernel have different shape"
            )

        Kc = (Knm - np.broadcast_arrays(Knm, self.K_fit_rows_)[1]) / self.scale_

        return Kc

    def fit_transform(self, Knm, Kmm, y=None, **fit_params):
        r"""Fit to data, then transform it.

        :param Knm: Kernel matrix between the reference data set and the active
                    set
        :type Knm: ndarray of shape (n_samples, n_active)

        :param Kmm: Kernel matrix between the active set and itself
        :type Kmm: ndarray of shape (n_active, n_active)

        :param y: ignored

        :param \**fit_params: necessary for compatibility with the functions of
                              the TransformerMixin class

        :return: tranformed matrix Kc
        """
        self.fit(Knm=Knm, Kmm=Kmm)
        return self.transform(Knm)
