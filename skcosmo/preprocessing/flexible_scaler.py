import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing._data import KernelCenterer
from sklearn.utils.validation import FLOAT_DTYPES


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

        check_is_fitted(self, attributes=["n_samples_seen_", "n_features_"])

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


class KernelNormalizer(KernelCenterer):
    """Kernel centering method, similar to KernelCenterer,
    but with additional scaling and passing of precomputed kernel means.


    Attributes
    ----------
    K_fit_rows_ : array of shape (n_samples,)
        Average of each column of kernel matrix.

    K_fit_all_ : float
        Average of kernel matrix.

    Examples
    --------
    >>> from skcosmo.preprocessing import KernelNormalizer
    >>> from sklearn.metrics.pairwise import pairwise_kernels
    >>> X = [[ 1., -2.,  2.],
    ...      [ -2.,  1.,  3.],
    ...      [ 4.,  1., -2.]]
    >>> K = pairwise_kernels(X, metric='linear')
    >>> K
    array([[  9.,   2.,  -2.],
           [  2.,  14., -13.],
           [ -2., -13.,  21.]])
    >>> transformer = KernelNormalizer().fit(K)
    >>> transformer
    KernelNormalizer()
    >>> transformer.transform(K)
    array([[ 0.39473684,  0.        , -0.39473684],
           [ 0.        ,  1.10526316, -1.10526316],
           [-0.39473684, -1.10526316,  1.5       ]])
    >>> transformer.scale_ * transformer.transform(K)
    array([[  5.,   0.,  -5.],
           [  0.,  14., -14.],
           [ -5., -14.,  19.]])
    >>>
    """

    def __init__(self):
        super().__init__()

    def fit(self, K=None, y=None, K_fit_rows=None, K_fit_all=None):
        """Fit KernelFlexibleCenterer

        :param K: Kernel matrix
        :type K: ndarray of shape (n_samples, n_samples)
        :param y: ignored
        :param K_fit_rows: an array with means for each column.
        :type K_fit_rows: array of shape (1, n_features)
        :param K_fit_all: an average for the whole kernel matrix
        :type K_fit_all: array

        :return: fitted transformer
        """

        if K_fit_rows is not None and K_fit_all is not None:
            self.K_fit_rows_ = K_fit_rows
            self.K_fit_all_ = K_fit_all
        else:
            super().fit(K, y)

        Kc = self._validate_data(K, copy=True, dtype=FLOAT_DTYPES, reset=False)

        K_pred_cols = (np.sum(Kc, axis=1) / self.K_fit_rows_.shape[0])[:, np.newaxis]

        Kc -= self.K_fit_rows_
        Kc -= K_pred_cols
        Kc += self.K_fit_all_

        self.scale_ = np.trace(Kc) / K.shape[0]

        return self

    def transform(self, K, copy=True):
        """Center kernel matrix.

        Parameters
        ----------
        K : ndarray of shape (n_samples1, n_samples2)
            Kernel matrix.

        copy : bool, default=True
            Set to False to perform inplace computation.

        Returns
        -------
        K_new : ndarray of shape (n_samples1, n_samples2)
        """

        return super().transform(K, copy) / self.scale_

    def fit_transform(
        self, K, y=None, copy=True, K_fit_rows=None, K_fit_all=None, **fit_params
    ):
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
        return self.transform(K, copy)


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

        Knm_centered = Knm - self.K_fit_rows_

        Khat = Knm_centered @ np.linalg.pinv(Kmm, self.rcond) @ Knm_centered.T

        self.scale_ = np.sqrt(np.trace(Khat) / Knm.shape[0])

        return self

    def transform(self, Knm, y=None):
        """Centering our Kernel. Previously you should fit data.

        :param Knm: Kernel matrix between the reference data set and the active
                    set
        :type Knm: ndarray of shape (n_samples, n_active)
        :param y: ignored

        :return: tranformed matrix Kc

        """
        check_is_fitted(self, attributes=["scale_", "K_fit_rows_", "n_active_"])

        if Knm.shape[1] != self.n_active_:
            raise ValueError(
                "The reference kernel and received kernel have different shape"
            )

        Kc = (Knm - self.K_fit_rows_) / self.scale_

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
