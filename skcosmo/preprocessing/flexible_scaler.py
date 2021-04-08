import numpy as np
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.preprocessing._data import KernelCenterer
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    _check_sample_weight,
    check_is_fitted,
)


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

    def fit(self, X, y=None, sample_weight=None):
        """Compute mean and scaling to be applied for subsequent normalization.

        :param X: Matrix
        :type X: ndarray
        :param y: ignored
        :param sample_weight: weights for each sample. Sample weighting can be used to center (and scale) data using a weighted mean. Weights are internally normalized before preprocessing.
        :type sample_weight: array of shape (n_samples,)

        :return: itself
        """

        self.n_samples_seen_, self.n_features_ = X.shape

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
            sample_weight = sample_weight / np.sum(sample_weight)

        if self.with_mean:
            self.mean_ = np.average(X, weights=sample_weight, axis=0)
        else:
            self.mean_ = np.zeros(self.n_features_)

        self.scale_ = 1.0
        if self.with_std:
            X_mean = np.average(X, weights=sample_weight, axis=0)
            var = np.average((X - X_mean) ** 2, weights=sample_weight, axis=0)

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

    def fit_transform(self, X, y=None, sample_weight=None, **fit_params):
        r"""Fit to data, then transform it.

        :param X: Matrix
        :type X: ndarray
        :param y: ignored
        :param sample_weight: weights for each sample. Sample weighting can be used to center (and scale) data using a weighted mean. Weights are internally normalized before preprocessing.

        :type sample_weight: array of shape (n_samples,)
        :param \**fit_params: necessary for compatibility with the functions of the TransformerMixin class

        :return: itself
        """
        self.fit(X, y, sample_weight=sample_weight)
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

    :param with_center: If True, center the kernel matrix before scaling. If False, do not center the kernel
    :type with_center: bool
    :param with_trace: If True, scale the kernel so that the trace is equal to the number of samples. If False, do not scale the kernel
    :type with_trace: bool

    Attributes
    ----------
    K_fit_rows_ : array of shape (n_samples,)
        Average of each column of kernel matrix.

    K_fit_all_ : float
        Average of kernel matrix.

    sample_weight_ : float
        Sample weights (if provided during the fit)

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

    def __init__(self, with_center=True, with_trace=True):
        self.with_center = with_center
        self.with_trace = with_trace
        super().__init__()

    def fit(self, K=None, y=None, sample_weight=None):
        """Fit KernelFlexibleCenterer

        :param K: Kernel matrix
        :type K: ndarray of shape (n_samples, n_samples)
        :param y: ignored
        :param sample_weight: weights for each sample. Sample weighting can be used to center (and scale) data using a weighted mean. Weights are internally normalized before preprocessing.

        :type sample_weight: array of shape (n_samples,)
        :return: fitted transformer
        """

        Kc = self._validate_data(K, copy=True, dtype=FLOAT_DTYPES, reset=False)

        if sample_weight is not None:
            self.sample_weight_ = _check_sample_weight(sample_weight, K, dtype=K.dtype)
            self.sample_weight_ = self.sample_weight_ / np.sum(self.sample_weight_)
        else:
            self.sample_weight_ = sample_weight

        if self.with_center:
            if self.sample_weight_ is not None:
                self.K_fit_rows_ = np.average(K, weights=self.sample_weight_, axis=0)
                self.K_fit_all_ = np.average(
                    self.K_fit_rows_, weights=self.sample_weight_
                )
            else:
                super().fit(K, y)

            K_pred_cols = np.average(Kc, weights=self.sample_weight_, axis=1)[
                :, np.newaxis
            ]
        else:
            self.K_fit_rows_ = np.zeros(Kc.shape[1])
            self.K_fit_all_ = 0.0
            K_pred_cols = np.zeros((Kc.shape[0], 1))

        if self.with_trace:

            Kc -= self.K_fit_rows_
            Kc -= K_pred_cols
            Kc += self.K_fit_all_

            self.scale_ = np.trace(Kc) / Kc.shape[0]
        else:
            self.scale_ = 1.0

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

        check_is_fitted(self)
        K = self._validate_data(K, copy=copy, dtype=FLOAT_DTYPES, reset=False)

        if self.with_center:
            K_pred_cols = np.average(K, weights=self.sample_weight_, axis=1)[
                :, np.newaxis
            ]
        else:
            K_pred_cols = np.zeros((K.shape[0], 1))

        K -= self.K_fit_rows_
        K -= K_pred_cols
        K += self.K_fit_all_

        return K / self.scale_

    def fit_transform(self, K, y=None, sample_weight=None, copy=True, **fit_params):
        r"""Fit to data, then transform it.

        :param K: Kernel matrix
        :type K: ndarray of shape (n_samples, n_samples)
        :param y: ignored
        :param sample_weight: weights for each sample. Sample weighting can be used to center (and scale) data using a weighted mean. Weights are internally normalized before preprocessing.

        :type sample_weight: array of shape (n_samples,)
        :param \**fit_params: necessary for compatibility with the functions of the TransformerMixin class

        :return: tranformed matrix Kc
        """
        self.fit(K, y, sample_weight=sample_weight)
        return self.transform(K, copy)


class SparseKernelCenterer(TransformerMixin, BaseEstimator):
    """Kernel centering method for sparse kernels, similar to
    KernelFlexibleCenterer


    """

    def __init__(self, with_center=True, with_trace=True, rcond=1e-12):
        """
        Initialize SparseKernelCenterer.

        :param with_center: If True, center the kernel matrix before scaling. If False, do not center the kernel
        :type with_center: bool
        :param with_trace: If True, scale the kernel so that the trace is equal to the number of samples. If False, do not scale the kernel
        :type with_trace: bool
        :param rcond: conditioning parameter to use when computing the
                      Nystrom-approximated kernel for scaling
        :type rcond: float, default 1E-12
        """

        self.with_center = with_center
        self.with_trace = with_trace
        self.rcond = rcond

    def fit(self, Knm, Kmm, y=None, sample_weight=None):
        """Fit KernelFlexibleCenterer

        :param Knm: Kernel matrix between the reference data set and the active
                    set
        :type Knm: ndarray of shape (n_samples, n_active)

        :param Kmm: Kernel matrix between the active set and itself
        :type Kmm: ndarray of shape (n_active, n_active)

        :param y: ignored
        :param sample_weight: weights for each sample. Sample weighting can be used to center (and scale) data using a weighted mean. Weights are internally normalized before preprocessing.

        :type sample_weight: array of shape (n_samples,)

        :return: itself
        """

        if Knm.shape[1] != Kmm.shape[0]:
            raise ValueError(
                "The reference kernel is not commensurate shape with the"
                "active kernel."
            )

        if Kmm.shape[0] != Kmm.shape[1]:
            raise ValueError("The active kernel is not square.")

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, Knm, dtype=Knm.dtype)
            sample_weight = sample_weight / np.sum(sample_weight)

        self.n_active_ = Kmm.shape[0]

        if self.with_center:
            self.K_fit_rows_ = np.average(Knm, weights=sample_weight, axis=0)
        else:
            self.K_fit_rows_ = np.zeros(Knm.shape[1])

        if self.with_trace:
            Knm_centered = Knm - self.K_fit_rows_

            Khat = Knm_centered @ np.linalg.pinv(Kmm, self.rcond) @ Knm_centered.T

            self.scale_ = np.sqrt(np.trace(Khat) / Knm.shape[0])
        else:
            self.scale_ = 1.0

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

    def fit_transform(self, Knm, Kmm, y=None, sample_weight=None, **fit_params):
        r"""Fit to data, then transform it.

        :param Knm: Kernel matrix between the reference data set and the active
                    set
        :type Knm: ndarray of shape (n_samples, n_active)

        :param Kmm: Kernel matrix between the active set and itself
        :type Kmm: ndarray of shape (n_active, n_active)

        :param y: ignored
        :param sample_weight: weights for each sample. Sample weighting can be used to center (and scale) data using a weighted mean. Weights are internally normalized before preprocessing.

        :type sample_weight: array of shape (n_samples,)

        :param \**fit_params: necessary for compatibility with the functions of
                              the TransformerMixin class

        :return: tranformed matrix Kc
        """
        self.fit(Knm=Knm, Kmm=Kmm, sample_weight=sample_weight)
        return self.transform(Knm)
