import numpy as np
from joblib import (
    Parallel,
    delayed,
)

from ..linear_model import (
    OrthogonalRegression,
    RidgeRegression2FoldCV,
)
from ..model_selection import train_test_split
from ..preprocessing import StandardFlexibleScaler


def pointwise_global_reconstruction_error(
    X,
    Y,
    train_idx=None,
    test_idx=None,
    scaler=None,
    estimator=None,
):
    r"""Computes the pointwise global reconstruction error using the source X
    to reconstruct the features or samples of target Y based on a minimization
    by linear regression:

    .. math::

        GRE^{(i)}(X,Y) = \min_W ||y_i - x_iW||

    If used with X and Y of shape (n_samples, n_features) it computes the pointwise
    global reconstruction error of the features as defined in Ref. [Goscinski2021]_.
    In this case the number of samples of X and Y should agree with each other,
    but the number of features can be different. The error is expressed per sample.

    If used with X and Y of shape(n_features, n_samples) it computes the
    reconstruction error of the samples. In this case the number of
    features of X and Y should agree with each other, but the number of
    samples can be different. The error is expressed per feature.

    The default parameters mimics the ones of Ref. [Goscinski2021]_.

    Parameters
    ----------
    X : ndarray of shape (n_samples, X_n_features)
        Source data which reconstructs target Y.
        For feature reconstruction of Y using X use input shape (samples, features).
        For sample reconstruction of Y using X use input shape (features, samples).

    Y : ndarray of shape (n_samples, Y_n_targets)
        Target data which is reconstructed with X.
        For feature reconstruction of Y using X use input shape (samples, features).
        For sample reconstruction of Y using X use input shape (features, samples).

    train_idx : ndarray, dtype=int, default=None
        array of indices used for training, if None,
        If None, the complement of the ``test_idx`` is used. If ``train_size`` is
        also None, 2-fold split is taken.

    test_idx : ndarray, dtype=int, default=None
        array of indices used for training, if None,
        If None, the complement of the ``train_idx`` is used. If ``test_size`` is
        also None, 2-fold split is taken.

    scaler : object implementing fit/transfom
        Scales the X and Y before computing the reconstruction measure.
        The default value scales the features such that the reconstruction
        measure on the training set is upper bounded to 1.

    estimator : object implementing fit/predict, default=None
        Sklearn estimator used to reconstruct features/samples.

    Returns
    -------
    pointwise_global_reconstruction_error : ndarray
        The global reconstruction error for each sample/point

    """
    (
        train_idx,
        test_idx,
        scaler,
        estimator,
    ) = check_global_reconstruction_measures_input(
        X, Y, train_idx, test_idx, scaler, estimator
    )
    X_train, X_test, Y_train, Y_test = (
        X[train_idx],
        X[test_idx],
        Y[train_idx],
        Y[test_idx],
    )

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    scaler.fit(Y_train)
    Y_train = scaler.transform(Y_train)
    Y_test = scaler.transform(Y_test)

    estimator.fit(X_train, Y_train)

    return np.linalg.norm(Y_test - estimator.predict(X_test), axis=1)


def global_reconstruction_error(
    X,
    Y,
    test_idx=None,
    train_idx=None,
    scaler=None,
    estimator=None,
):
    r"""Computes the global reconstruction error using the source X
    to reconstruct the features or samples of target Y based on a minimization
    by linear regression:

    .. math::

        GRE(X,Y) = \min_W ||Y - XW||

    If used with X and Y of shape (n_samples, n_features) it computes the
    global reconstruction error of the features as defined in Ref. [Goscinski2021]_.
    In this case the number of samples of X and Y should agree with each other,
    but the number of features can be different. The error is expressed per sample.

    If used with X and Y of shape(n_features, n_samples) it computes the
    reconstruction error of the samples. In this case the number of
    features of X and Y should agree with each other, but the number of
    samples can be different. The error is expressed per feature.

    The default parameters mimics the ones of Ref. [Goscinski2021]_.

    Parameters
    ----------
    X : ndarray of shape (n_samples, X_n_features)
        Source data which reconstructs target Y.
        For feature reconstruction of Y using X use input shape (samples, features).
        For sample reconstruction of Y using X use input shape (features, samples).

    Y : ndarray of shape (n_samples, Y_n_targets)
        Target data which is reconstructed with X.
        For feature reconstruction of Y using X use input shape (samples, features).
        For sample reconstruction of Y using X use input shape (features, samples).

    train_idx : ndarray, dtype=int, default=None
        array of indices used for training, if None,
        If None, the complement of the ``test_idx`` is used. If ``train_size`` is
        also None, 2-fold split is taken.

    test_idx : ndarray, dtype=int, default=None
        array of indices used for training, if None,
        If None, the complement of the ``train_idx`` is used. If ``test_size`` is
        also None, 2-fold split is taken.

    scaler : object implementing fit/transfom
        Scales the X and Y before computing the reconstruction measure.
        The default value scales the features such that the reconstruction
        measure on the training set is upper bounded to 1.

    estimator : object implementing fit/predict, default=None
        Sklearn estimator used to reconstruct features/samples.

    Returns
    -------
    global_reconstruction_error : ndarray
        The global reconstruction error

    """
    pointwise_global_reconstruction_error_values = (
        pointwise_global_reconstruction_error(
            X,
            Y,
            train_idx=train_idx,
            test_idx=test_idx,
            scaler=scaler,
            estimator=estimator,
        )
    )
    return np.linalg.norm(pointwise_global_reconstruction_error_values) / np.sqrt(
        len(pointwise_global_reconstruction_error_values)
    )


def pointwise_global_reconstruction_distortion(
    X,
    Y,
    test_idx=None,
    train_idx=None,
    scaler=None,
    estimator=None,
):
    r"""Computes the pointwise global reconstruction distortion using the source X
    to reconstruct the features or samples of target Y based on a minimization
    by orthogonal regression:

    .. math::

        GRD^{(i)}(X,Y) = \min_Q ||y_i - x_iQ\|| \quad\mathrm{subject\ to}\quad Q^TQ=I

    If used with X and Y of shape (n_samples, n_features) it computes the pointwise
    global reconstruction distortion of the features as defined in Ref. [Goscinski2021]_.
    In this case the number of samples of X and Y should agree with each other,
    but the number of features can be different. The distortion is expressed per sample.

    If used with X and Y of shape(n_features, n_samples) it computes the
    reconstruction distortion of the samples. In this case the number of
    features of X and Y should agree with each other, but the number of
    samples can be different. The distortion is expressed per feature.

    The default parameters mimics the ones of Ref. [Goscinski2021]_.

    Parameters
    ----------
    X : ndarray of shape (n_samples, X_n_features)
        Source data which reconstructs target Y.
        For feature reconstruction of Y using X use input shape (samples, features).
        For sample reconstruction of Y using X use input shape (features, samples).

    Y : ndarray of shape (n_samples, Y_n_targets)
        Target data which is reconstructed with X.
        For feature reconstruction of Y using X use input shape (samples, features).
        For sample reconstruction of Y using X use input shape (features, samples).

    train_idx : ndarray, dtype=int, default=None
        array of indices used for training, if None,
        If None, the complement of the ``test_idx`` is used. If ``train_size`` is
        also None, 2-fold split is taken.

    test_idx : ndarray, dtype=int, default=None
        array of indices used for training, if None,
        If None, the complement of the ``train_idx`` is used. If ``test_size`` is
        also None, 2-fold split is taken.

    scaler : object implementing fit/transfom
        Scales the X and Y before computing the reconstruction measure.
        The default value scales the features such that the reconstruction
        measure on the training set is upper bounded to 1.

    estimator : object implementing fit/predict, default=None
        Sklearn estimator used to reconstruct features/samples.

    Returns
    -------
    pointwise_global_reconstruction_distortion : ndarray
        The global reconstruction distortion for each sample/point

    """
    (
        train_idx,
        test_idx,
        scaler,
        estimator,
    ) = check_global_reconstruction_measures_input(
        X, Y, train_idx, test_idx, scaler, estimator
    )
    X_train, X_test, Y_train, Y_test = (
        X[train_idx],
        X[test_idx],
        Y[train_idx],
        Y[test_idx],
    )

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    scaler.fit(Y_train)
    Y_train = scaler.transform(Y_train)
    Y_test = scaler.transform(Y_test)

    predictions_Y_test = estimator.fit(X_train, Y_train).predict(X_test)
    orthogonal_predictions_Y_test = (
        OrthogonalRegression(use_orthogonal_projector=False)
        .fit(X_train, estimator.predict(X_train))
        .predict(X_test)
    )

    return np.linalg.norm(predictions_Y_test - orthogonal_predictions_Y_test, axis=1)


def global_reconstruction_distortion(
    X,
    Y,
    test_idx=None,
    train_idx=None,
    scaler=None,
    estimator=None,
):
    r"""Computes the global reconstruction distortion using the source X
    to reconstruct the features or samples of target Y based on a minimization
    by orthogonal regression:

    .. math::

        GRD(X,Y) = \min_Q ||y - XQ\|| \quad\mathrm{subject\ to}\quad Q^TQ=I

    If used with X and Y of shape (n_samples, n_features) it computes the
    global reconstruction distortion of the features as defined in Ref. [Goscinski2021]_.
    In this case the number of samples of X and Y should agree with each other,
    but the number of features can be different. The distortion is expressed per sample.

    If used with X and Y of shape(n_features, n_samples) it computes the
    reconstruction distortion of the samples. In this case the number of
    features of X and Y should agree with each other, but the number of
    samples can be different. The distortion is expressed per feature.

    The default parameters mimics the ones of Ref. [Goscinski2021]_.

    Parameters
    ----------
    X : ndarray of shape (n_samples, X_n_features)
        Source data which reconstructs target Y.
        For feature reconstruction of Y using X use input shape (samples, features).
        For sample reconstruction of Y using X use input shape (features, samples).

    Y : ndarray of shape (n_samples, Y_n_targets)
        Target data which is reconstructed with X.
        For feature reconstruction of Y using X use input shape (samples, features).
        For sample reconstruction of Y using X use input shape (features, samples).

    train_idx : ndarray, dtype=int, default=None
        array of indices used for training, if None,
        If None, the complement of the ``test_idx`` is used. If ``train_size`` is
        also None, 2-fold split is taken.

    test_idx : ndarray, dtype=int, default=None
        array of indices used for training, if None,
        If None, the complement of the ``train_idx`` is used. If ``test_size`` is
        also None, 2-fold split is taken.

    scaler : object implementing fit/transfom
        Scales the X and Y before computing the reconstruction measure.
        The default value scales the features such that the reconstruction
        measure on the training set is upper bounded to 1.

    estimator : object implementing fit/predict, default=None
        Sklearn estimator used to reconstruct features/samples.

    Returns
    -------
    global_reconstruction_distortion : ndarray
        The global reconstruction distortion

    """
    pointwise_global_reconstruction_distortion_values = (
        pointwise_global_reconstruction_distortion(
            X,
            Y,
            train_idx=train_idx,
            test_idx=test_idx,
            scaler=scaler,
            estimator=estimator,
        )
    )
    return np.linalg.norm(pointwise_global_reconstruction_distortion_values) / np.sqrt(
        len(pointwise_global_reconstruction_distortion_values)
    )


def pointwise_local_reconstruction_error(
    X,
    Y,
    n_local_points,
    test_idx=None,
    train_idx=None,
    scaler=None,
    estimator=None,
    n_jobs=None,
):
    r"""Computes the pointwise local reconstruction error using the source X
    to reconstruct the features or samples of target Y based on a minimization
    by linear regression:

    .. math::

        \tilde{\mathbf{x}}'_i = \bar{\mathbf{x}} + (\mathbf{x}_i
                                - \bar{\mathbf{x}})\mathbf{P}^{(i)}

    .. math::

        LRE^{(i)}(X,Y) = \|\mathbf{x}'_i - \tilde{\mathbf{x}}'_i\|^2

    If used with X and Y of shape (n_samples, n_features) it computes the pointwise
    local reconstruction error of the features as defined in Ref. [Goscinski2021]_.
    In this case the number of samples of X and Y should agree with each other,
    but the number of features can be different. The error is expressed per sample.

    If used with X and Y of shape(n_features, n_samples) it computes the
    reconstruction error of the samples. In this case the number of
    features of X and Y should agree with each other, but the number of
    samples can be different. The error is expressed per feature.

    The default parameters mimics the ones of Ref. [Goscinski2021]_.

    Parameters
    ----------
    X : ndarray of shape (n_samples, X_n_features)
        Source data which reconstructs target Y.
        For feature reconstruction of Y using X use input shape (samples, features).
        For sample reconstruction of Y using X use input shape (features, samples).

    Y : ndarray of shape (n_samples, Y_n_targets)
        Target data which is reconstructed with X.
        For feature reconstruction of Y using X use input shape (samples, features).
        For sample reconstruction of Y using X use input shape (features, samples).

    n_local_points : int,
        Number of neighbour points used to compute the local reconstruction weight for
        each sample/point.

    train_idx : ndarray, dtype=int, default=None
        array of indices used for training, if None,
        If None, the complement of the ``test_idx`` is used. If ``train_size`` is
        also None, 2-fold split is taken.

    test_idx : ndarray, dtype=int, default=None
        array of indices used for training, if None,
        If None, the complement of the ``train_idx`` is used. If ``test_size`` is
        also None, 2-fold split is taken.

    scaler : object implementing fit/transfom
        Scales the X and Y before computing the reconstruction measure.
        The default value scales the features such that the reconstruction
        measure on the training set is upper bounded to 1.

    estimator : object implementing fit/predict, default=None
        Sklearn estimator used to reconstruct features/samples.

    Returns
    -------
    pointwise_local_reconstruction_error : ndarray
        The local reconstruction error for each sample/point

    """
    (
        train_idx,
        test_idx,
        scaler,
        estimator,
    ) = check_local_reconstruction_measures_input(
        X, Y, n_local_points, train_idx, test_idx, scaler, estimator
    )
    X_train, X_test, Y_train, Y_test = (
        X[train_idx],
        X[test_idx],
        Y[train_idx],
        Y[test_idx],
    )

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    scaler.fit(Y_train)
    Y_train = scaler.transform(Y_train)
    Y_test = scaler.transform(Y_test)

    squared_dist = (
        np.sum(X_train**2, axis=1)
        + np.sum(X_test**2, axis=1)[:, np.newaxis]
        - 2 * X_test @ X_train.T
    )

    n_test = X_test.shape[0]

    def local_reconstruction_error_i(i):
        # comments correspond notation in [gfrm]_
        local_env_idx = np.argsort(squared_dist[i])[:n_local_points]
        # D_{k-neigh}^{(i)}
        local_X_train = X_train[local_env_idx]
        # \bar{x}_F
        local_X_train_mean = np.mean(X_train[local_env_idx], axis=0)
        # D_{k-neigh}^{(i)}
        local_Y_train = Y_train[local_env_idx]
        # \bar{x}_F'
        local_Y_train_mean = np.mean(Y_train[local_env_idx], axis=0)
        # P_{FF'}
        estimator.fit(
            local_X_train - local_X_train_mean,
            local_Y_train - local_Y_train_mean,
        )
        # \tilde{x}_i' = \bar{x}_{F'} + (x_i - \bar{x}_F)P_{FF'}
        tilde_x_i_dash_test = local_Y_train_mean + estimator.predict(
            X_test[i, :][np.newaxis, :] - local_X_train_mean
        )
        # \|x_i' -  \tilde{x}_i'\|
        return np.linalg.norm(Y_test[i, :][np.newaxis, :] - tilde_x_i_dash_test)

    pointwise_local_reconstruction_error_values = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(local_reconstruction_error_i)(i) for i in range(n_test)
        )
    )
    return pointwise_local_reconstruction_error_values


def local_reconstruction_error(
    X,
    Y,
    n_local_points,
    test_idx=None,
    train_idx=None,
    scaler=None,
    estimator=None,
    n_jobs=None,
):
    r"""Computes the local reconstruction error using the source X
    to reconstruct the features or samples of target Y based on a minimization
    by linear regression:

    .. math::

        LRE(X,Y) = \sqrt{\sum_i LRE^{(i)}(X,Y)}/\sqrt{n_\text{test}}

    If used with X and Y of shape (n_samples, n_features) it computes the
    local reconstruction error of the features as defined in Ref. [Goscinski2021]_.
    In this case the number of samples of X and Y should agree with each other,
    but the number of features can be different. The error is expressed per sample.

    If used with X and Y of shape(n_features, n_samples) it computes the
    reconstruction error of the samples. In this case the number of
    features of X and Y should agree with each other, but the number of
    samples can be different. The error is expressed per feature.

    The default parameters mimics the ones of Ref. [Goscinski2021]_.

    Parameters
    ----------
    X : ndarray of shape (n_samples, X_n_features)
        Source data which reconstructs target Y.
        For feature reconstruction of Y using X use input shape (samples, features).
        For sample reconstruction of Y using X use input shape (features, samples).

    Y : ndarray of shape (n_samples, Y_n_targets)
        Target data which is reconstructed with X.
        For feature reconstruction of Y using X use input shape (samples, features).
        For sample reconstruction of Y using X use input shape (features, samples).

    n_local_points : int,
        Number of neighbour points used to compute the local reconstruction weight for
        each sample/point.

    train_idx : ndarray, dtype=int, default=None
        array of indices used for training, if None,
        If None, the complement of the ``test_idx`` is used. If ``train_size`` is
        also None, 2-fold split is taken.

    test_idx : ndarray, dtype=int, default=None
        array of indices used for training, if None,
        If None, the complement of the ``train_idx`` is used. If ``test_size`` is
        also None, 2-fold split is taken.

    scaler : object implementing fit/transfom
        Scales the X and Y before computing the reconstruction measure.
        The default value scales the features such that the reconstruction
        measure on the training set is upper bounded to 1.

    estimator : object implementing fit/predict, default=None
        Sklearn estimator used to reconstruct features/samples.

    Returns
    -------
    local_reconstruction_error : ndarray
        The local reconstruction error

    """
    pointwise_local_reconstruction_error_values = pointwise_local_reconstruction_error(
        X,
        Y,
        n_local_points,
        train_idx=train_idx,
        test_idx=test_idx,
        scaler=scaler,
        estimator=estimator,
        n_jobs=n_jobs,
    )
    return np.linalg.norm(pointwise_local_reconstruction_error_values) / np.sqrt(
        len(pointwise_local_reconstruction_error_values)
    )


def check_global_reconstruction_measures_input(
    X, Y, train_idx, test_idx, scaler, estimator
):
    """Returns default reconstruction measure inputs for all None parameters"""
    assert len(X) == len(Y)
    if (train_idx is None) and (test_idx is None):
        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=0.5,
            train_size=0.5,
            random_state=0x5F3759DF,
            shuffle=True,
            train_test_overlap=False,
        )
    elif train_idx is None:
        train_idx = np.setdiff1d(np.arange(len(X)), test_idx)
    elif test_idx is None:
        test_idx = np.setdiff1d(np.arange(len(X)), train_idx)

    if scaler is None:
        scaler = StandardFlexibleScaler()

    if estimator is None:
        estimator = RidgeRegression2FoldCV(
            alphas=np.geomspace(1e-9, 0.9, 20),
            alpha_type="relative",
            regularization_method="cutoff",
            random_state=0x5F3759DF,
            shuffle=True,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
    return train_idx, test_idx, scaler, estimator


def check_local_reconstruction_measures_input(
    X, Y, n_local_points, train_idx, test_idx, scaler, estimator
):
    """Returns default reconstruction measure inputs for all None parameters"""
    # only needs to check one extra parameter
    assert len(X) >= n_local_points
    return check_global_reconstruction_measures_input(
        X, Y, train_idx, test_idx, scaler, estimator
    )
