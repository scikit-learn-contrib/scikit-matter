import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.utils import check_random_state, extmath

from skmatter.datasets import load_degenerate_CH4_manifold
from skmatter.metrics import (
    check_global_reconstruction_measures_input,
    check_local_reconstruction_measures_input,
    componentwise_prediction_rigidity,
    global_reconstruction_distortion,
    global_reconstruction_error,
    local_prediction_rigidity,
    local_reconstruction_error,
    pairwise_mahalanobis_distances,
    periodic_pairwise_euclidean_distances,
    pointwise_local_reconstruction_error,
)


@pytest.fixture(scope="module")
def pr_features():
    """Fixture for prediction rigidity features."""
    soap_features = load_degenerate_CH4_manifold().data["SOAP_power_spectrum"]
    soap_features = soap_features[:11]
    # each structure in CH4 has 5 environmental feature, because there are 5 atoms
    # per structure and each atom is one environment
    features = [
        soap_features[i * 5 : (i + 1) * 5] for i in range(len(soap_features) // 5)
    ]
    # add a single environment structure to check value
    features = features + [soap_features[-1:]]
    return features


@pytest.fixture(scope="module")
def pr_comp_features():
    """Fixture for componentwise prediction rigidity features."""
    soap_features = load_degenerate_CH4_manifold().data["SOAP_power_spectrum"]
    soap_features = soap_features[:11]
    bi_features = load_degenerate_CH4_manifold().data["SOAP_bispectrum"]
    bi_features = bi_features[:11]
    comp_features = np.column_stack([soap_features, bi_features])
    comp_features = [
        comp_features[i * 5 : (i + 1) * 5] for i in range(len(comp_features) // 5)
    ]
    return comp_features


@pytest.fixture(scope="module")
def pr_comp_dims():
    """Fixture for component dimensions."""
    soap_features = load_degenerate_CH4_manifold().data["SOAP_power_spectrum"]
    soap_features = soap_features[:11]
    bi_features = load_degenerate_CH4_manifold().data["SOAP_bispectrum"]
    bi_features = bi_features[:11]
    return np.array([soap_features.shape[1], bi_features.shape[1]])


@pytest.fixture(scope="module")
def alpha():
    """Alpha parameter for prediction rigidity."""
    return 1e-8


def test_local_prediction_rigidity(pr_features, alpha):
    LPR, rank_diff = local_prediction_rigidity(pr_features, pr_features, alpha)
    msg = (
        "LPR of the single environment structure is incorrectly lower than 1: "
        f"LPR = {LPR[-1]}"
    )
    assert LPR[-1] >= 1, msg
    assert rank_diff == 0, (
        f"LPR Covariance matrix rank is not full, with a difference of:{rank_diff}"
    )


def test_componentwise_prediction_rigidity(pr_comp_features, alpha, pr_comp_dims):
    _CPR, _LCPR, _rank_diff = componentwise_prediction_rigidity(
        pr_comp_features, pr_comp_features, alpha, pr_comp_dims
    )


@pytest.fixture(scope="module")
def rm_features_small():
    """Fixture for small reconstruction measures features."""
    features = load_iris().data
    return features[:20, [0, 1]]


@pytest.fixture(scope="module")
def rm_features_large():
    """Fixture for large reconstruction measures features."""
    features = load_iris().data
    return features[:20, [0, 1, 0, 1]]


@pytest.fixture(scope="module")
def rm_features_rotated_small(rm_features_small):
    """Fixture for rotated small reconstruction measures features."""
    random_state = 0
    random_state = check_random_state(random_state)
    random_orthonormal_mat = extmath.randomized_range_finder(
        np.eye(rm_features_small.shape[1]),
        size=rm_features_small.shape[1],
        n_iter=10,
        random_state=random_state,
    )
    return rm_features_small @ random_orthonormal_mat


@pytest.fixture(scope="module")
def eps():
    """Tolerance for reconstruction measures."""
    return 1e-5


@pytest.fixture(scope="module")
def n_local_points():
    """Number of local points for reconstruction measures."""
    return 15


def test_global_reconstruction_error_identity(rm_features_large, eps):
    gfre_val = global_reconstruction_error(rm_features_large, rm_features_large)
    assert abs(gfre_val) < eps, (
        f"global_reconstruction_error {gfre_val} surpasses threshold for zero {eps}"
    )


def test_global_reconstruction_error_small_to_large(
    rm_features_small, rm_features_large, eps
):
    # tests that the GRE of a small set of features onto a larger set of features
    # returns within a threshold of zero
    gfre_val = global_reconstruction_error(rm_features_small, rm_features_large)
    assert abs(gfre_val) < eps, (
        f"global_reconstruction_error {gfre_val} surpasses threshold for zero {eps}"
    )


def test_global_reconstruction_error_large_to_small(
    rm_features_large, rm_features_small, eps
):
    # tests that the GRE of a large set of features onto a smaller set of features
    # returns within a threshold of zero
    gfre_val = global_reconstruction_error(rm_features_large, rm_features_small)
    assert abs(gfre_val) < eps, (
        f"global_reconstruction_error {gfre_val} surpasses threshold for zero {eps}"
    )


def test_global_reconstruction_distortion_identity(rm_features_large, eps):
    # tests that the GRD of a set of features onto itself returns within a threshold
    # of zero
    gfrd_val = global_reconstruction_distortion(rm_features_large, rm_features_large)
    assert abs(gfrd_val) < eps, (
        f"global_reconstruction_error {gfrd_val} surpasses threshold for zero {eps}"
    )


def test_global_reconstruction_distortion_small_to_large(
    rm_features_small, rm_features_large
):
    # tests that the GRD of a small set of features onto a larger set of features
    # returns within a threshold of zero
    # should just run
    global_reconstruction_error(rm_features_small, rm_features_large)


def test_global_reconstruction_distortion_large_to_small(
    rm_features_large, rm_features_small
):
    # tests that the GRD of a large set of features onto a smaller set of features
    # returns within a threshold of zero
    # should just run
    global_reconstruction_error(rm_features_large, rm_features_small)


def test_global_reconstruction_distortion_small_to_rotated_small(
    rm_features_small, rm_features_rotated_small, eps
):
    # tests that the GRD of a small set of features onto a rotation of itself
    # returns within a threshold of zero
    gfrd_val = global_reconstruction_distortion(
        rm_features_small, rm_features_rotated_small
    )
    assert abs(gfrd_val) < eps, (
        f"global_reconstruction_error {gfrd_val} surpasses threshold for zero {eps}"
    )


def test_local_reconstruction_error_identity(rm_features_large, n_local_points, eps):
    # tests that the local reconstruction error of a set of features onto itself
    # returns within a threshold of zero
    lfre_val = local_reconstruction_error(
        rm_features_large, rm_features_large, n_local_points
    )
    assert abs(lfre_val) < eps, (
        f"local_reconstruction_error {lfre_val} surpasses threshold for zero {eps}"
    )


def test_local_reconstruction_error_small_to_large(
    rm_features_small, rm_features_large, n_local_points, eps
):
    # tests that the local reconstruction error of a small set of features onto a
    # larger set of features returns within a threshold of zero
    lfre_val = local_reconstruction_error(
        rm_features_small, rm_features_large, n_local_points
    )
    assert abs(lfre_val) < eps, (
        f"local_reconstruction_error {lfre_val} surpasses threshold for zero {eps}"
    )


def test_local_reconstruction_error_large_to_small(
    rm_features_large, rm_features_small, n_local_points, eps
):
    # tests that the local reconstruction error of a larger set of features onto a
    # smaller set of features returns within a threshold of zero
    lfre_val = local_reconstruction_error(
        rm_features_large, rm_features_small, n_local_points
    )
    assert abs(lfre_val) < eps, (
        f"local_reconstruction_error {lfre_val} surpasses threshold for zero {eps}"
    )


def test_local_reconstruction_error_train_idx(rm_features_large, n_local_points):
    # tests that the local reconstruction error works when specifying a manual
    # train idx
    lfre_val = pointwise_local_reconstruction_error(
        rm_features_large,
        rm_features_large,
        n_local_points,
        train_idx=np.arange((len(rm_features_large) // 4)),
    )
    test_size = len(rm_features_large) - (len(rm_features_large) // 4)
    msg = (
        "size of pointwise LFRE "
        f"{len(lfre_val)} differs from expected test set size {test_size}"
    )
    assert len(lfre_val) == test_size, msg


def test_local_reconstruction_error_test_idx(rm_features_large, n_local_points):
    # tests that the local reconstruction error works when specifying a manual
    # train idx
    lfre_val = pointwise_local_reconstruction_error(
        rm_features_large,
        rm_features_large,
        n_local_points,
        test_idx=np.arange((len(rm_features_large) // 4)),
    )
    test_size = len(rm_features_large) // 4
    msg = (
        "size of pointwise LFRE "
        f"{len(lfre_val)} differs from expected test set size {test_size}"
    )
    assert len(lfre_val) == test_size, msg


def test_source_target_len():
    # tests that the source and target features have the same lenght
    X = np.array([[1, 2, 3], [4, 5, 6]])
    Y = np.array([[1, 2, 3]])

    train_idx = [0]
    test_idx = [1]
    scaler = None
    estimator = None

    with pytest.raises(ValueError) as context:
        check_global_reconstruction_measures_input(
            X, Y, train_idx, test_idx, scaler, estimator
        )

    expected_message = "First dimension of X (2) and Y (1) must match"
    assert str(context.value) == expected_message


def test_len_n_local_points():
    # tests that source len is greater or equal than n_local_points in LFRE
    X = np.array([[1, 2, 3], [4, 5, 6]])
    Y = np.array([[1, 1, 1], [2, 2, 2]])

    n_local_points = 10
    train_idx = [0]
    test_idx = [1]
    scaler = None
    estimator = None

    with pytest.raises(ValueError) as context:
        check_local_reconstruction_measures_input(
            X, Y, n_local_points, train_idx, test_idx, scaler, estimator
        )

    expected_message = (
        f"X has {len(X)} samples but n_local_points={n_local_points}. "
        "Must have at least n_local_points samples"
    )
    assert str(context.value) == expected_message


@pytest.fixture(scope="module")
def dt_X():
    """Fixture for distance test X array."""
    return np.array([[1, 2], [3, 4], [5, 6]])


@pytest.fixture(scope="module")
def dt_Y():
    """Fixture for distance test Y array."""
    return np.array([[7, 8], [9, 10]])


@pytest.fixture(scope="module")
def dt_covs():
    """Fixture for distance test covariances."""
    return np.array([[[1, 0.5], [0.5, 1]], [[1, 0.0], [0.0, 1]]])


@pytest.fixture(scope="module")
def dt_cell():
    """Fixture for distance test cell."""
    return [5, 7]


@pytest.fixture(scope="module")
def dt_distances():
    """Fixture for expected euclidean distances."""
    return np.array(
        [
            [8.48528137, 11.3137085],
            [5.65685425, 8.48528137],
            [2.82842712, 5.65685425],
        ]
    )


@pytest.fixture(scope="module")
def dt_periodic_distances():
    """Fixture for expected periodic distances."""
    return np.array(
        [
            [1.41421356, 2.23606798],
            [3.16227766, 1.41421356],
            [2.82842712, 3.16227766],
        ]
    )


@pytest.fixture(scope="module")
def dt_mahalanobis_distances(dt_distances):
    """Fixture for expected mahalanobis distances."""
    return np.array(
        [
            [
                [10.39230485, 13.85640646],
                [6.92820323, 10.39230485],
                [3.46410162, 6.92820323],
            ],
            dt_distances,
        ]
    )


def test_euclidean_distance(dt_X, dt_Y, dt_distances):
    distances = periodic_pairwise_euclidean_distances(dt_X, dt_Y)
    np.testing.assert_allclose(
        distances,
        dt_distances,
        err_msg=f"Calculated distance does not match expected value. "
        f"Calculated: {distances} Expected: {dt_distances}",
    )


def test_periodic_euclidean_distance(dt_X, dt_Y, dt_cell, dt_periodic_distances):
    distances = periodic_pairwise_euclidean_distances(dt_X, dt_Y, cell_length=dt_cell)
    np.testing.assert_allclose(
        distances,
        dt_periodic_distances,
        err_msg=f"Calculated distance does not match expected value. "
        f"Calculated: {distances} Expected: {dt_periodic_distances}",
    )


def test_mahalanobis_distance(dt_X, dt_Y, dt_covs, dt_mahalanobis_distances):
    distances = pairwise_mahalanobis_distances(dt_X, dt_Y, dt_covs)
    np.testing.assert_allclose(
        distances,
        dt_mahalanobis_distances,
        err_msg=f"Calculated distance does not match expected value. "
        f"Calculated: {distances} Expected: {dt_mahalanobis_distances}",
    )
