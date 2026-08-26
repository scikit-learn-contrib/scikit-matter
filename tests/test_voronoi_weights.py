import numpy as np
import pytest

from skmatter.sample_selection import voronoi_weights


@pytest.fixture
def simple_case():
    # three points sit next to landmark 0, one next to the far landmark 1
    # no points, so its Voronoi cell is empty
    X_full = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    X_landmarks = np.array([[0.0, 0.0], [10.0, 10.0]])
    return X_full, X_landmarks


def test_weight_per_landmark_in_input_order(simple_case):
    X_full, X_landmarks = simple_case
    weights = voronoi_weights(X_full, X_landmarks)

    assert weights.shape == (X_landmarks.shape[0],)
    assert weights[0] > weights[1]


def test_normalized_sums_to_one(simple_case):
    X_full, X_landmarks = simple_case
    weights = voronoi_weights(X_full, X_landmarks)
    assert np.isclose(weights.sum(), 1.0)


def test_raw_counts(simple_case):
    X_full, X_landmarks = simple_case
    weights = voronoi_weights(X_full, X_landmarks, normalize=False)
    np.testing.assert_array_equal(weights, [3.0, 0.0])


def test_empty_cell_gets_zero(simple_case):
    X_full, X_landmarks = simple_case
    weights = voronoi_weights(X_full, X_landmarks)
    assert weights[1] == 0.0


def test_power_interpolates_toward_uniform():
    X_full = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [10.0, 10.0]])
    X_landmarks = np.array([[0.0, 0.0], [10.0, 10.0]])

    density = voronoi_weights(X_full, X_landmarks)
    damped = voronoi_weights(X_full, X_landmarks, power=0.5)
    uniform = voronoi_weights(X_full, X_landmarks, power=0.0)

    np.testing.assert_allclose(density, [0.8, 0.2])
    np.testing.assert_allclose(damped, [2.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(uniform, [0.5, 0.5])


def test_power_negative_raises(simple_case):
    X_full, X_landmarks = simple_case
    with pytest.raises(ValueError, match="power must be non-negative"):
        voronoi_weights(X_full, X_landmarks, power=-1.0)


def test_empty_input_raises(simple_case):
    _, X_landmarks = simple_case
    with pytest.raises(ValueError, match="0 sample"):
        voronoi_weights(np.empty((0, 2)), X_landmarks)
