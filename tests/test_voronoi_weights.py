import numpy as np
import pytest

from skmatter.sample_selection import voronoi_weights
from skmatter.sample_selection import _voronoi_weights as vw_module


@pytest.fixture
def simple_case():
    # three points sit next to landmark 0; landmark 1 is far away and wins
    # no points, so its Voronoi cell is empty
    X_full = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    X_landmarks = np.array([[0.0, 0.0], [10.0, 10.0]])
    return X_full, X_landmarks


def test_shape_and_order(simple_case):
    X_full, X_landmarks = simple_case
    weights = voronoi_weights(X_full, X_landmarks)
    assert weights.shape == (X_landmarks.shape[0],)


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


def test_sample_weights_are_summed(simple_case):
    X_full, X_landmarks = simple_case
    weights = voronoi_weights(
        X_full, X_landmarks, sample_weights=[1.0, 2.0, 3.0], normalize=False
    )
    np.testing.assert_array_equal(weights, [6.0, 0.0])


def test_sample_weights_length_mismatch(simple_case):
    X_full, X_landmarks = simple_case
    with pytest.raises(ValueError, match="sample_weights has length"):
        voronoi_weights(X_full, X_landmarks, sample_weights=[1.0, 2.0])


def test_chunking_matches_single_pass(monkeypatch):
    rng = np.random.default_rng(0)
    X_full = rng.standard_normal((500, 4))
    X_landmarks = X_full[:15]

    full = voronoi_weights(X_full, X_landmarks)
    monkeypatch.setattr(vw_module, "_CHUNK_ELEMENTS", 30)  # ~2 rows per chunk
    chunked = voronoi_weights(X_full, X_landmarks)

    np.testing.assert_array_equal(full, chunked)


def test_zero_total_does_not_divide_by_zero(simple_case):
    # all-zero sample weights -> zero total, normalize must not blow up
    X_full, X_landmarks = simple_case
    weights = voronoi_weights(X_full, X_landmarks, sample_weights=[0.0, 0.0, 0.0])
    np.testing.assert_array_equal(weights, [0.0, 0.0])
