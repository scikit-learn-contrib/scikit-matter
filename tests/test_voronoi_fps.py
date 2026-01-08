import numpy as np
import pytest
from sklearn.datasets import load_diabetes as get_dataset
from sklearn.exceptions import NotFittedError

from skmatter.sample_selection import FPS, VoronoiFPS


@pytest.fixture
def X():
    """Feature matrix for VoronoiFPS tests."""
    X, _ = get_dataset(return_X_y=True)
    return X


@pytest.fixture
def idx():
    """Expected indices for VoronoiFPS tests."""
    return [0, 123, 441, 187, 117, 276, 261, 281, 251, 193]


def test_restart(X, idx):
    """Checks that the model can be restarted with a new number of
    features and `warm_start`
    """
    selector = VoronoiFPS(n_to_select=1, initialize=idx[0])
    selector.fit(X)

    for i in range(2, len(idx)):
        selector.n_to_select = i
        selector.fit(X, warm_start=True)
        assert selector.selected_idx_[i - 1] == idx[i - 1]


def test_initialize_with_idx(X, idx):
    """Test initialization with idx fixture value"""
    selector = VoronoiFPS(n_to_select=1, initialize=idx[0])
    selector.fit(X)


def test_initialize_with_random(X):
    """Test initialization with 'random' string"""
    selector = VoronoiFPS(n_to_select=1, initialize="random")
    selector.fit(X)


def test_initialize_invalid(X):
    """Test that invalid initialization raises an error"""
    with pytest.raises(ValueError) as cm:
        selector = VoronoiFPS(n_to_select=1, initialize="bad")
        selector.fit(X)
    assert str(cm.value) == "Invalid value of the initialize parameter"


def test_switching_point_auto(X):
    """Check work of the switching point calculator into the
    _init_greedy_search function
    """
    selector = VoronoiFPS(n_to_select=1)
    selector.fit(X)
    assert 1 > selector.full_fraction


def test_switching_point_manual(X):
    """Test manual full_fraction setting"""
    selector = VoronoiFPS(n_to_select=1, full_fraction=0.5)
    selector.fit(X)
    assert selector.full_fraction == 0.5


def test_switching_point_bad_ntrial(X):
    """Test bad n_trial_calculation"""
    with pytest.raises(ValueError) as cm:
        selector = VoronoiFPS(n_to_select=1, n_trial_calculation=0)
        selector.fit(X)
    assert str(cm.value) == "Number of trial calculation should be more or equal to 1"


def test_switching_point_float_ntrial(X):
    """Test float n_trial_calculation"""
    with pytest.raises(TypeError) as cm:
        selector = VoronoiFPS(n_to_select=1, n_trial_calculation=0.3)
        selector.fit(X)
    assert str(cm.value) == "Number of trial calculation should be integer"


def test_switching_point_large_ff(X):
    """Test large full_fraction"""
    selector = VoronoiFPS(n_to_select=1, full_fraction=1.1)
    with pytest.raises(ValueError) as cm:
        selector.fit(X)
    assert str(cm.value) == (
        "Switching point should be real and more than 0 and less than 1. "
        f"Received {selector.full_fraction}"
    )


def test_switching_point_string_ff(X):
    """Test string full_fraction"""
    selector = VoronoiFPS(n_to_select=1, full_fraction="STRING")
    with pytest.raises(ValueError) as cm:
        selector.fit(X)
    assert str(cm.value) == (
        "Switching point should be real and more than 0 and less than 1. "
        f"Received {selector.full_fraction}"
    )


def test_get_distances(X):
    """Checks that the hausdorff distances are returnable after fitting"""
    selector = VoronoiFPS(n_to_select=1)
    selector.fit(X)
    _ = selector.get_select_distance()


def test_get_distances_not_fitted(X):
    """Test get_select_distance without fitting"""
    with pytest.raises(NotFittedError):
        selector = VoronoiFPS(n_to_select=1)
        _ = selector.get_select_distance()


def test_comparison(X):
    """Checks that the voronoi FPS strictly computes less distances
    than its normal FPS counterpart.
    """
    vselector = VoronoiFPS(n_to_select=X.shape[0] - 1)
    vselector.fit(X)

    selector = FPS(n_to_select=X.shape[0] - 1)
    selector.fit(X)

    assert np.allclose(vselector.selected_idx_, selector.selected_idx_)


def test_nothing_updated_points():
    """Checks that in the case where we have no points to update, the code
    still works fine
    """
    X = np.array([[1, 1], [4, 4], [10, 10], [100, 100]])
    selector = VoronoiFPS(n_to_select=3, initialize=0)
    try:
        selector.fit(X)
        f = 1
    except Exception:
        f = 0
    assert f == 1

    assert (
        len(np.where(selector.vlocation_of_idx == (selector.n_selected_ - 2))[0]) == 1
    )


def test_calculate_dSL(X):
    selector = VoronoiFPS(n_to_select=3)
    selector.fit(X)

    active_points = np.where(
        selector.dSL_[selector.vlocation_of_idx] < selector.hausdorff_
    )[0]

    ap = selector._get_active(X, selector.selected_idx_[-1])

    assert np.allclose(active_points, ap)

    selector = VoronoiFPS(n_to_select=1)

    ap = selector._get_active(X, 0)

    assert np.allclose(np.arange(X.shape[0]), ap)


def test_score(X, idx):
    """Check that function score return hausdorff distance"""
    selector = VoronoiFPS(n_to_select=3, initialize=0)
    selector.fit(X)

    assert np.allclose(
        selector.hausdorff_,
        selector.score(X, selector.selected_idx_[-1]),
    )


def test_unique_selected_idx_zero_score():
    """
    Tests that the selected idxs are unique, which may not be the
    case when the score is numerically zero
    """
    np.random.seed(0)
    n_samples = 10
    n_features = 15
    X = np.random.rand(n_samples, n_features)
    X[1] = X[0]
    X[2] = X[0]
    selector_problem = VoronoiFPS(n_to_select=n_samples, initialize=3).fit(X)
    assert len(selector_problem.selected_idx_) == len(
        set(selector_problem.selected_idx_)
    )
