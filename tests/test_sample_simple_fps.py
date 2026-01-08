import numpy as np
import pytest
from sklearn.datasets import load_diabetes as get_dataset
from sklearn.utils.validation import NotFittedError

from skmatter.sample_selection import FPS


@pytest.fixture
def X_and_idx():
    X, _ = get_dataset(return_X_y=True)
    idx = [0, 123, 441, 187, 117, 276, 261, 281, 251, 193]
    return X, idx


def test_restart(X_and_idx):
    """Checks that the model can be restarted with a new number of samples and
    `warm_start`.
    """
    X, idx = X_and_idx
    selector = FPS(n_to_select=1, initialize=idx[0])
    selector.fit(X)

    for i in range(2, len(idx)):
        selector.n_to_select = i
        selector.fit(X, warm_start=True)
        assert selector.selected_idx_[i - 1] == idx[i - 1]


def test_initialize(X_and_idx):
    """Checks that the model can be initialized in all applicable manners and throws
    an error otherwise.
    """
    X, idx = X_and_idx

    for initialize in [idx[0], "random"]:
        selector = FPS(n_to_select=1, initialize=initialize)
        selector.fit(X)

    initialize = idx[:4]
    selector = FPS(n_to_select=len(idx) - 1, initialize=initialize)
    selector.fit(X)
    for i in range(4):
        assert selector.selected_idx_[i] == idx[i]

    initialize = np.array(idx[:4])
    selector = FPS(n_to_select=len(idx) - 1, initialize=initialize)
    selector.fit(X)
    for i in range(4):
        assert selector.selected_idx_[i] == idx[i]

    initialize = np.array([1, 5, 3, 0.25])
    with pytest.raises(ValueError, match="Invalid value of the initialize parameter"):
        selector = FPS(n_to_select=len(idx) - 1, initialize=initialize)
        selector.fit(X)

    initialize = np.array([[1, 5, 3], [2, 4, 6]])
    with pytest.raises(ValueError, match="Invalid value of the initialize parameter"):
        selector = FPS(n_to_select=len(idx) - 1, initialize=initialize)
        selector.fit(X)

    with pytest.raises(ValueError, match="Invalid value of the initialize parameter"):
        selector = FPS(n_to_select=1, initialize="bad")
        selector.fit(X)


def test_get_distances(X_and_idx):
    """Checks that the hausdorff distances are returnable after fitting."""
    X, _ = X_and_idx
    selector = FPS(n_to_select=1)
    selector.fit(X)
    selector.get_select_distance()

    with pytest.raises(NotFittedError, match="instance is not fitted"):
        selector = FPS(n_to_select=1)
        selector.get_select_distance()


def test_threshold(X_and_idx):
    X, idx = X_and_idx
    selector = FPS(
        n_to_select=10,
        score_threshold=5e-2,
        score_threshold_type="absolute",
    )
    selector.fit(X)
    assert len(selector.selected_idx_) == 6
    assert selector.selected_idx_.tolist() == idx[:6]

    selector = FPS(
        n_to_select=10,
        score_threshold=0.4,
        score_threshold_type="relative",
    )
    selector.fit(X)
    assert len(selector.selected_idx_) == 5
    assert selector.selected_idx_.tolist() == idx[:5]


def test_unique_selected_idx_zero_score():
    """
    Tests that the selected idxs are unique, which may not be the
    case when the score is numerically zero.
    """
    np.random.seed(0)
    n_samples = 10
    n_features = 15
    X = np.random.rand(n_samples, n_features)
    X[1] = X[0]
    X[2] = X[0]
    X[3] = X[0]
    selector_problem = FPS(n_to_select=len(X)).fit(X)
    assert len(selector_problem.selected_idx_) == len(
        set(selector_problem.selected_idx_)
    )
