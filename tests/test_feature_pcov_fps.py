import pytest
from sklearn.datasets import load_diabetes as get_dataset

from skmatter.feature_selection import PCovFPS


@pytest.fixture
def X_y_idx():
    X, y = get_dataset(return_X_y=True)
    idx = [0, 2, 6, 7, 1, 3, 4]
    return X, y, idx


def test_restart(X_y_idx):
    """Check that the model can be restarted with a new number of features and
    `warm_start`.
    """
    X, y, idx = X_y_idx
    selector = PCovFPS(n_to_select=1, initialize=idx[0])
    selector.fit(X, y=y)

    for i in range(2, len(idx)):
        selector.n_to_select = i
        selector.fit(X, y=y, warm_start=True)
        assert selector.selected_idx_[i - 1] == idx[i - 1]


def test_no_mixing_1(X_y_idx):
    """Check that the model throws an error when mixing = 1.0."""
    X, y, _ = X_y_idx
    selector = PCovFPS(n_to_select=1, mixing=1.0)
    with pytest.raises(ValueError) as cm:
        selector.fit(X, y=y)
    assert (
        str(cm.value)
        == "Mixing = 1.0 corresponds to traditional FPS. Please use the FPS class."
    )
