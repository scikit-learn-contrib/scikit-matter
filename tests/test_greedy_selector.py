import numpy as np
import pytest
from sklearn.datasets import load_diabetes as get_dataset
from sklearn.exceptions import NotFittedError

from skmatter._selection import GreedySelector


class GreedyTester(GreedySelector):
    def __init__(
        self, n_to_select=None, score_threshold=None, selection_type="feature", **kwargs
    ):
        super().__init__(
            selection_type=selection_type,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            **kwargs,
        )

    def score(self, X, y=None):
        scores = np.linalg.norm(X, axis=0)
        scores[self.selected_idx_] = 0.0
        return scores


@pytest.fixture
def X():
    X, _ = get_dataset(return_X_y=True)
    return X


def test_bad_type(X):
    with pytest.raises(
        ValueError, match="Only feature and sample selection supported."
    ):
        _ = GreedyTester(selection_type="bad").fit(X)


def test_score_threshold(X):
    selector = GreedyTester(score_threshold=200, n_to_select=7)
    with pytest.warns(
        Warning,
        match=r"Score threshold of 200 reached\..*Terminating search at \d+ / 7\.",
    ):
        selector.fit(X)


def test_score_threshold_and_full(X):
    with pytest.raises(ValueError) as cm:
        _ = GreedyTester(score_threshold=20, full=True, n_to_select=12).fit(X)
    assert str(cm.value) == "You cannot specify both `score_threshold` and `full=True`."


def test_bad_score_threshold_type(X):
    with pytest.raises(ValueError) as cm:
        _ = GreedyTester(score_threshold_type="bad").fit(X)
    assert (
        str(cm.value)
        == "invalid score_threshold_type, expected one of 'relative' or 'absolute'"
    )


def test_bad_warm_start(X):
    selector = GreedyTester()
    with pytest.raises(
        ValueError,
        match=(
            "Cannot fit with warm_start=True without having been previously initialized"
        ),
    ):
        selector.fit(X, warm_start=True)


def test_bad_y(X):
    _, Y = get_dataset(return_X_y=True)
    Y = Y[:2]
    selector = GreedyTester(n_to_select=2)
    with pytest.raises(ValueError):
        selector.fit(X=X, y=Y)


def test_bad_transform(X):
    selector = GreedyTester(n_to_select=2)
    selector.fit(X)
    with pytest.raises(ValueError) as cm:
        _ = selector.transform(X[:, :3])
    assert (
        str(cm.value)
        == "X has 3 features, but GreedyTester is expecting 10 features as input."
    )


def test_no_nfeatures(X):
    selector = GreedyTester()
    selector.fit(X)
    assert len(selector.selected_idx_) == X.shape[1] // 2


def test_decimal_nfeatures(X):
    selector = GreedyTester(n_to_select=0.2)
    selector.fit(X)
    assert len(selector.selected_idx_) == int(X.shape[1] * 0.2)


@pytest.mark.parametrize("nf", [1.2, "1", 20])
def test_bad_nfeatures(X, nf):
    selector = GreedyTester(n_to_select=nf)
    with pytest.raises(ValueError) as cm:
        selector.fit(X)
    assert str(cm.value) == (
        "n_to_select must be either None, an integer in "
        "[1, n_features] representing the absolute number "
        "of features, or a float in (0, 1] representing a "
        f"percentage of features to select. Got {nf} "
        f"features and an input with {X.shape[1]} feature."
    )


def test_not_fitted():
    with pytest.raises(NotFittedError):
        selector = GreedyTester()
        _ = selector._get_support_mask()


def test_fitted(X):
    selector = GreedyTester()
    selector.fit(X)
    _ = selector._get_support_mask()

    Xr = selector.transform(X)
    assert Xr.shape[1] == X.shape[1] // 2


def test_size_input():
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    selector_sample = GreedyTester(selection_type="sample")
    selector_feature = GreedyTester(selection_type="feature")
    with pytest.raises(ValueError) as cm:
        selector_feature.fit(X)
    assert str(cm.value) == (
        f"Found array with 1 feature(s) (shape={X.shape}) while a minimum of 2 is "
        "required by GreedyTester."
    )

    X = X.reshape(1, -1)

    with pytest.raises(ValueError) as cm:
        selector_sample.fit(X)
    assert str(cm.value) == (
        f"Found array with 1 sample(s) (shape={X.shape}) while a minimum of 2 is "
        "required by GreedyTester."
    )
