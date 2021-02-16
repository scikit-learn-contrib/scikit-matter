import unittest
import numpy as np

from sklearn.datasets import load_boston
from sklearn import exceptions
from sklearn.exceptions import NotFittedError

from skcosmo.feature_selection._greedy import GreedySelector


class GreedyTester(GreedySelector):
    def __init__(
        self, n_features_to_select=None, score_thresh_to_select=None, **kwargs
    ):
        super().__init__(
            scoring=self.score,
            n_features_to_select=n_features_to_select,
            score_thresh_to_select=score_thresh_to_select,
            **kwargs,
        )

    def score(self, X, y=None):
        scores = np.linalg.norm(X, axis=0)
        scores[self.selected_idx_] = 0.0
        return scores


class TestGreedy(unittest.TestCase):
    def setUp(self):
        self.X, _ = load_boston(return_X_y=True)

    def test_score_threshold(self):
        selector = GreedyTester(score_thresh_to_select=20, n_features_to_select=12)
        with self.assertWarns(
            Warning, msg="Score threshold of 20 reached. Terminating search at 10 / 12."
        ):
            selector.fit(self.X)

    def test_score_threshold_and_full(self):
        with self.assertRaises(ValueError) as cm:
            _ = GreedyTester(
                score_thresh_to_select=20, full=True, n_features_to_select=12
            )
            self.assertEqual(
                str(cm.message),
                "You cannot specify both `score_thresh_to_select` and `full=True`.",
            )

    def test_bad_warm_start(self):
        selector = GreedyTester()
        with self.assertRaises(ValueError) as cm:
            selector.fit(self.X, warm_start=True)
            self.assertTrue(
                str(cm.message),
                "Cannot fit with warm_start=True without having been previously initialized",
            )

    def test_bad_y(self):
        self.X, self.Y = load_boston(return_X_y=True)
        selector = GreedyTester(n_features_to_select=2)
        with self.assertRaises(ValueError):
            selector.fit(X=self.X, y=self.Y[:2])

    def test_bad_transform(self):
        selector = GreedyTester(n_features_to_select=2)
        with self.assertRaises(exceptions.NotFittedError):
            _ = selector.transform(self.X)

    def test_no_nfeatures(self):
        selector = GreedyTester()
        selector.fit(self.X)
        self.assertEqual(len(selector.selected_idx_), self.X.shape[1] // 2)

    def test_decimal_nfeatures(self):
        selector = GreedyTester(n_features_to_select=0.2)
        selector.fit(self.X)
        self.assertEqual(len(selector.selected_idx_), int(self.X.shape[1] * 0.2))

    def test_bad_nfeatures(self):
        for nf in [1.2, "1", 20]:
            with self.subTest(n_features=nf):
                selector = GreedyTester(n_features_to_select=nf)
                with self.assertRaises(ValueError) as cm:
                    selector.fit(self.X)
                    self.assertEqual(
                        str(cm.message),
                        (
                            "n_features_to_select must be either None, an "
                            "integer in [1, n_features - 1] "
                            "representing the absolute "
                            "number of features, or a float in (0, 1] "
                            "representing a percentage of features to "
                            f"select. Got {nf}"
                        ),
                    )

    def test_not_fitted(self):
        with self.assertRaises(NotFittedError):
            selector = GreedyTester()
            _ = selector._get_support_mask()

    def test_fitted(self):
        selector = GreedyTester()
        selector.fit(self.X)
        _ = selector._get_support_mask()

    def test_no_tqdm(self):
        """
        This test checks that the selector cannot use a progress bar when tqdm
        is not installed
        """
        import sys

        sys.modules["tqdm"] = None

        with self.assertRaises(ImportError) as cm:
            _ = GreedyTester(progress_bar=True)
            self.assertEqual(
                str(cm.exception),
                "tqdm must be installed to use a progress bar."
                "Either install tqdm or re-run with"
                "progress_bar = False",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
