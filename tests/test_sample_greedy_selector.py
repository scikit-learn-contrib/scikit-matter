import unittest
import numpy as np

from sklearn.datasets import load_boston
from sklearn import exceptions
from sklearn.exceptions import NotFittedError

from skcosmo.sample_selection._greedy import GreedySelector


class GreedyTester(GreedySelector):
    def __init__(self, n_samples_to_select=None, score_thresh_to_select=None, **kwargs):
        super().__init__(
            scoring=self.score,
            n_samples_to_select=n_samples_to_select,
            score_thresh_to_select=score_thresh_to_select,
            **kwargs,
        )

    def score(self, X, y=None):
        scores = np.linalg.norm(X, axis=1)
        scores[self.selected_idx_] = 0.0
        return scores


class TestGreedy(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_boston(return_X_y=True)

    def test_restart(self):
        """ tests that restart does not crash """
        selector = GreedyTester(n_samples_to_select=2)
        selector.fit(self.X, self.y)
        self.assertEqual(len(selector.selected_idx_), 2)

        selector.n_samples_to_select = 5
        selector.fit(self.X, self.y, warm_start=True)
        self.assertEqual(len(selector.selected_idx_), 5)

    def test_restart_with_y(self):
        """ tests that restart with y does not crash """
        selector = GreedyTester(n_samples_to_select=2)
        selector.fit(self.X, None)
        self.assertEqual(len(selector.selected_idx_), 2)

        selector.n_samples_to_select = 5
        selector.fit(self.X, None, warm_start=True)
        self.assertEqual(len(selector.selected_idx_), 5)

    def test_score_threshold(self):
        selector = GreedyTester(score_thresh_to_select=600, n_samples_to_select=400)
        with self.assertWarns(
            Warning,
            msg="Score threshold of 600 reached. Terminating search at 139 / 400.",
        ):
            selector.fit(self.X)

    def test_score_threshold_with_y(self):
        selector = GreedyTester(score_thresh_to_select=600, n_samples_to_select=400)
        with self.assertWarns(
            Warning,
            msg="Score threshold of 600 reached. Terminating search at 139 / 400.",
        ):
            selector.fit(self.X, self.y)

    def test_score_threshold_and_full(self):
        with self.assertRaises(ValueError) as cm:
            _ = GreedyTester(
                score_thresh_to_select=20, full=True, n_samples_to_select=12
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

    def test_good_y(self):
        selector = GreedyTester(n_samples_to_select=2)
        selector.fit(X=self.X, y=self.y)
        y_s = selector.transform(self.y)
        self.assertEqual(selector.y_selected_.shape[0], 2)
        self.assertEqual(y_s.shape[0], 2)

    def test_bad_y(self):
        selector = GreedyTester(n_samples_to_select=2)
        with self.assertRaises(ValueError):
            selector.fit(X=self.X, y=self.y[:2])

    def test_bad_transform(self):
        selector = GreedyTester(n_samples_to_select=2)
        with self.assertRaises(exceptions.NotFittedError):
            _ = selector.transform(self.X)

    def test_no_nsamples(self):
        selector = GreedyTester()
        selector.fit(self.X)
        self.assertEqual(len(selector.selected_idx_), self.X.shape[0] // 2)

    def test_decimal_nsamples(self):
        selector = GreedyTester(n_samples_to_select=0.2)
        selector.fit(self.X)
        self.assertEqual(len(selector.selected_idx_), int(self.X.shape[0] * 0.2))

    def test_bad_nsamples(self):
        for nf in [1.2, "1", 2000]:
            with self.subTest(n_samples=nf):
                selector = GreedyTester(n_samples_to_select=nf)
                with self.assertRaises(ValueError) as cm:
                    selector.fit(self.X)
                    self.assertEqual(
                        str(cm.message),
                        (
                            "n_samples_to_select must be either None, an "
                            "integer in [1, n_samples - 1] "
                            "representing the absolute "
                            "number of samples, or a float in (0, 1] "
                            "representing a percentage of samples to "
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
