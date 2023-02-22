import unittest

from sklearn.datasets import load_diabetes as get_dataset
from sklearn.utils.validation import NotFittedError

from skcosmo.sample_selection import FPS


class TestFPS(unittest.TestCase):
    def setUp(self):
        self.X, _ = get_dataset(return_X_y=True)
        self.idx = [0, 123, 441, 187, 117, 276, 261, 281, 251, 193]

    def test_restart(self):
        """
        This test checks that the model can be restarted with a new number of
        samples and `warm_start`
        """

        selector = FPS(n_to_select=1, initialize=self.idx[0])
        selector.fit(self.X)

        for i in range(2, len(self.idx)):
            selector.n_to_select = i
            selector.fit(self.X, warm_start=True)
            self.assertEqual(selector.selected_idx_[i - 1], self.idx[i - 1])

    def test_initialize(self):
        """
        This test checks that the model can be initialized in all applicable manners
        and throws an error otherwise
        """

        for initialize in [self.idx[0], "random"]:
            with self.subTest(initialize=initialize):
                selector = FPS(n_to_select=1, initialize=initialize)
                selector.fit(self.X)

        initialize = self.idx[:4]
        with self.subTest(initialize=initialize):
            selector = FPS(n_to_select=len(self.idx) - 1, initialize=initialize)
            selector.fit(self.X)
            for i in range(4):
                self.assertEqual(selector.selected_idx_[i], self.idx[i])

        with self.assertRaises(ValueError) as cm:
            selector = FPS(n_to_select=1, initialize="bad")
            selector.fit(self.X)
        self.assertEquals(
            str(cm.exception), "Invalid value of the initialize parameter"
        )

    def test_get_distances(self):
        """
        This test checks that the haussdorf distances are returnable after fitting
        """
        selector = FPS(n_to_select=1)
        selector.fit(self.X)
        _ = selector.get_select_distance()

        with self.assertRaises(NotFittedError):
            selector = FPS(n_to_select=1)
            _ = selector.get_select_distance()

    def test_threshold(self):
        selector = FPS(
            n_to_select=10,
            score_threshold=5e-2,
            score_threshold_type="absolute",
        )
        selector.fit(self.X)
        self.assertEqual(len(selector.selected_idx_), 6)
        self.assertEqual(selector.selected_idx_.tolist(), self.idx[:6])

        selector = FPS(
            n_to_select=10,
            score_threshold=0.4,
            score_threshold_type="relative",
        )
        selector.fit(self.X)
        self.assertEqual(len(selector.selected_idx_), 5)
        self.assertEqual(selector.selected_idx_.tolist(), self.idx[:5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
