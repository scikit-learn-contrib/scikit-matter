import unittest

from sklearn.datasets import load_boston

from skcosmo.sample_selection import PCovFPS


class TestPCovFPS(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_boston(return_X_y=True)
        self.idx = [39, 410, 492, 102, 54, 413, 34, 346, 126, 134, 433, 380]

    def test_restart(self):
        """
        This test checks that the model can be restarted with a new number of
        samples and `warm_start`
        """

        selector = PCovFPS(n_to_select=1, initialize=self.idx[0])
        selector.fit(self.X, y=self.y)

        for i in range(2, len(self.idx)):
            selector.n_to_select = i
            selector.fit(self.X, y=self.y, warm_start=True)
            self.assertEqual(selector.selected_idx_[i - 1], self.idx[i - 1])

    def test_no_mixing_1(self):
        """
        This test checks that the model throws an error when mixing = 1.0
        """

        with self.assertRaises(ValueError) as cm:
            _ = PCovFPS(n_to_select=1, mixing=1.0)
            self.assertEquals(
                str(cm.message),
                "Mixing = 1.0 corresponds to traditional FPS."
                "Please use the FPS class.",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
