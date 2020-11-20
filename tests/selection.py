import unittest
import skcosmo
from sklearn.datasets import load_boston


class PCovSelectionTest:
    def test_no_Y(self):
        X, Y = load_boston(return_X_y=True)

        with self.assertRaises(Exception) as cm:
            self.model(X=X, mixing=0.5)
            self.assertEqual(
                str(cm.exception), r"For $\alpha < 1$, $Y$ must be supplied."
            )

    def test_no_X(self):
        X, Y = load_boston(return_X_y=True)

        with self.assertRaises(Exception):
            self.model(mixing=0.5, Y=Y)

    def test_bad_Y(self):
        X, Y = load_boston(return_X_y=True)

        with self.assertRaises(Exception):
            self.model(X=X, mixing=0.5, Y=Y[:20])

    def test_bad_X(self):
        X, Y = load_boston(return_X_y=True)

        with self.assertRaises(Exception):
            self.model(X=X[:10], mixing=0.5, Y=Y)


class FeatureFPSTest(unittest.TestCase, PCovSelectionTest):
    def setup(self):
        self.model = skcosmo.selection.FeatureFPS


class FeatureCURTest(unittest.TestCase, PCovSelectionTest):
    def setup(self):
        self.model = skcosmo.selection.FeatureCUR


class SampleFPSTest(unittest.TestCase, PCovSelectionTest):
    def setup(self):
        self.model = skcosmo.selection.SampleFPS


class SampleCURTest(unittest.TestCase, PCovSelectionTest):
    def setup(self):
        self.model = skcosmo.selection.SampleCUR


if __name__ == "__main__":
    unittest.main()
