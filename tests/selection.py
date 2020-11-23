import unittest
from sklearn.datasets import load_boston
import numpy as np

from skcosmo.selection import FeatureCUR, FeatureFPS, SampleCUR, SampleFPS

EPSILON = 1e-8


class PCovSelectionTest:
    def test_no_Y(self):
        """
        This test checks that if mixing < 1, the selection method raises an
        exception when no Y is supplied
        """
        X, Y = load_boston(return_X_y=True)

        with self.assertRaises(Exception) as cm:
            self.model(X=X, mixing=0.5)
            self.assertEqual(
                str(cm.exception), r"For $\alpha < 1$, $Y$ must be supplied."
            )

    def test_no_X(self):
        """
        This test checks that X is supplied
        """
        X, Y = load_boston(return_X_y=True)

        with self.assertRaises(Exception):
            self.model(mixing=0.5, Y=Y)

    def test_bad_Y(self):
        """
        This test checks that the model errors when Y contains a different
        number of samples from X
        """
        X, Y = load_boston(return_X_y=True)

        with self.assertRaises(Exception):
            self.model(X=X, mixing=0.5, Y=Y[:20])

    def test_bad_X(self):
        """
        This test checks that the model errors when X contains a different
        number of samples from Y
        """
        X, Y = load_boston(return_X_y=True)

        with self.assertRaises(Exception):
            self.model(X=X[:10], mixing=0.5, Y=Y)


class FeatureFPSTest(unittest.TestCase, PCovSelectionTest):
    def setup(self):
        self.model = FeatureFPS


class FeatureCURTest(unittest.TestCase, PCovSelectionTest):
    def setup(self):
        self.model = FeatureCUR

    def test_orthogonalize(self):
        """
        This test checks that the orthogonalization that occurs after each
        selection leads to a null feature column
        """

        self.setup()
        X, Y = load_boston(return_X_y=True)

        cur = self.model(mixing=0.5, X=X, Y=Y, iterative=True)

        for i in range(X.shape[1]):
            with self.subTest(i=i):
                cur.select(1)
                self.assertLessEqual(
                    np.linalg.norm(cur.A_current[:, cur.idx[-1]]), EPSILON
                )

    def test_others_orthogonal(self):
        """
        This test checks that the orthogonalization that occurs after each
        selection leads to orthogonal feature columns
        """
        self.setup()
        X, Y = load_boston(return_X_y=True)

        cur = self.model(mixing=0.5, X=X, Y=Y, iterative=True)

        for i in range(X.shape[1]):
            with self.subTest(i=i):
                cur.select(1)
                for j in range(X.shape[1]):
                    self.assertLessEqual(
                        np.dot(cur.A_current[:, cur.idx[-1]], cur.A_current[:, j]),
                        EPSILON,
                    )


class SampleFPSTest(unittest.TestCase, PCovSelectionTest):
    def setup(self):
        self.model = SampleFPS


class SampleCURTest(unittest.TestCase, PCovSelectionTest):
    def setup(self):
        self.model = SampleCUR

    def test_orthogonalize(self):
        """
        This test checks that the orthogonalization that occurs after each
        selection leads to a null sample row
        """
        self.setup()
        X, Y = load_boston(return_X_y=True)

        cur = self.model(mixing=0.5, X=X, Y=Y, iterative=True)

        for i in range(X.shape[0]):
            with self.subTest(i=i):
                cur.select(1)
                self.assertLessEqual(
                    np.linalg.norm(cur.A_current[cur.idx[-1]]), EPSILON
                )

    def test_others_orthogonal(self):
        """
        This test checks that the orthogonalization that occurs after each
        selection leads to orthogonal sample rows
        """
        self.setup()
        X, Y = load_boston(return_X_y=True)

        cur = self.model(mixing=0.5, X=X, Y=Y, iterative=True)

        for i in range(X.shape[0]):
            with self.subTest(i=i):
                cur.select(1)
                for j in range(X.shape[0]):
                    self.assertLessEqual(
                        np.dot(cur.A_current[cur.idx[-1]], cur.A_current[j]), EPSILON
                    )


if __name__ == "__main__":
    unittest.main()
