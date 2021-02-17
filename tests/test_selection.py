import unittest
from sklearn.datasets import load_boston
import numpy as np

from skcosmo.sample_selection import SampleCUR, SampleFPS

EPSILON = 1e-6


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

    def test_quick_run(self):
        """
        This test checks that the model can select 2 indices
        """
        X, Y = load_boston(return_X_y=True)

        model = self.model(X=X, mixing=0.5, Y=Y)
        model.select(n=2)
        self.assertEqual(len(model.idx), 2)

    def test_negative_select(self):
        """
        This test checks that the model errors when asked to select negative indices
        """
        X, Y = load_boston(return_X_y=True)

        with self.assertRaises(ValueError):
            model = self.model(X=X, mixing=0.5, Y=Y)
            model.select(n=-1)

    def test_no_tqdm(self):
        """
        This test checks that the model cannot use a progress bar when tqdm
        is not installed
        """
        import sys

        sys.modules["tqdm"] = None

        X, Y = load_boston(return_X_y=True)

        with self.assertRaises(ImportError) as cm:
            _ = self.model(X=X, mixing=0.5, Y=Y, progress_bar=True)
            self.assertEqual(
                str(cm.exception),
                "tqdm must be installed to use a progress bar."
                "Either install tqdm or re-run with"
                "progress_bar = False",
            )


class SampleFPSTest(unittest.TestCase, PCovSelectionTest):
    def setUp(self):
        self.model = SampleFPS

    def test_supplied_indices(self):
        """
        This test checks FPS will accept pre-defined indices.
        """
        X, Y = load_boston(return_X_y=True)
        model = self.model(X=X, mixing=0.5, Y=Y, idxs=[488])
        model.select(n=4)


class SampleCURTest(unittest.TestCase, PCovSelectionTest):
    def setUp(self):
        self.model = SampleCUR

    def test_orthogonalize(self):
        """
        This test checks that the orthogonalization that occurs after each
        selection leads to a null sample row
        """
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
    unittest.main(verbosity=2)
