import unittest

from sklearn.datasets import load_iris
import skcosmo.model_selection
import sklearn.model_selection


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = load_iris().data[:10]
        cls.seed = 0x5F3759DF

    def test_train_test_splits(self):
        # see if train_test_split of skcosmo agrees with the one of sklearn
        sklearn_outputs = sklearn.model_selection.train_test_split(
            self.X, random_state=self.seed
        )
        skcosmo_outputs = skcosmo.model_selection.train_test_split(
            self.X, random_state=self.seed
        )
        for i in range(len(skcosmo_outputs)):
            self.assertTrue((sklearn_outputs[i] == skcosmo_outputs[i]).all())

    def test_train_test_splits_train_test_overlap(self):
        # tests that a test/train split which necessitates overlap returns the right number of points in each set
        X_train, X_test = skcosmo.model_selection.train_test_split(
            self.X,
            train_size=0.8,
            test_size=0.8,
            train_test_overlap=True,
            random_state=self.seed,
        )
        self.assertTrue(len(X_train) == len(X_test) == int(0.8 * self.X.shape[0]))

    def test_train_test_splits_train_test_overlap_full_test_set(self):
        # tests that the entire dataset can be used as the testing set
        X_train, X_test = skcosmo.model_selection.train_test_split(
            self.X,
            train_size=0.8,
            test_size=1.0,
            train_test_overlap=True,
            random_state=self.seed,
        )
        self.assertTrue((self.X == X_test).all())

    def test_train_test_splits_train_test_overlap_full_train_test_set(self):
        # tests that the full dataset can be "split" to both train and test set
        X_train, X_test = skcosmo.model_selection.train_test_split(
            self.X,
            train_size=1.0,
            test_size=1.0,
            train_test_overlap=True,
            random_state=self.seed,
        )
        self.assertTrue((X_train == X_test).all())


if __name__ == "__main__":
    unittest.main()
