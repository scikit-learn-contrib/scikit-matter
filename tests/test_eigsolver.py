import unittest
import numpy as np
import warnings

from skcosmo.utils import eig_solver
from skcosmo.preprocessing import StandardFlexibleScaler as SFS


class EigSolverTest(unittest.TestCase):
    def setUp(self):
        self.X = SFS().fit_transform(np.random.uniform(-5, 5, size=(100, 100)))

    def test_reverse_sorted(self):
        """
        Tests that the eigenvalues returned are in inverse order
        """
        v, _ = eig_solver(self.X)
        self.assertTrue(all([vv >= max(v[i:]) for i, vv in enumerate(v)]))

    def test_no_ncomponents(self):
        """
        Tests that if n_components is None and tol is None,
        returns full eigendecomposition
        """
        v, U = eig_solver(self.X, tol=None)
        self.assertTrue(len(v) == self.X.shape[0] and U.shape[0] == self.X.shape[0])

    def test_tolerance(self):
        """
        Tests that tolerance truncates the eigendecomposition
        """
        tol = 1e-3
        v, _ = eig_solver(self.X, tol=tol)
        self.assertGreaterEqual(max(v), tol)

    def test_add_null(self):
        """
        Tests that null eigenvectors / values are returned when n_components is
        greater than the number of significant eigenpairs
        """
        v, _ = np.linalg.eig(self.X)

        tol = 1e-3
        n_eig = len(v[v >= tol])
        n_components = self.X.shape[0]

        v, U = eig_solver(self.X, n_components=n_components, tol=tol, add_null=True)

        # Final result should be padded to the correct size
        self.assertTrue(len(v) == n_components)
        self.assertTrue(U.shape[0] == n_components)

        # All padded eigenvalues should be equal to 0
        self.assertEqual(np.max(np.abs(v[n_eig:])), 0)

    def test_no_add_null(self):
        """
        Tests that warning is raised when n_components is
        greater than the number of significant eigenpairs and add_null=False
        """
        v, _ = np.linalg.eig(self.X)

        tol = 1e-3
        n_eig = len(v[v >= tol])
        n_components = self.X.shape[0]

        with warnings.catch_warnings(record=True) as w:
            v, U = eig_solver(
                self.X, n_components=n_components, tol=tol, add_null=False
            )

            self.assertEqual(
                str(w[-1].message),
                f"There are fewer than {n_components} "
                "significant eigenpair(s). Resulting decomposition"
                f"will be truncated to {n_eig} eigenpairs.",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
