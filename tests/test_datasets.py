import unittest

from skcosmo.datasets import load_degenerate_manifold


class BaseTests(unittest.TestCase):
    def test_load_degenerate_manifold(self):
        # test if representations have correct shape
        degenerate_manifold = load_degenerate_manifold()
        self.assertTrue(degenerate_manifold.data.SOAP_power_spectrum.shape == (162, 12))
        self.assertTrue(degenerate_manifold.data.SOAP_bispectrum.shape == (162, 12))
        # test access
        degenerate_manifold.DESCR


if __name__ == "__main__":
    unittest.main()
