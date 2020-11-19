import unittest

from skcosmo.datasets import load_degenerate_CH4_manifold


class BaseTests(unittest.TestCase):
    def setUp(self):
        self.degenerate_CH4_manifold = load_degenerate_CH4_manifold()

    def test_load_degenerate_CH4_manifold_power_spectrum_shape(self):
        # test if representations have correct shape
        self.assertTrue(
            self.degenerate_CH4_manifold.data.SOAP_power_spectrum.shape == (162, 12)
        )

    def test_load_degenerate_CH4_manifold_bispectrum_shape(self):
        self.assertTrue(
            self.degenerate_CH4_manifold.data.SOAP_bispectrum.shape == (162, 12)
        )

    def test_load_degenerate_CH4_manifold_access_descr(self):
        self.degenerate_CH4_manifold.DESCR


if __name__ == "__main__":
    unittest.main()
