import unittest

from skcosmo.datasets import load_degenerate_CH4_manifold, load_csd_1000r


class DegenerateCH4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.degenerate_CH4_manifold = load_degenerate_CH4_manifold()

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


class CSDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csd = load_csd_1000r()

    def test_load_csd_1000r_shape(self):
        # test if representations and properties have commensurate shape
        self.assertTrue(self.csd.data.X.shape[0] == self.csd.data.y.shape[0])

    def test_load_csd_1000r_access_descr(self):
        self.csd.DESCR


if __name__ == "__main__":
    unittest.main()
