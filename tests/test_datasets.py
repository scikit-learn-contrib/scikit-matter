import unittest

from skcosmo.datasets import load_degenerate_CH4_manifold, load_csd_1000r, load_nice_dataset

class NICEDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nice_data = load_nice_dataset()
        
    def test_load_nice_data(self):
        # test if representations and properties have commensurate shape
        self.assertTrue(self.nice_data.data.X.shape[0] == self.nice_data.data.y.shape[0])
        self.assertTrue(self.nice_data.data.X.shape[0] == 500)
        self.assertTrue(self.nice_data.data.X.shape[1] == 160)
        self.assertTrue(len(self.nice_data.data.X.shape) == 2)
        
    def test_load_nice_data_descr(self):
        self.nice_data.DESCR
        
        
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
