import unittest
import numpy as np

from skmatter.datasets import (
    load_degenerate_CH4_manifold,
    load_csd_1000r,
    load_nice_dataset,
    load_who_dataset,
)


class NICEDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nice_data = load_nice_dataset()

    def test_load_nice_data(self):
        # test if representations and properties have commensurate shape
        self.assertTrue(
            self.nice_data.data.X.shape[0] == self.nice_data.data.y.shape[0]
        )
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


class WHOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.size = 24240
        cls.shape = (2020, 12)
        cls.value = 5.00977993011475
        try:
            import pandas as pd  # noqa F401

            cls.has_pandas = True
            cls.who = load_who_dataset()
        except ImportError:
            cls.has_pandas = False

    def test_load_dataset_without_pandas(self):
        """
        Check if the correct exception occurs when pandas isn't present.
        """
        if self.has_pandas is False:
            with self.assertRaises(ImportError) as cm:
                _ = load_who_dataset()
            self.assertEqual(str(cm.exception), "load_who_dataset requires pandas.")

    def test_dataset_size_and_shape(self):
        """
        Check if the correct number of datapoints are present in the dataset.
        Also check if the size of the dataset is correct.
        """
        if self.has_pandas is True:
            self.assertEqual(self.who["data"].size, self.size)
            self.assertEqual(self.who["data"].shape, self.shape)

    def test_datapoint_value(self):
        """
        Check if the value of a datapoint at a certain location is correct.
        """
        if self.has_pandas is True:
            self.assertTrue(
                np.allclose(
                    self.who["data"]["SE.XPD.TOTL.GD.ZS"][1924], self.value, rtol=1e-6
                )
            )


if __name__ == "__main__":
    unittest.main()
