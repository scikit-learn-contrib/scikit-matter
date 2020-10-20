import numpy as np
from sklearn import exceptions
from sklearn.utils.validation import check_X_y
import unittest
from .Sparse_Base_Tests import Sparse_Base_Tests

class SparseKPCA_tests(Sparse_Base_Tests):
    def __init__(self, *args, **kwargs):
        super().__init__(model = 'KPCA', *args, **kwargs )

     # Basic Test of model PCA Errors, that None return np.nan
    def test_pca_errors(self):

         for i, mixing in enumerate(self.alphas):
             with self.subTest(error=self.pca_errors[i]):
                 self.assertFalse(np.isnan(self.pca_errors[i]))


if __name__ == '__main__':
    unittest.main()
