import numpy as np
import unittest
from sklearn_sparse.sparse_tests.Sparse_Base_Tests import Sparse_Base_Tests
from sklearn_sparse.SparseKRR import SparseKRR

class SparseKRR_tests(Sparse_Base_Tests):
    def __init__(self,model='KRR', *args, **kwargs):
        super().__init__(model = 'KRR', *args, **kwargs )

     # Basic Test of model LR Errors, that None return np.nan

    def test_krr_errors(self):

         for i, mixing in enumerate(self.alphas):
             with self.subTest(error=self.lr_errors[i]):
                 self.assertFalse(np.isnan(self.lr_errors[i]))

     # Test that model LR Errors are monotonic with alpha
    def test_active_krr_monotonicity(self):
         self.add_errors_active()

         for i, _ in enumerate(self.alphas[:-1]):
             with self.subTest(i=i):
                 lr1 = round(self.lr_errors_active[i], self.rounding)
                 lr2 = round(self.lr_errors_active[i+1], self.rounding)
                 self.assertTrue(lr1 <= lr2,
                                 msg=f'LR Error Non-Monotonic\n {lr1} >  {lr2}'
                                 )
if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase( SparseKRR_tests )
    unittest.TextTestRunner().run( suite )