import numpy as np
from sklearn import exceptions
from sklearn.utils.validation import check_X_y
import unittest
from sklearn_sparse.sparse_tests.Sparse_Base_Tests import Sparse_Base_Tests

class SparseKPCovR_tests(Sparse_Base_Tests):
    def __init__(self,model='KPCovR', *args, **kwargs):
        super().__init__(model = 'KPCovR', *args, **kwargs )

    def add_errors(self):
         for i, mixing in enumerate(self.alphas):
             if(np.isnan(self.lr_errors[i]) or np.isnan(self.pca_errors[i])):
                 Yp, _, Xr = self.run_sparse(mixing=mixing)
                 self.lr_errors[i] = self.rel_error(self.Y, Yp)
                 self.pca_errors[i] = self.rel_error(self.X, Xr)

     # Basic Test of model PCA Errors, that None return np.nan
    def test_pca_errors(self):

         for i, mixing in enumerate(self.alphas):
             with self.subTest(error=self.pca_errors[i]):
                 self.assertFalse(np.isnan(self.pca_errors[i]))

    # Basic Test of model LR Errors, that None return np.nan
    def test_krr_errors(self):

         for i, mixing in enumerate(self.alphas):
             with self.subTest(error=self.lr_errors[i]):
                 self.assertFalse(np.isnan(self.lr_errors[i]))


     # Test that model LR Errors are monotonic with alpha
    def test_krr_monotonicity(self):
         self.add_errors()

         for i, _ in enumerate(self.alphas[:-1]):
             with self.subTest(i=i):
                 lr1 = round(self.lr_errors[i], self.rounding)
                 lr2 = round(self.lr_errors[i+1], self.rounding)
                 self.assertTrue(lr1 <= lr2,
                                 msg=f'LR Error Non-Monotonic\n {lr1} >  {lr2}'
                                 )

     # Test that model PCA Errors are monotonic with alpha
    def test_pca_monotonicity(self):
         self.add_errors()

         for i, a in enumerate(self.alphas[:-1]):
             with self.subTest(i=i):
                 pca1 = round(self.pca_errors[i], self.rounding)
                 pca2 = round(self.pca_errors[i+1], self.rounding)
                 self.assertTrue(pca1 >= pca2,
                                 msg=f'PCA Error Non-Monotonic\n {pca2} >  {pca1}'
                                 )

    # Test that model PCA Errors are monotonic with n_active
    def test_active_pca_monotonicity(self):
         self.add_errors_active()

         for i, a in enumerate(self.n_active):
             with self.subTest(i=i):
                 pca1 = round(self.pca_errors_active[i], self.rounding)
                 pca2 = round(self.pca_errors_active[i+1], self.rounding)
                 self.assertTrue(pca2 >= pca1,
                                 msg=f'PCA Error Non-Monotonic\n {pca2} >  {pca1}'
                                 )

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

    def test_T_shape(self):
         _, T, _ = self.run_sparse(mixing=0.5)
         self.assertTrue(check_X_y(self.X, T, multi_output=True))
if __name__ == '__main__':
    unittest.main()
