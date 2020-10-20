import numpy as np
from sklearn import exceptions
from sklearn.utils.validation import check_X_y
import unittest
import time


class Sparse_Base_Tests(unittest.TestCase):
    def __init__(self,model, *args, **kwargs):
         super().__init__(*args, **kwargs)

         self.data = np.load('CSD-test.npz')
         self.X = self.data["X"]
         self.Y = self.data["Y"]
         self.Knm = None
         self.model = model

         self.error_tol = 1E-3
         self.rounding = -int(round(np.log10(self.error_tol)))

         self.kernels =["linear", "poly", "rbf", "sigmoid", "cosine"]
         n_mixing = 11
         n_active=10
         self.lr_errors = np.nan * np.zeros(n_mixing)
         self.pca_errors = np.nan * np.zeros(n_mixing)
         self.lr_errors_active = np.nan * np.zeros(n_active)
         self.pca_errors_active = np.nan * np.zeros(n_active)
         self.alphas = np.linspace(0, 1, n_mixing)
         self.n_active = [i for i in range(10,21)]

    def setUp(self):
         self.startTime = time.time()

    def tearDown(self):
         t = time.time() - self.startTime
         print('%s: %.3f' % (self.id(), t))

    def rel_error(self, A, B):
         return np.linalg.norm(A-B)**2.0 / np.linalg.norm(A)**2.0

    def add_errors_active(self):
        if (self.model == 'KPCovR'):
             for i, n_active in enumerate(self.n_active):
                 if(np.isnan(self.lr_errors_active[i]))or(np.isnan(self.pca_errors_active[i])) :
                     Yp, _, Xr = self.run_sparse(n_active=n_active)
                     self.lr_errors_active[i] = self.rel_error(self.Y, Yp)
                     self.pca_errors_active[i] = self.rel_error(self.X, Xr)
        elif(self.model=='KRR'):
            for i, n_active in enumerate( self.n_active ):
                if (np.isnan( self.pca_errors_active[i] )):
                    Yp  = self.run_sparse( n_active=n_active )
                    self.lr_errors_active[i] = self.rel_error( self.Y, Yp )


    def run_sparse(self, mixing=0.5,n_active=20, kernel='linear'):
        if (self.model =='KPCovR'):
             skpcovr = self.model(mixing=mixing,
                                n_components=2,
                                tol=1E-12, n_active=n_active,
                                kernel = kernel)
             if(self.Knm is None):
                 skpcovr.fit(self.X, self.Y)
                 Yp = skpcovr.predict(self.X)
                 T = skpcovr.transform(self.X)
             else:
                 skpcovr.fit(self.X, self.Y, Knm=self.Knm)
                 Yp = skpcovr.predict(self.X, Knm=self.Knm)
                 T = skpcovr.transform(self.X, Knm=self.Knm)
             Xr = skpcovr.inverse_transform(T)
             return Yp, T, Xr
        elif(self.model =='KPCA'):
            skpca = self.model( mixing=mixing,
                                  n_components=2,
                                  tol=1E-12, n_active=n_active,
                                  kernel=kernel )
            if (self.Knm is None):
                skpca.fit( self.X, self.Y )
                T = skpca.transform( self.X )
            else:
                skpca.fit( self.X, self.Y, Knm=self.Knm )
                T = skpca.transform( self.X, Knm=self.Knm )
            return T
        else:
            skrr = self.model( mixing=mixing,
                                tol=1E-12, n_active=n_active,
                                kernel=kernel)
            if (self.Knm is None):
                skrr.fit( self.X, self.Y )
                Yp = skrr.predict( self.X )
            else:
                skrr.fit( self.X, self.Y, Knm=self.Knm )
                Yp = skrr.predict( self.X, Knm=self.Knm )
            return Yp



    # Checks that the model will not transform before fitting
    def transform_nonfitted_failure(self):
        if(self.model=='KPCovR')or(self.model=='KPCA'):
             model = self.model(mixing=0.5,
                                n_components=2,
                                tol=1E-12,
                                n_active=20)
             with self.assertRaises(exceptions.NotFittedError):
                 if(self.Knm is not None):
                     _ = model.transform(self.X, Knm=self.Knm)
                 else:
                     _ = model.transform(self.X)


    # Checks that the model will not predict before fitting
    def predict_nonfitted_failure(self):
        if(self.model=='KPCovR')or(self.model=='KRR'):
             model = self.model(mixing=0.5,
                                n_components=2,
                                tol=1E-12,
                                n_active=20)
             with self.assertRaises(exceptions.NotFittedError):
                 if(self.Knm is not None):
                     _ = model.predict(self.X, Knm=self.Knm)
                 else:
                     _ = model.predict(self.X)

    #Check work of the algorithm with different kernels
    def test_kernel(self):
        for i, kernel in enumerate( self.kernels ):
            try:
                self.run_sparse(kernel=kernel)
            except: raise Exception(f'Kernel \'{kernel}\' doesn\'t work')