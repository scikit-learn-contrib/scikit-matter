from .Sparsified import _Sparsified
import numpy as np
from sklearn.preprocessing import KernelCenterer
from sklearn.exceptions import NotFittedError


class SparseKRR(_Sparsified):
    """
    TODO make a description
    Performs sparsified kernel ridge regression

    ----Inherited Attributes----
    PKT: projector from kernel space to latent space
    PTK: projector from latent space to kernel space
    PTX: projector from latent space to input space
    PXY: projector from input space to property space
    X: input used to train the model, if further kernels need be constructed
    X_sparse: active set used to train the model, if further kernels need be
              constructed
    center: (boolean) whether to shift all inputs to zero mean
    kernel: function to construct the kernel of the input data
    n_active: (int) size of the active set
    regularization: (float) parameter to offset all small eigenvalues for
                    regularization
    scale: (boolean) whether to scale all inputs to unit variance

    ----Inherited Methods----
    postprocess: un-centers and un-scales outputs according to scale and
                 center attributes
    preprocess: centers and scales provided inputs according to scale and
                center attributes

    ----Methods----
    fit: fit the kernel ridge regression model by computing regression weights
    statistics: provide available statistics for the regression
    transform: compute predicted Y values

    ----References----
        1.  M. Ceriotti, M. J. Willatt, G. Csanyi,
            'Machine Learning of Atomic - Scale Properties
            Based on Physical Principles', Handbook of Materials Modeling,
            Springer, 2018
        2.  A. J. Smola, B. Scholkopf, 'Sparse Greedy Matrix Approximation
            for Machine Learning', Proceedings of the 17th International
            Conference on Machine Learning, 911 - 918, 2000
    """

    def __init__(self, mixing=0.0, kernel="linear", gamma=None, degree=3,
                 coef0=1, kernel_params=None, n_active=None,
                 regularization=1E-12, tol=0, center=True):
        super().__init__(mixing=mixing, kernel=kernel, gamma=gamma, degree=degree,
                                         coef0=coef0, kernel_params=kernel_params, n_active=n_active,
                                         regularization=regularization, tol=tol, center=center)
    def fit(self, X, Y, Kmm=None, Knm=None):
        self._define_Kmm_Knm(X, Kmm, Knm)

        # Compute max eigenvalue of regularized model
        PKY = np.linalg.pinv(self.Knm.T @ self.Knm + (self.regularization * self.Kmm))
        self.pky_ = PKY @ self.Knm.T @ Y

    def predict(self, X=None, Knm=None):
        if X is None and Knm is None:
            raise Exception( "Error: required feature or kernel matrices" )
        if self.pky_ is None:
            raise NotFittedError("Error: must fit the KRR model before transforming")
        else:
            if Knm is None:
                Knm = self._get_kernel(X, self.X_sparse)
                Knm = KernelCenterer().fit_transform(Knm)

            Yp = Knm @ self.pky_
            return Yp

    def  transform(self, X):
        #return Knm matrix
        Knm = self._get_kernel( X, self.X_sparse )
        Knm = KernelCenterer().fit_transform(Knm)
        return Knm

    def fit_transform(self, X, y=None,Kmm=None, Knm=None):
        self.fit(X, y, Kmm, Knm)
        return self.Knm

    def fit_predict(self, X, Y, Kmm=None, Knm=None):
        self.fit(X,Y, Kmm, Knm)
        self.predict(self.Knm)
