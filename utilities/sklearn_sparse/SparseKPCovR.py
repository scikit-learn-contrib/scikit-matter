from .Sparsified import _Sparsified
import numpy as np
from sklearn.preprocessing import KernelCenterer
from sklearn.utils.validation import check_X_y
from sklearn.exceptions import NotFittedError


class SparseKPCovR(_Sparsified):
    """
    Performs KPCovR, as described in Helfrecht (2020)
    which combines Kernel Principal Components Analysis (KPCA)
    and Kernel Ridge Regression (KRR), on a sparse active set
    of variables

    ----Inherited Attributes----
    PKT: projector from kernel space to latent space
    PKY: projector from kernel space to property space
    PTK: projector from latent space to kernel space
    PTX: projector from latent space to input space
    PTY: projector from latent space to property space
    PXT: projector from input space to latent space
    PXY: projector from input space to property space
    X: input used to train the model, if further kernels need be constructed
    X_sparse: active set used to train the model, if further kernels need be
              constructed
    Yhat: regressed properties
    alpha: (float) mixing parameter between decomposition and regression
    center: (boolean) whether to shift all inputs to zero mean
    kernel: function to construct the kernel of the input data
    n_PC (int) number of principal components to store
    n_active: (int) size of the active set
    regularization: (float) parameter to offset all small eigenvalues for
                    regularization
    scale: (boolean) whether to scale all inputs to unit variance

    ----Inherited Methods----
    postprocess: un-centers and un-scales outputs according to scale and
                 center attributes
    preprocess: centers and scales provided inputs according to scale and
                center attributes

    ---References---
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015
    """

    def __init__(self, n_components, mixing=0.0, kernel="linear", gamma=None, degree=3,
                 coef0=1, kernel_params=None, n_active=None,
                 regularization=1E-12, tol=0, center=True):
        super().__init__(mixing=mixing, kernel=kernel, gamma=gamma, degree=degree,
                                         coef0=coef0, kernel_params=kernel_params, n_active=n_active,
                                         regularization=regularization, tol=tol, n_components=n_components, center=center)

    def fit(self, X, Y, X_sparse=None, Kmm=None, Knm=None):

        X, Y = check_X_y( X, Y, y_numeric=True, multi_output=True )
        if X_sparse is None:
            fps_idxs, _ = self.FPS(X, self.n_active)
            self.X_sparse = X[fps_idxs, :]
        else:
            self.X_sparse = X_sparse

        if Kmm is None:
            Kmm = self._get_kernel(self.X_sparse, self.X_sparse)

        if self.center:
            Kmm = KernelCenterer().fit_transform(Kmm)
        self.Kmm = Kmm

        if Knm is None:
            Knm = self._get_kernel(X, self.X_sparse)
        Knm = KernelCenterer().fit_transform(Knm)
        self.Knm = Knm
        vmm, Umm = self._eig_solver(self.Kmm, k=self.n_active)
        vmm_inv = np.linalg.pinv( np.diagflat(vmm[:self.n_active - 1]) )
        phi_active = self.Knm @ Umm[:, :self.n_active - 1] @ np.sqrt(vmm_inv)
        C = phi_active.T @ phi_active

        v_C, U_C = self._eig_solver(C, tol=0, k = self.n_active)
        U_C = U_C[:, v_C > 0]
        v_C = v_C[v_C > 0]
        v_C_inv = np.linalg.pinv( np.diagflat(v_C))
        Csqrt = U_C @ np.diagflat(np.sqrt(v_C)) @ U_C.T
        iCsqrt = U_C @ np.sqrt(v_C_inv) @ U_C.T

        C_pca = C

        C_lr = np.linalg.pinv(C + self.regularization * np.eye(C.shape[0]))
        C_lr = iCsqrt @ phi_active.T @ phi_active @ C_lr @ phi_active.T

        if len(Y.shape) == 1:
            C_lr = C_lr @ Y.reshape(-1, 1)
        else:
            C_lr = C_lr @ Y

        C_lr = C_lr @ C_lr.T

        Ct = self.mixing * C_pca + (1 - self.mixing) * C_lr

        v_Ct, U_Ct = self._eig_solver(Ct, tol=0, k=self.n_active)
        PPT = iCsqrt @ U_Ct[:, :self.n_components] @ np.diag(np.sqrt(v_Ct[:self.n_components]))

        PKT = Umm[:, :self.n_active - 1] @ np.sqrt(vmm_inv)

        self.pkt_ = PKT @ PPT

        T = self.Knm @ self.pkt_

        PT = np.linalg.pinv(T.T @ T) @ T.T
        self.pty_ = PT @ Y
        self.ptx_ = PT @ X

    def transform(self, X=None, Knm=None):

        if X is None and Knm is None:
            raise Exception( "Error: required feature or kernel matrices" )

        if self.pkt_ is None:
            raise NotFittedError("Error: must fit the KPCovR before transforming")
        else:
            if Knm is None:
                Knm = self._get_kernel(X, self.X_sparse)
                Knm = KernelCenterer().fit_transform(Knm)

            return self._project( Knm, 'pkt_')

    def predict(self, X=None, K=None):
        T = self.transform(X, K)
        return self._project(T,'pty_')

    def fit_transform(self, X, Y=None, X_sparse=None, Kmm=None, Knm=None):
        self.fit(X,Y, X_sparse, Kmm, Knm)
        return self.transform(X, Knm)

    def fit_predict(self, X, Y=None, X_sparse=None, Kmm=None, Knm=None, K=None):
        self.fit( X, Y, X_sparse, Kmm, Knm )
        return self.predict(X, K)

    def inverse_transform(self, T):
        if T is None:
            raise Exception( "Error: required feature matrix" )

        if self.ptx_ is None:
            raise NotFittedError("Error: must fit the KPCovR before transforming")
        return self._project(T,'ptx_')
