from .Sparsified import _Sparsified
import numpy as np
from sklearn.preprocessing import KernelCenterer
from sklearn.exceptions import NotFittedError

class SparseKPCA(_Sparsified):
    """
    #TODO make a description
    ----Inherited Attributes----
    PKT: projector from kernel space to latent space
    PKY: projector from kernel space to property space
    PKY: projector from kernel space to property space (null)
    PTK: projector from latent space to kernel space
    PXT: projector from input space to latent space
    X: input used to train the model, if further kernels need be constructed
    X_sparse: active set used to train the model, if further kernels need be
             constructed
    center: (boolean) whether to shift all inputs to zero mean
    kernel: function to construct the kernel of the input data
    n_PC (int) number of principal components to store
    n_active: (int) size of the active set
    regularization: (float) parameter to offset all small eigenvalues for
                     regularization
    scale: (boolean) whether to scale all inputs to unit variance

    ----Methods----
    fit: fit the KPCA
    statistics: provide available statistics for the decomposition
    transform: transform data based on the KPCA fit

    ----Inherited Methods----
    postprocess: un-centers and un-scales outputs according to scale and
                 center attributes
    preprocess: centers and scales provided inputs according to scale and
                center attributes

    ----References----
        1.  https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
        2.  M. E. Tipping, 'Sparse Kernel Principal Component Analysis',
            Advances in Neural Information Processing Systems 13, 633 - 639, 2001
    """

    def __init__(self, n_components, mixing=0.0, kernel="linear", gamma=None, degree=3,
                 coef0=1, kernel_params=None, n_active=None,
                 regularization=1E-12, tol=0, center=True):
        super().__init__(mixing=mixing, kernel=kernel, gamma=gamma, degree=degree,
                                         coef0=coef0, kernel_params=kernel_params, n_active=n_active,
                                         regularization=regularization, tol=tol, n_components=n_components, center=center)

    def fit(self, X,y=None, Kmm=None, Knm=None):
        self._define_Kmm_Knm(X, Kmm, Knm)
         # Compute eigendecomposition of kernel
        vmm, Umm = self._eig_solver(self.Kmm, k=self.n_active)

        U_active = Umm[:, :self.n_active ]
        vmm_inv = np.linalg.pinv(np.diagflat(vmm))
        v_invsqrt = np.sqrt(vmm_inv)
        U_active = U_active @ v_invsqrt

        phi_active = self.Knm @ U_active

        C = phi_active.T @ phi_active

        v_C, U_C = self._eig_solver(C)

        self.pkt_ = U_active @ U_C[:, :self.n_components]
        T = self.Knm @ self.pkt_
        v_C_inv = np.linalg.pinv( np.diagflat(v_C[:self.n_components]) )
        self.ptx_ = v_C_inv @ T.T @ X

    def predict(self, X):
        pass
    #TODO deside, wgat we should with predict in this case. We should realize predict(abs of Sparcified)

    def transform(self, X=None, Knm=None):

        if X is None and Knm is None:
            raise Exception( "Error: required feature or kernel matrices" )

        if self.pkt_ is None:
            raise NotFittedError("Error: must fit the KPCA before transforming")
        else:
            if Knm is None:
                Knm = self._get_kernel(X, self.X_sparse)
                Knm = KernelCenterer().fit_transform(Knm)

            return self._project( Knm, 'pkt_' )

    def fit_transform(self, X, y=None,Kmm=None, Knm=None):
        self.fit(X,y,Kmm,Knm)
        self.transform(X,self.Knm)
