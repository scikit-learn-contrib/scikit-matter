from .Sparsified import _Sparsified
import numpy as np

from sklearn.exceptions import NotFittedError


class SparseKPCA(_Sparsified):
    """
    :param mixing: mixing parameter. Shows the relationship between SprseKPCA and SparseKRR (used only for KPCovR)
    :param kernel: the kernel used for this learning
    :param gamma: exponential factor of the rbf and sigmoid kernel
    :param degree: polynomial kernel degree
    :param coef0: free term of the polynomial and sigmoid kernel
    :param kernel_params: kernel parameter set
    :param n_active: the size of the small dataset used in learning
    :param regularization: regularization parameter for SparseKRR and SparseKRCovR
    :param tol: Relative accuracy for eigenvalues (stopping criterion) The default value of 0 implies machine precision.
    :param center: if True, centering is carried out
    :param n_jobs: The number of jobs to use for the computation the kernel. This works by breaking down the pairwise matrix into n_jobs even slices and computing them in parallel.
    :param n_components: number of components for which selection is carried out in SparseKPCA and SparseKRCovR
    """

    def __init__(
        self,
        n_components,
        mixing=0.0,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        n_active=None,
        regularization=1e-12,
        tol=0,
        center=True,
    ):
        self.pkt_ = None
        super().__init__(
            mixing=mixing,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            n_active=n_active,
            regularization=regularization,
            tol=tol,
            n_components=n_components,
            center=center,
        )

    def fit(self, X=None, y=None, Kmm=None, Knm=None):
        """
        Fit SparseKPCA model

        :param X: feature matrix
        :param y: matrix of properties
        :param Kmm: kernel evaluated between the active set samples.
        :param Knm: kernel matrix between X and X_sparse
        :return: itself
        """
        if X is None:
            assert Knm is not None and Kmm is not None
        self._define_Kmm_Knm(X, Kmm, Knm)
        # Compute eigendecomposition of kernel
        vmm, Umm = self._eig_solver(self.Kmm, k=self.n_active)

        U_active = Umm[:, : self.n_active]
        vmm_inv = np.linalg.pinv(np.diagflat(vmm))
        v_invsqrt = np.sqrt(vmm_inv)
        U_active = U_active @ v_invsqrt

        phi_active = self.Knm @ U_active

        C = phi_active.T @ phi_active

        v_C, U_C = self._eig_solver(C)

        self.pkt_ = U_active @ U_C[:, : self.n_components]
        T = self.Knm @ self.pkt_
        v_C_inv = np.linalg.pinv(np.diagflat(v_C[: self.n_components]))
        self.ptx_ = v_C_inv @ T.T @ X

    def transform(self, X=None, Knm=None):
        """
        Projecting  feature matrix into the latent space
        :param X: feature matrix
        :param Knm: kernel matrix between X and X_sparse
        :return: T, projection into the latent space
        """

        if X is None and Knm is None:
            raise Exception("Error: required feature or kernel matrices")

        if self.pkt_ is None:
            raise NotFittedError("Error: must fit the KPCA before transforming")
        else:
            if Knm is None:
                Knm = self._get_kernel(X, self.X_sparse)
            if self.center:
                Knm = self.kfc.transform(Knm)

            return self._project(Knm, "pkt_")

    def fit_transform(self, X, y=None, Kmm=None, Knm=None):
        """
        Both fit and transform

        :param X: feature matrix
        :param y: properties matrix
        :param Kmm: kernel evaluated between the active set samples.
        :param Knm: kernel matrix between X and X_sparse
        :return: T, projection into the latent space
        """
        self.fit(X, y, Kmm, Knm)
        self.transform(X, self.Knm)
