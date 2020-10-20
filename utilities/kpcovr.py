import numpy as np
from scipy.spatial.distance import cdist
from .general import get_stats, FPS, sorted_eig, eig_inv
from .kernels import linear_kernel, gaussian_kernel, center_kernel

kernels = {"gaussian": gaussian_kernel,
           "linear": linear_kernel
           }


class KPCovR:

    """
        Performs KPCovR, as described in Helfrecht (2020),
        which combines Kernel Principal Components Analysis (KPCA)
        and Kernel Ridge Regression (KRR)

        ---Arguments---
        X:              independent (predictor) variable
        Y:              dependent (response) variable
        alpha:          tuning parameter
        n_PC:          number of principal components to retain
        kernel_type:    kernel function, may be either type str or function,
                        defaults to a linear kernel

        ---Returns---
        Xp:             X values projected into the latent (PCA-like) space
        Yp:             predicted Y values
        Xr:             Reconstructed X values from the latent (PCA-like) space
        Lx:             KPCA loss
        Ly:             KR loss

        ---References---
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015
    """

    def __init__(self, alpha=0.0, n_PC=2,
                 kernel_type='linear', regularization=1e-12):

        self.alpha = alpha
        self.n_PC = n_PC
        self.regularization = regularization

        self.PKT = None
        self.PTK = None
        self.PTY = None
        self.X = None  # To save the X which was used to train the model

        if(isinstance(kernel_type, str)):
            if(kernel_type in kernels):
                self.kernel = kernels[kernel_type]
            else:
                raise Exception('Kernel Error: \
                                  Please specify either {}\
                                  or pass a suitable kernel function.\
                                '.format(kernels.keys()))
        elif(callable(kernel_type)):
            self.kernel = kernel_type
        else:
            raise Exception('Kernel Error: \
                              Please specify either {}\
                              or pass a suitable kernel function.\
                            '.format(kernels.keys()))

    def fit(self, X, Y, K=None, Yhat=None):
        """
            Fits the KPCovR to the training inputs and outputs

            ---Arguments---
            X:              independent (predictor) variable
            Y:              dependent (response) variable

            ---Returns---
            PKT:              Weights projecting kernel into latent PCA-space
            T:              Projection of kernel in latent PCA-space
            PTY:              Projector from latent PCA-space to Y-space
            PTK:             Projector from latent PCA-space to kernel-space
        """

        if(K is None):
            K = self.kernel(X, X)
            self.K = K
            K = center_kernel(K)
        else:
            self.K = K

        if(X is not None):
            self.X = X
        else:
            print("No input data supplied during fitting." +
                  "\nTransformations and statistics only" +
                  "available with pre-computed kernels.")

        # Compute maximum eigenvalue of kernel matrix

        if(Yhat is None):
            Yhat = K @ np.linalg.pinv(K, rcond=self.regularization) @ Y

        if(len(Y.shape) == 1):
            Yhat = Yhat.reshape(-1, 1)

        K_pca = K #/ (np.trace(K) / X.shape[0])
        K_lr = np.matmul(Yhat, Yhat.T)

        Kt = (self.alpha * K_pca) + (1.0 - self.alpha) * K_lr
        self.Kt = Kt

        self.v, self.U = sorted_eig(
            Kt, thresh=self.regularization, n=self.n_PC)

        P_krr = np.linalg.solve(K + np.eye(len(K)) * self.regularization, Yhat)
        P_krr = np.matmul(P_krr, Yhat.T)

        P_kpca = np.eye(K.shape[0]) #/ (np.trace(K) / K.shape[0])

        P = (self.alpha * P_kpca) + (1.0 - self.alpha) * P_krr

        v_inv = np.diagflat(eig_inv(self.v[:self.n_PC]))

        self.PKT = np.matmul(P, np.matmul(self.U[:, :self.n_PC],
                                          np.sqrt(v_inv)))
        T = np.matmul(K, self.PKT)

        PT = np.matmul(T.T, T)
        PT = np.linalg.pinv(PT)
        PT = np.matmul(PT, T.T)
        self.PTK = np.matmul(PT, K)
        self.PTY = np.matmul(PT, Y)
        self.PTX = np.matmul(PT, X)

    def transform(self, X=None, K=None):
        """
            Transforms a set of inputs using the computed KPCovR

            ---Arguments---
            X:              independent (predictor) variable

            ---Returns---
            Xp:             X values projected into the latent (PCA-like) space
            Yp:             predicted Y values

        """
        if self.PKT is None:
            print("Error: must fit the PCovR model before transforming")
        elif(X is None and K is None):
            print("Either the kernel or input data must be specified.")
        else:

            if(K is None and self.X is not None):
                K = self.kernel(X, self.X)
                K = center_kernel(K, reference=self.K)
            elif(K is None):
                print("This functionality is not available.")
                return

            T = np.matmul(K, self.PKT)
            Yp = np.matmul(T, self.PTY)
            Xr = np.matmul(T, self.PTX)

            if(T.shape[1] == 1):
                T = np.array((T.reshape(-1), np.zeros(T.shape[0]))).T
            return T, Yp, Xr

    def loss(self, X=None, Y=None, K=None):
        """
            Computes the loss values for KPCovR on the given predictor and
            response variables

            ---Arguments---
            X:              independent (predictor) variable
            Y:              dependent (response) variable

            ---Returns---
            Lx:             KPCA loss
            Ly:             KR loss

        """

        if(K is None and self.X is not None):
            K = self.kernel(X, self.X)

            K = center_kernel(K, reference=self.K)
        elif(K is None):
            raise ValueError(
                "Must provide a kernel or a feature vector, in which case the train features should be available in the class")

        Tp, Yp, Xr = self.transform(X=X, K=K)
        Kapprox = np.matmul(Tp, self.PTK)

        Lkpca = np.linalg.norm(K - Kapprox)**2/np.linalg.norm(K)**2
        Lkrr = np.linalg.norm(Y - Yp)**2/np.linalg.norm(Y)**2

        return Lkpca, Lkrr

    def statistics(self, X, Y, K=None):
        """
            Computes the loss values and reconstruction errors for
            KPCovR for the given predictor and response variables

            ---Arguments---
            X:              independent (predictor) variable
            Y:              dependent (response) variable

            ---Returns---
            dictionary of available statistics

        """

        if(K is None and self.X is not None):
            K = self.kernel(X, self.X)
            K = center_kernel(K, reference=self.K)

        Tp, Yp, Xr = self.transform(X=X, K=K)

        Kapprox = np.matmul(Tp, self.PTK)

        return get_stats(x=X, y=Y, yp=Yp, t=Tp, xr=Xr,
                         k=K, kapprox=Kapprox)


class SparseKPCovR:

    """
        Performs KPCovR, as described in Helfrecht (2020)
        which combines Kernel Principal Components Analysis (KPCA)
        and Kernel Ridge Regression (KRR), on a sparse active set
        of variables

        ---Arguments---
        X:              independent (predictor) variable
        Y:              dependent (response) variable
        alpha:          tuning parameter
        n_PC:          number of principal components to retain
        kernel_type:    kernel function, may be either type str or function,
                        defaults to a linear kernel

        ---Returns---
        Xp:             X values projected into the latent (PCA-like) space
        Yp:             predicted Y values
        Lx:             KPCA loss
        Ly:             KR loss

        ---References---
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015
    """

    def __init__(self, alpha, n_PC, n_active=100, regularization=1e-6, kernel_type="linear"):

        self.alpha = alpha
        self.n_active = n_active
        self.n_PC = n_PC
        self.regularization = regularization

        self.PKT = None
        self.PTY = None
        self.PTX = None
        self.X = None  # To save the X which was used to train the model

        if(isinstance(kernel_type, str)):
            if(kernel_type in kernels):
                self.kernel = kernels[kernel_type]
            else:
                raise Exception('Kernel Error: \
                                  Please specify either {}\
                                  or pass a suitable kernel function.\
                                '.format(kernels.keys()))
        elif(callable(kernel_type)):
            self.kernel = kernel_type
        else:
            raise Exception('Kernel Error: \
                              Please specify either {}\
                              or pass a suitable kernel function.\
                            '.format(kernels.keys()))

    def fit(self, X, Y, X_sparse=None, Kmm=None, Knm=None):
        """
            Fits the KPCovR to the training inputs and outputs

            ---Arguments---
            X:              independent (predictor) variable
            Y:              dependent (response) variable

            ---Returns---
            PKT:              Weights projecting K into latent PCA-space
            T:              Projection of K in latent PCA-space
            PTY:              Projector from latent PCA-space to Y-space
            PTX:             Projector from latent PCA-space to X-space
        """

        if(X_sparse is None):
            fps_idxs, _ = FPS(X, self.n_active)
            self.X_sparse = X[fps_idxs, :]
        else:
            self.X_sparse = X_sparse

        if(Kmm is None):
            Kmm = self.kernel(self.X_sparse, self.X_sparse)
            self.Kmm = Kmm
            Kmm = center_kernel(Kmm)
        else:
            self.Kmm = Kmm

        if(Knm is None):
            Knm = self.kernel(X, self.X_sparse)
            Knm = center_kernel(Knm, reference=self.Kmm)
        self.Knm = Knm

        vmm, Umm = sorted_eig(
            Kmm, thresh=self.regularization, n=self.n_active)
        vmm_inv = eig_inv(vmm[:self.n_active-1])

        self.barKM = np.mean(self.Knm, axis=0)

        phi_active = np.matmul(self.Knm, Umm[:, :self.n_active-1])
        phi_active = np.matmul(phi_active, np.diagflat(np.sqrt(vmm_inv)))
        barPhi = np.mean(phi_active, axis=0)
        phi_active -= barPhi

        C = np.dot(phi_active.T, phi_active)

        v_C, U_C = sorted_eig(C, thresh=0)
        U_C = U_C[:, v_C > 0]
        v_C = v_C[v_C > 0]
        v_C_inv = eig_inv(v_C)

        Csqrt = np.matmul(np.matmul(U_C, np.diagflat(np.sqrt(v_C))), U_C.T)
        iCsqrt = np.matmul(
            np.matmul(U_C, np.diagflat(np.sqrt(v_C_inv))), U_C.T)

        C_pca = C #/ (np.trace(C)/C.shape[0])

        C_lr = np.linalg.pinv(C + self.regularization*np.eye(C.shape[0]))
        C_lr = np.matmul(phi_active, C_lr)
        C_lr = np.matmul(phi_active.T, C_lr)
        C_lr = np.matmul(iCsqrt, C_lr)
        C_lr = np.matmul(C_lr, phi_active.T)

        if(len(Y.shape) == 1):
            C_lr = np.matmul(C_lr, Y.reshape(-1, 1))
        else:
            C_lr = np.matmul(C_lr, Y)

        C_lr = np.matmul(C_lr, C_lr.T)

        Ct = self.alpha*C_pca + (1-self.alpha)*C_lr

        v_Ct, U_Ct = sorted_eig(Ct, thresh=0)

        PPT = np.matmul(iCsqrt, U_Ct[:, :self.n_PC])
        PPT = np.matmul(PPT, np.diag(np.sqrt(v_Ct[:self.n_PC])))

        PKT = np.matmul(Umm[:, :self.n_active-1],
                        np.diagflat(np.sqrt(vmm_inv)))
        self.PKT = np.matmul(PKT, PPT)
        self.barT = np.matmul(barPhi, PPT)

        T = np.matmul(self.Knm, self.PKT) - self.barT

        PT = np.matmul(T.T, T)
        PT = np.linalg.pinv(PT)
        PT = np.matmul(PT, T.T)
        self.PTY = np.matmul(PT, Y)
        self.PTX = np.matmul(PT, X)

    def transform(self, X, Knm=None):
        """
            Transforms a set of inputs using the computed Sparse KPCovR

            ---Arguments---
            X:              independent (predictor) variable

            ---Returns---
            Xp:             X values projected into the latent (PCA-like) space
            Yp:             predicted Y values

        """

        if self.PKT is None:
            print("Error: must fit the PCovR model before transforming")
        else:

            if(Knm is None):
                Knm = self.kernel(X, self.X_sparse)
                Knm = center_kernel(Knm, reference=self.Kmm)

            T = np.matmul(Knm, self.PKT) - self.barT
            Yp = np.matmul(T, self.PTY)
            Xr = np.matmul(T, self.PTX)
            return T, Yp, Xr

    def loss(self, X, Y, Knm=None):
        """
            Computes the loss values for KPCovR on the given predictor and
            response variables

            ---Arguments---
            X:              independent (predictor) variable
            Y:              dependent (response) variable

            ---Returns---
            Lx:             KPCA loss
            Ly:             KR loss

        """

        if(Knm is None):
            Knm = self.kernel(X, self.X_sparse)
            Knm = center_kernel(Knm, reference=self.Kmm)

        T, Yp, Xr = self.transform(X, Knm=Knm)

        Lkpca = np.linalg.norm(Xr - X)**2/np.linalg.norm(X)**2
        Lkrr = np.linalg.norm(Y - Yp)**2/np.linalg.norm(Y)**2

        return Lkpca, Lkrr

    def statistics(self, X, Y, Knm=None):
        """
            Computes the loss values and reconstruction errors for
            KPCovR for the given predictor and response variables

            ---Arguments---
            X:              independent (predictor) variable
            Y:              dependent (response) variable

            ---Returns---
            dictionary of available statistics

        """

        if(Knm is None):
            Knm = self.kernel(X, self.X_sparse)
            Knm = center_kernel(Knm, reference=self.Kmm)

        T, Yp, Xr = self.transform(X, Knm=Knm)

        return get_stats(x=X, y=Y, yp=Yp, t=T, xr=Xr)
