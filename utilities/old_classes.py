import numpy as np
from .general import FPS, get_stats, sorted_eig, eig_inv
from .kernels import linear_kernel, center_kernel, gaussian_kernel

kernels = {
    "gaussian": gaussian_kernel,
    "linear": linear_kernel
}


class PCA:
    """
        Performs principal component analysis

       ---Attributes---
        n_PC: number of PCA components to retain
            (`None` retains all components)
        C: covariance matrix of the data
        self.U: eigenvalues of the covariance matrix
        V: eigenvectors of the covariance matrix

       ---Methods---
        fit: fit the PCA
        transform: transform data based on the PCA fit

       ---References---
        1.  https: /  / en.wikipedia.org / wiki / Principal_component_analysis
        2.  M. E. Tipping 'Sparse Kernel Principal Component Analysis',
            Advances in Neural Information Processing Systems 13, 633 - 639, 2001
    """

    def __init__(self, n_PC=None, regularization=1e-12):

        # Initialize attributes

        self.n_PC = n_PC
        self.C = None
        self.PXT = None
        self.PTX = None
        self.regularization = regularization

    def fit(self, X):
        """
            Fits the PCA

           ---Arguments---
            X: centered data on which to build the PCA
        """

        # Compute covariance
        self.C = np.matmul(X.T, X) / (X.shape[0] - 1)

        # Compute eigendecomposition of covariance matrix
        v, U = sorted_eig(self.C, thresh=self.regularization, n=self.n_PC)

        self.PXT = U[:, :self.n_PC]
        self.PTX = self.PXT.T

    def transform(self, X):
        """
            Transforms the PCA

           ---Arguments---
            X: centered data to transform based on the PCA
        """

        if self.PXT is None:
            print("Error: must fit the PCA before transforming")
        else:

            # Compute PCA scores
            T = np.matmul(X, self.PXT)
            return T

    def statistics(self, X):
        T = self.transform(X)
        X_PCA = np.matmul(T, self.PTX)
        return get_stats(x=X, t=T, xr=X_PCA)


class LR:
    """
        Performs linear regression

       ---Attributes---
        w: regression weights

       ---Methods---
        fit: fit the linear regression model by computing regression weights
        transform: compute predicted Y values

       ---References---
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligents
            Laboratory Systems 14(1): 155-164, 1992
        2.  https://en.wikipedia.org/wiki/Linear_regression
    """

    def __init__(self, regularization=1e-12):
        self.PXY = None
        self.regularization = regularization

    def fit(self, X, Y):
        """
            Fits the linear regression model

           ---Arguments---
            X: centered, independent (predictor) variable
            Y: centered, dependent (response) variable
        """

        # Compute inverse of covariance
        XTX = np.matmul(X.T, X)
        XTX = XTX + self.regularization * np.eye(X.shape[1])
        XTX = np.linalg.pinv(XTX)

        # Compute LR solution
        self.PXY = np.matmul(XTX, X.T)
        self.PXY = np.matmul(self.PXY, Y)

    def transform(self, X):
        """
            Computes predicted Y values

           ---Arguments---
            X: centered, independent (predictor) variable
        """

        # Compute predicted Y
        Yp = np.matmul(X, self.PXY)

        return Yp

    def statistics(self, X, Y):
        Yp = self.transform(X)
        return get_stats(x=X, y=Y, yp=Yp)


class KPCA:
    """
        Performs kernel principal component analysis on a dataset
        based on a kernel between all of the constituent data points

       ---Arguments---
        K: kernel matrix
        n_PC: number of principal components to retain in the decomposition

       ---Returns---
        T: KPCA scores

       ---References---
        1.  https: /  / en.wikipedia.org / wiki / Kernel_principal_component_analysis
        2.  M. E. Tipping, 'Sparse Kernel Principal Component Analysis',
            Advances in Neural Information Processing Systems 13, 633 - 639, 2001
    """

    def __init__(self, n_PC, regularization=1e-12, kernel_type="linear"):
        self.n_PC = n_PC
        self.PTX = None
        self.PKT = None
        self.PTK = None
        self.regularization = regularization
        self.v, self.U = None, None

        if(isinstance(kernel_type, str)):
            self.kernel = kernels[kernel_type]
        else:
            self.kernel = kernel_type

    def fit(self, X=None, K=None):

        if(K is None):
            K = self.kernel(X, X)
            K = center_kernel(K)

        if(X is not None):
            self.X = X
        else:
            print("No input data supplied during fitting. \
                   \nTransformations/statistics only available for kernel inputs.")

        # Compute eigendecomposition of kernel
        self.v, self.U = sorted_eig(
            K, thresh=self.regularization, n=self.n_PC)

        v_inv = eig_inv(self.v[:self.n_PC])

        self.PKT = np.matmul(self.U[:, :self.n_PC],
                             np.diagflat(np.sqrt(v_inv)))
        self.T = np.matmul(K, self.PKT)
        self.PTK = np.matmul(np.diagflat(v_inv),
                             np.matmul(self.T.T, K))
        if(X is not None):
            self.PTX = np.matmul(np.diagflat(v_inv),
                                 np.matmul(self.T.T, X))

    def transform(self, X=None, K=None):
        if self.PKT is None:
            print("Error: must fit the KPCA before transforming")
        elif X is None and K is None:
            print("Either the kernel or input data must be specified.")
        else:

            # Compute KPCA transformation
            if(K is None):
                K = self.kernel(X, self.X)
                K = center_kernel(K, reference=self.kernel(self.X, self.X))

            T = np.matmul(K, self.PKT)

            return T

    def statistics(self, X=None, Y=None, K=None):
        if X is None and K is None:
            print("Either the kernel or input data must be specified.")
        else:
            # Compute KPCA transformation

            if(K is None):
                K = self.kernel(X, self.X)
                K = center_kernel(K, reference=self.kernel(self.X, self.X))

            T = np.matmul(K, self.PKT)
            Kapprox = np.matmul(T, self.PTK)
            if(self.PTX is not None):
                Xr = np.matmul(T, self.PTX)
            else:
                Xr = None

            return get_stats(k=K, kapprox=Kapprox, x=X, xr=Xr,
                             t=T)


class KRR:
    """
        Performs kernel ridge regression

       ---Arguments---
        K: kernel matrix
        Y: property values
        regularization: regularization factor to ensure positive definiteness

       ---Returns---
        w: regression weights

       ---References---
        1.  M. Ceriotti, M. J. Willatt, G. Csanyi,
            'Machine Learning of Atomic - Scale Properties
            Based on Physical Principles', Handbook of Materials Modeling,
            Springer, 2018
    """

    def __init__(self, regularization=1.0E-16, kernel_type="linear"):

        self.regularization = regularization
        self.PKY = None
        self.X = None

        if(isinstance(kernel_type, str)):
            self.kernel = kernels[kernel_type]
        else:
            self.kernel = kernel_type

    def fit(self, X=None, Y=None, K=None):

        if(K is None):
            K = self.kernel(X, X)
            self.K = K.copy()
            K = center_kernel(K, self.K)

        if(X is not None):
            self.X = X
        else:
            print("No input data supplied during fitting. \
                   \nTransformations/statistics only available for kernel inputs.")

        # Regularize the model
        Kreg = K + np.eye(K.shape[0]) * self.regularization

        # Solve the model
        self.PKY = np.linalg.solve(Kreg, Y)

    def transform(self, X=None, K=None):
        if self.PKY is None:
            print("Error: must fit the KPCA before transforming")
        elif X is None and K is None:
            print("Either the kernel or input data must be specified.")
        else:

            # Compute KPCA transformation
            if(K is None):
                K = self.kernel(X, self.X)
                K = center_kernel(K, reference=self.K)

            Yp = np.matmul(K, self.PKY)

            return Yp

    def statistics(self, X=None, Y=None, K=None):

        Yp = self.transform(X=X, K=K)

        return get_stats(x=X, y=Y, yp=Yp, k=K)


class SparseKPCA:
    """
        Performs kernel principal component analysis on a dataset
        based on a kernel between furthest point sampling of the constituent
        data points

       ---Arguments---
        K: kernel matrix
        n_PC: number of principal components to retain in the decomposition

       ---Returns---
        T: KPCA scores

       ---References---
        1.  https: /  / en.wikipedia.org / wiki / Kernel_principal_component_analysis
        2.  M. E. Tipping, 'Sparse Kernel Principal Component Analysis',
            Advances in Neural Information Processing Systems 13, 633 - 639, 2001
    """

    def __init__(self, n_PC, n_active=100, regularization=1e-12,
                 kernel_type="linear"):

        self.n_PC = n_PC
        self.n_active = n_active
        self.PKT = None
        self.X = None
        self.regularization = regularization

        if(isinstance(kernel_type, str)):
            self.kernel = kernels[kernel_type]
        else:
            self.kernel = kernel_type

    def fit(self, X, i_sparse=None, X_sparse=None, Kmm=None, Knm=None):

        if(i_sparse is None):
            i_sparse, _ = FPS(X, self.n_active)

        if(X_sparse is None):
            self.X_sparse = X[i_sparse, :]
        else:
            self.X_sparse = X_sparse

        if(Kmm is None):
            Kmm = self.kernel(self.X_sparse, self.X_sparse)
            Kmm = center_kernel(Kmm)
        self.Kmm = Kmm

        if(Knm is None):
            Knm = self.kernel(X, self.X_sparse)
            Knm = center_kernel(Knm, Kmm)
        self.barKM = np.mean(Knm, axis=0)

        # Compute eigendecomposition of kernel
        self.vmm, self.Umm = sorted_eig(
            Kmm, thresh=self.regularization, n=self.n_active)

        U_active = self.Umm[:, :self.n_active - 1]
        v_invsqrt = np.diagflat(np.sqrt(eig_inv(self.vmm[0:self.n_active - 1])))
        self.U_active = np.matmul(U_active, v_invsqrt)

        phi_active = np.matmul(Knm - self.barKM, self.U_active)

        C = np.dot(phi_active.T, phi_active)

        self.v_C, self.U_C = sorted_eig(
            C, thresh=self.regularization, n=self.n_active)

        self.PKT = np.matmul(self.U_active, self.U_C[:, :self.n_PC])
        self.T = np.matmul(Knm - self.barKM, self.PKT)
        self.PTX = np.matmul(np.diagflat(
            eig_inv(self.v_C[:self.n_PC])), np.matmul(self.T.T, X))

    def transform(self, X, Knm=None):
        if self.PKT is None:
            print("Error: must fit the KPCA before transforming")
        else:
            if(Knm is None):
                Knm = self.kernel(X, self.X_sparse)

            # Compute KPCA transformation
            T = np.matmul(Knm - self.barKM, self.PKT)

            return T

    def statistics(self, X, Knm=None, K_test=None):

        T = self.transform(X, Knm=Knm)
        Kapprox = np.matmul(T, T.T)

        if(K_test is None):
            K_test = self.kernel(X, X)
            K_test = center_kernel(K_test)

        Xr = np.matmul(T, self.PTX)

        return get_stats(k=K_test, kapprox=Kapprox, x=X, xr=Xr,
                         t=T)


class SparseKRR:
    """
        Performs sparsified kernel ridge regression

        TODO: Put in terms of X, Y
       ---Arguments---
        Knm: kernel between the whole dataset and the 'representative' data points
        Kmm: kernel between the 'representative' data points and themselves
        Y: property values
        sigma: regularization parameter
        regularization: additional regularization scale based on
                        the maximum eigenvalue of sigma * Kmm + Knm.T x Knm

       ---Returns---
        Yp: predicted property values

       ---References---
        1.  M. Ceriotti, M. J. Willatt, G. Csanyi,
            'Machine Learning of Atomic - Scale Properties
            Based on Physical Principles', Handbook of Materials Modeling,
            Springer, 2018
        2.  A. J. Smola, B. Scholkopf, 'Sparse Greedy Matrix Approximation
            for Machine Learning', Proceedings of the 17th International
            Conference on Machine Learning, 911 - 918, 2000
    """

    def __init__(self, regularization=1.0E-16,
                 n_active=100, kernel_type='linear'):

        self.regularization = regularization
        self.n_active = n_active
        self.w = None

        if(isinstance(kernel_type, str)):
            self.kernel = kernels[kernel_type]
        else:
            self.kernel = kernel_type

    def fit(self, X, Y, i_sparse=None, X_sparse=None, Kmm=None, Knm=None):

        if(i_sparse is None):
            i_sparse, _ = FPS(X, self.n_active)

        if(X_sparse is None):
            self.X_sparse = X[i_sparse, :]
        else:
            self.X_sparse = X_sparse

        if(Kmm is None):
            Kmm = self.kernel(self.X_sparse, self.X_sparse)
            Kmm = center_kernel(Kmm)
        self.Kmm = Kmm

        if(Knm is None):
            Knm = self.kernel(X, self.X_sparse)
            Knm = center_kernel(Knm, Kmm)

        # Compute max eigenvalue of regularized model
        PKY = np.linalg.pinv(np.matmul(Knm.T, Knm) + self.regularization * Kmm)
        PKY = np.matmul(PKY, Knm.T)
        self.PKY = np.matmul(PKY, Y)

    def transform(self, X, Knm=None):
        if self.PKY is None:
            print("Error: must fit the KRR model before transforming")
        else:
            if(Knm is None):
                Knm = self.kernel(X, self.X_sparse)

            Yp = np.matmul(Knm, self.PKY)

            return Yp

    def statistics(self, X, Y, Knm=None):
        P = self.transform(X, Knm=Knm)

        return get_stats(x=X, y=Y, yp=P)


class MDS:
    """
        Performs multidimensional scaling

       ---Attributes---
        n_MDS: number of PCA components to retain
            (`None` retains all components)
        K: inner product of the data, hereon kernel
        U: eigenvalues of the kernel matrix
        V: eigenvectors of the kernel matrix

       ---Methods---
        fit: fit the MDS
        transform: transform data based on the MDS fit

       ---References---
        1.  https: /  / en.wikipedia.org / wiki / Multidimensional_scaling
        2.  Torgerson, W.S. 'Multidimensional scaling: I. Theory and method',
            Psychometrika 17, 401 - 419, 1952
            https: /  / doi.org / 10.1007 / BF02288916
    """

    def __init__(self, n_MDS=None, regularization=1e-12):

        # Initialize attributes

        self.n_MDS = n_MDS
        self.K = None
        self.PXT = None
        self.regularization = regularization

    def fit(self, X):
        """
            Fits the PCA

           ---Arguments---
            X: centered and normalized data on which to build the MDS
        """

        # Compute covariance
        self.K = np.matmul(X, X.T)

        # Compute eigendecomposition of covariance matrix
        v, U = sorted_eig(self.K, thresh=self.regularization, n=self.n_MDS)

        T = np.matmul(U[:, :self.n_MDS], np.diagflat(np.sqrt(v[:self.n_MDS])))
        self.PXT = np.linalg.lstsq(X, T, rcond=None)[0]
        self.PTX = np.linalg.lstsq(T, X, rcond=None)[0]

    def transform(self, X):
        """
            Transforms using MDS

           ---Arguments---
            X: centered data to transform based on MDS
        """

        if self.PXT is None:
            print("Error: must fit the MDS before transforming")
        else:

            # Compute PCA scores
            T = np.matmul(X, self.PXT)
            return T

    def statistics(self, X):
        T = self.transform(X)
        Xr = np.matmul(T, self.PTX)
        stats = get_stats(x=X, t=T, xr=Xr)
        kernel_error = np.linalg.norm(np.matmul(X, X.T) - np.matmul(T, T.T)) ** 2.0
        stats['Strain'] = kernel_error / np.linalg.norm(np.matmul(X, X.T))
        return stats


class PCovR:
    """
        Performs PCovR, detecting whether the data set is in Structure or
        Feature Space

       ---Arguments---
        X: independent (predictor) variable, centered and normalized
        Y: dependent (response) variable, centered and normalized
        alpha: tuning parameter
        n_PC: number of principal components to retain
        loss: compute individual PCA and linear regression loss terms

       ---Returns---
        Xp: X values projected into the latent (PCA - like) space
        Xr: reconstructed X values from the PCA
        Yp: predicted Y values
        B: regression coefficients for predicting Y
        Lx: PCA loss
        Ly: linear regression loss

       ---References---
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015
    """

    def __init__(self, alpha=0.0, n_PC=None, regularization=1e-12, space='auto'):
        self.alpha = alpha
        self.n_PC = n_PC
        self.regularization = regularization

        self.v = None
        self.U = None
        self.PXT = None
        self.PTX = None
        self.PTY = None

        self.space = space

    def compute_K(self, X, Y, Yhat=None):

        # Compute K
        if(Yhat is None):
            lr = LR(regularization=self.regularization)
            lr.fit(X, Y)

            self.Yhat = lr.transform(X)

        else:
            self.Yhat = Yhat

        if(len(Y.shape) == 1):
            self.Yhat = self.Yhat.reshape(-1, 1)

        K_pca = np.matmul(X, X.T)
        K_lr = np.matmul(self.Yhat, self.Yhat.T)

        self.K = (self.alpha * K_pca) + (1.0 - self.alpha) * K_lr

    def fit_feature_space(self, X, Y, Yhat=None):

        self.compute_K(X, Y, Yhat=Yhat)

        # Compute the inverse square root of the covariance of X
        C = np.matmul(X.T, X)
        v_C, U_C = sorted_eig(C, thresh=self.regularization)
        U_C = U_C[:, v_C > self.regularization]
        v_C = v_C[v_C > self.regularization]

        Csqrt = np.matmul(np.matmul(U_C, np.diagflat(np.sqrt(v_C))), U_C.T)
        iCsqrt = np.matmul(
            np.matmul(U_C, np.diagflat(np.sqrt(eig_inv(v_C)))), U_C.T)

        Ct = np.matmul(iCsqrt, X.T)
        Ct = np.matmul(np.matmul(Ct, self.K), Ct.T)

        v_Ct, U_Ct = sorted_eig(Ct, thresh=self.regularization, n=self.n_PC)

        v_inv = eig_inv(v_Ct[:self.n_PC])

        PXV = np.matmul(iCsqrt, U_Ct[:, :self.n_PC])

        self.PXT = np.matmul(PXV,
                             np.diagflat(np.sqrt(v_Ct[:self.n_PC])))
        self.PTX = np.matmul(np.diagflat(np.sqrt(v_inv)),
                             np.matmul(U_Ct[:, :self.n_PC].T, Csqrt))
        PTY = np.matmul(np.diagflat(np.sqrt(v_inv)),
                        U_Ct[:, :self.n_PC].T)
        PTY = np.matmul(PTY, iCsqrt)
        self.PTY = np.matmul(np.matmul(PTY, X.T), Y)

    def fit_structure_space(self, X, Y, Yhat=None):

        self.compute_K(X, Y, Yhat=Yhat)
        self.v, self.U = sorted_eig(
            self.K, thresh=self.regularization, n=self.n_PC)

        v_inv = eig_inv(self.v[:self.n_PC])
        self.T = np.matmul(self.U[:, :self.n_PC],
                           np.diagflat(np.sqrt(self.v[:self.n_PC])))

        P_lr = np.matmul(X.T, X) + np.eye(X.shape[1]) * self.regularization
        P_lr = np.linalg.pinv(P_lr)
        P_lr = np.matmul(P_lr, X.T)
        P_lr = np.matmul(P_lr, Y)

        if(len(Y.shape) == 1):
            P_lr = P_lr.reshape((-1, 1))

        P_lr = np.matmul(P_lr, self.Yhat.T)

        P_pca = X.T

        P = (self.alpha * P_pca) + (1.0 - self.alpha) * P_lr
        self.PXT = np.matmul(P, np.matmul(self.U[:, :self.n_PC],
                                          np.diag(np.sqrt(v_inv))))
        self.PTY = np.matmul(np.diagflat(v_inv),
                             np.matmul(self.T.T, Y))

        self.PTX = np.matmul(np.diagflat(v_inv),
                             np.matmul(self.T.T, X))

    def fit(self, X, Y, Yhat=None):

        sample_heavy = X.shape[0] > X.shape[1]
        if((self.space == 'feature' or sample_heavy) and self.space != 'structure'):
            if(X.shape[0] > X.shape[1] and self.space != 'feature'):
                print("# samples > # features, computing in feature space")
            self.fit_feature_space(X, Y, Yhat=Yhat)
        elif(self.space == 'structure' or not sample_heavy):
            if(sample_heavy and self.space != 'structure'):
                print("# samples < # features, computing in structure space")
            self.fit_structure_space(X, Y, Yhat=Yhat)
        else:
            raise Exception('Space Error: \
                              Please specify either space = "structure",\
                              "feature", or "auto" to designate the space to \
                              compute in.\
                            ')

    def transform(self, X):
        if self.PXT is None or self.PTY is None:
            print("Error: must fit the PCovR model before transforming")
        else:
            # self.Undo whitening of PCA scores and scale by norm
            # Xp = np.linalg.norm(X)*

            T = np.matmul(X, self.PXT)

            Yp = np.matmul(np.matmul(X, self.PXT), self.PTY)

            Xr = np.matmul(T, self.PTX)

            return T, Yp, Xr

    def loss(self, X, Y):
        T, Yp, Xr = self.transform(X)

        Lpca = np.linalg.norm(X - Xr) ** 2 / np.linalg.norm(X) ** 2
        Llr = np.linalg.norm(Y - Yp) ** 2 / np.linalg.norm(Y) ** 2

        return Lpca, Llr

    def statistics(self, X, Y):
        T, Yp, Xr = self.transform(X)

        return get_stats(x=X, y=Y, yp=Yp, t=T, xr=Xr)
