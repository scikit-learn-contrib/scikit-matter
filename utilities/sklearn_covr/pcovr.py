import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.utils.validation import check_X_y
from scipy.sparse.linalg import eigs

from ._base import _BasePCovR

class PCovR(_BasePCovR):
    """
    Performs PCovR, detecting whether the data set is in Sample or Feature Space

    ----Attributes----
    space: whether to compute in feature or sample space
    n_components: number of latent space components for dimensionality reduction
    regularization: regularization parameter for linear models
    tol: tolerance for small eigenvalues in eigendecompositions
    alpha: (float) mixing parameter between decomposition and regression
    ptx_: projector from latent space to input space
    pty_: projector from latent space to property space
    pxt_: projector from input space to latent space
    Yhat: regressed properties

    ----References----
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015
    """

    def __init__(self, alpha=0.0, n_components=None,
                 regularization=1e-6, tol=1e-12,
                 eig_solver="sparse",
                 space=None, lr_args={}, *args, **kwargs,
                 ):
        super().__init__(alpha=alpha, n_components=n_components,
                         regularization=regularization, tol=tol,
                         *args, **kwargs)
        self.space = space
        self.lr_args = lr_args
        self.eig_solver = eig_solver
        self.n_components = n_components
        self.Yhat = None
        self.W = None

    def fit(self, X, Y, Yhat=None, W=None):
        # as required by the superclass

        X, Y = check_X_y(X, Y, y_numeric=True, multi_output=True)

        if Yhat is None or W is None:
            self._compute_Yhat(X, Y)
        else:
            self.Yhat = Yhat
            self.W = W

        # Sparse eigensolvers will not work when seeking N-1 eigenvalues
        if min(X.shape) <= self.n_components:
            self.eig_solver = 'full'

        self._fit(X, Y)

    def _fit(self, X, Y):
        """
        Method for computing the PCovR projectors
        """
        if self.space is None:
            if X.shape[0] > X.shape[1]:
                self.space = 'feature'
            else:
                self.space = 'structure'

        if self.space == 'feature':
            self._fit_feature_space(X, Y)
        else:
            self._fit_sample_space(X, Y)

        self.mean_ = np.mean(X, axis =0)

    def _compute_Yhat(self, X, Y):
        """
        Method for computing the approximation of Y
        """

        if self.Yhat is None:
            lr = LR(self.lr_args)  # some sort of args
            lr.fit(X, Y)
            self.Yhat = lr.predict(X)
            self.W = lr.coef_.T

        if self.W is None:
            W = np.linalg.pinv(np.dot(X.T, X), rcond=self.regularization)
            W = np.linalg.multi_dot([W, X.T, Y])
            self.W = W

    def _fit_feature_space(self, X, Y):
        """
        In sample-space PCovR, the projectors are determined by:

        C̃ = α X.T X + (1 - α) (X.T X)^(-1/2) X.T Ŷ Ŷ.T X (X.T X)^(-1/2)

        where

        P_XT = (X.T X)^(-1/2) U_C̃.T Λ_C̃^(1/2)
        P_TX = Λ_C̃^(-1/2) U_C̃.T (X.T X)^(1/2)
        P_TY = Λ_C̃^(-1/2) U_C̃.T (X.T X)^(-1/2) X.T Y

        """

        C = np.dot(X.T, X)

        # it is necessary to use the full SVD to decompose C
        v, U = self._eig_solver(C, full_matrix=True)
        S = v ** 0.5
        S_inv = np.linalg.pinv(np.diagflat(S))

        Csqrt = np.linalg.multi_dot([U, np.diagflat(S), U.T])
        iCsqrt = np.linalg.multi_dot([U, S_inv, U.T])

        C_lr = iCsqrt @ X.T @ self.Yhat
        C_lr = C_lr @ C_lr.T

        # note about Ctilde definition
        Ct = self.alpha * C + (1.0 - self.alpha) * C_lr

        v, U = self._eig_solver(Ct)
        S = v ** 0.5

        self.pxt_ = np.linalg.multi_dot([iCsqrt, U, np.diagflat(S)])
        self.ptx_ = np.linalg.multi_dot([S_inv, U.T, Csqrt])
        self.pty_ = np.linalg.multi_dot(
            [S_inv, U.T, iCsqrt, X.T, Y])

    def _fit_sample_space(self, X, Y):
        """
        In sample-space PCovR, the projectors are determined by:

        K̃ = α X X.T + (1 - α) Ŷ Ŷ.T

        where

        P_XT = α X.T + (1 - α) P_XY Ŷ.T
        P_TX = Λ_K̃^(-1/2) U_K̃.T X
        P_TY = Λ_K̃^(-1/2) U_K̃.T Y

        """


        Kt = (self.alpha * np.dot(X, X.T)) + \
             (1.0 - self.alpha) * np.dot(self.Yhat, self.Yhat.T)

        v, U = self._eig_solver(Kt)
        S = v ** 0.5

        T = np.dot(U, np.diagflat(S))

        P = (self.alpha * X.T) + (1.0 - self.alpha) * \
            np.dot(self.W, self.Yhat.T)
        self.pxt_ = np.linalg.multi_dot([P, U, np.diagflat(1/S)])
        self.pty_ = np.linalg.multi_dot([np.diagflat(1/S**2.0), T.T, Y])
        self.ptx_ = np.linalg.multi_dot([np.diagflat(1/S**2.0), T.T, X])

    def _eig_solver(self, matrix, full_matrix=False):
        if(self.eig_solver=='sparse' and full_matrix==False):
            v, U= eigs(matrix, k=self.n_components, tol=self.tol)
        else:
            v, U = np.linalg.eig(matrix)

        U = np.real(U[:, np.argsort(v)])
        v = np.real(v[np.argsort(v)])

        U = U[:, v > self.tol]
        v = v[v > self.tol]

        if(len(v)==1):
            U = U.reshape(-1,1)

        return v, U

    def transform(self, X):
        return super()._project(X, 'pxt_')

    def inverse_transform(self, T):
        return super()._project(T, 'ptx_')

    def predict(self, X):
        # Predict based on X only
        return super()._project(self.transform(X), 'pty_')
