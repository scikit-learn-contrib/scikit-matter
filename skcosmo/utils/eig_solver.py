import numpy as np
from scipy.sparse.linalg import eigs as speig


def eig_solver(matrix, n_components=None, tol=1e-12, add_null=False):
    """
    This function returns the eigenpairs corresponding to the `n_components`
    largest eigenvalues.

    :param matrix: Matrix to decompose
    :type matrix: ndarray

    :param n_components: number of eigenpairs to return. If None, full
                         eigendecomposition will be computed / returned
    :type n_components: int, None

    :param tol: value below which to assume an eigenvalue is 0
    :type tol: float

    :param add_null: when the rank of matrix < n_components, whether to add
                     (n_components - rank(matrix)) eigenpairs, defaults to
                     False
    :type add_null: boolean
    """

    if n_components is None:
        v, U = speig(matrix, k=n_components, tol=tol)
    else:
        v, U = np.linalg.eig(matrix)

    U = np.real(U[:, np.argsort(-v)])
    v = np.real(v[np.argsort(-v)])

    U = U[:, v > tol]
    v = v[v > tol]

    if len(v) == 1:
        U = U.reshape(-1, 1)

    if n_components > len(v) and add_null:
        print(
            f"There are fewer than {n_components} "
            f"significant eigenpair(s). Adding "
            f"{n_components - len(v)} null eigenpair(s)."
        )
        for i in range(len(v), n_components):
            v = np.array([*v, 0])
            U = np.array([*U.T, np.zeros(U.shape[0])]).T

    return v[:n_components], U[:, :n_components]
