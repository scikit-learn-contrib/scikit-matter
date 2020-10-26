import numpy as np

def feature_orthogonalizer(idx, A_proxy, Y_proxy, tol=1E-12):
    if A_proxy is not None:
        Aci = A_proxy[:, idx]

        if Y_proxy is not None:
            v = np.linalg.pinv(np.matmul(Aci.T, Aci), rcond=tol)
            v = np.matmul(Aci, v)
            v = np.matmul(v, Aci.T)

            Y_proxy -= np.matmul(v, Y_proxy)

        v = A_proxy[:, idx[-1]] / np.sqrt(
            np.matmul(A_proxy[:, idx[-1]], A_proxy[:, idx[-1]]))

        for i in range(A_proxy.shape[1]):
            A_proxy[:, i] -= v * np.dot(v, A_proxy[:, i])
    return A_proxy, Y_proxy

def sample_orthogonalizer(idx, A_proxy, Y_proxy, tol=1E-12):
    if A_proxy is not None:
        if Y_proxy is not None:
            Y_proxy -= A_proxy @ (np.linalg.pinv(
                A_proxy[idx].T @ A_proxy[idx],
                rcond=tol) @ A_proxy[idx].T) @ Y_proxy[idx]

        Ajnorm = np.dot(A_proxy[idx[-1]],
                        A_proxy[idx[-1]])
        for i in range(A_proxy.shape[0]):
            A_proxy[i] -= (np.dot(A_proxy[i], A_proxy[idx[-1]]
                                         ) / Ajnorm) * A_proxy[idx[-1]]
    return A_proxy, Y_proxy
