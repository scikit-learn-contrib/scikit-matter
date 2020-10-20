import numpy as np
from scipy.spatial.distance import cdist

def self_linear_kernel(XA):
    """
        Build a dot product kernel between all samples in XA

        ---Arguments---
        XA: vectors of data from which to build the kernel,
            where each row is a sample and each column is a feature.
            If XA is a list, the kernel is averaged over the list

        ---Returns---
        K: dot product kernel matrix between XA samples
    """

    K = np.zeros((len(XA), len(XA)))

    flag_A = isinstance(XA, list)

    # XA structures
    if flag_A:
        for idx_i in range(len(XA)):
            for idx_j in range(idx_i, len(XA)):
                kij = np.matmul(XA[idx_i], XA[idx_j].T)
                K[idx_i, idx_j] = K[idx_j, idx_i] = kij.mean()

    # XA environments
    else:
        K = np.matmul(XA, XA.T)

    return K

def linear_kernel(XA, XB=None):
    """
        Builds a dot product kernel

        ---Arguments---
        XA, XB: vectors of data from which to build the kernel,
            where each row is a sample and each column is a feature.
            If XA or XB is a list, the kernel is averaged over the list

        ---Returns---
        K: dot product kernel between XA (and XB)
    """

    flag_A = isinstance(XA, list)
    flag_B = isinstance(XB, list)

    if XB is None:
        return self_linear_kernel(XA)

    else:
        K = np.zeros((len(XA), len(XB)))

        # XA and XB structures
        if flag_A and flag_B:
            for idx_i in range(len(XA)):
                for idx_j in (range(len(XB))):
                    ki = np.matmul(XA[idx_i], XB[idx_i].T)
                    K[idx_i, idx_j] = ki.mean()

        # XA structures, XB environments
        elif flag_A:
            for idx_i in range(len(XA)):
                ki = np.matmul(XA[idx_i], XB.T)
                K[idx_i, :] = ki.mean(axis=0)

        # XA environments, XB structures
        elif flag_B:
            for idx_j in range(len(XB)):
                kj = np.matmul(XA, XB[idx_j].T)
                K[:, idx_j] = ki.mean(axis=1)

        # XA and XB environments
        else:
            K = np.matmul(XA, XB.T)

        return K


def self_gaussian_kernel(XA, gamma=1.0):
    """
        Build a Gaussian kernel between all samples in XA

        ---Arguments---
        XA: vectors of data from which to build the kernel,
            where each row is a sample and each column is a feature.
            If XA is a list, the kernel is averaged over the list
        gamma: kernel width

        ---Returns---
        K: gaussian kernel matrix between XA samples
    """

    K = np.zeros((len(XA), len(XA)))

    flag_A = isinstance(XA, list)

    # XA structures
    if flag_A:
        for idx_i in range(len(XA)):
            for idx_j in range(idx_i, len(XA)):
                kij = np.exp(-gamma*cdist(XA[idx_i], XA[idx_j], metric="sqeuclidean"))
                K[idx_i, idx_j] = K[idx_j, idx_i] = kij.mean()

    # XA environments
    else:
        K = np.exp(-gamma*cdist(XA, XA, metric="sqeuclidean"))

    return K


def gaussian_kernel(XA, XB=None, gamma=1.0):
    """
    Build a Gaussian kernel between all samples in XA and XB

    ---Arguments---
    XA, XB: vectors of data from which to build the kernel,
        where each row is a sample and each column is a feature.
        If XA or XB is a list, the kernel is averaged over the list
    gamma: kernel width

    ---Returns---
    K: gaussian kernel matrix between XA (and XB)
    """

    flag_A = isinstance(XA, list)
    flag_B = isinstance(XB, list)

    if XB is None:
        return self_gaussian_kernel(XA, gamma=gamma)

    else:
        K = np.zeros((len(XA), len(XB)))

        # XA and XB structures
        if flag_A and flag_B:
            for idx_i in range(len(XA)):
                for idx_j in (range(len(XB))):
                    ki = np.exp(-gamma*cdist(XA[idx_i], XB[idx_j], metric="sqeuclidean"))
                    K[idx_i, idx_j] = ki.mean()

        # XA structures, XB environments
        elif flag_A:
            for idx_i in range(len(XA)):
                ki = np.exp(-gamma*cdist(XA[idx_i], XB, metric="sqeuclidean"))
                K[idx_i, :] = ki.mean(axis=0)

        # XA environments, XB structures
        elif flag_B:
            for idx_j in range(len(XB)):
                kj = np.exp(-gamma*cdist(XA, XB[idx_j], metric="sqeuclidean"))
                K[:, idx_j] = ki.mean(axis=1)

        # XA and XB environments
        else:
            K = np.exp(-gamma*cdist(XA, XB, metric="sqeuclidean"))

        return K


def center_kernel(K, reference=None, ref_cmean=None, ref_rmean=None, ref_mean=None):
    """
        Centers a kernel

        ---Arguments---
        K: kernel to center
        reference: kernel relative to whos RKHS one should center
                   defaults to K

        ---Returns---
        Kc: centered kernel

        ---References---
        1.  https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
        2.  M. Welling, 'Kernel Principal Component Analysis',
            https://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-PCA.pdf
    """

    K_ref = reference
    if K_ref is None:
        K_ref = K
    else:
        if K.shape[1] != K_ref.shape[0]:
            raise ValueError(
                "The kernel to be centered and the reference have inconsistent sizes")
    if K_ref.shape[0] != K_ref.shape[1]:
        raise ValueError(
            "The reference kernel is not square, and does not define a RKHS")
    # oneMN = np.ones((K.shape[0], K.shape[1]))/K.shape[1]
    # oneNN = np.ones((K.shape[1], K.shape[1]))/K.shape[1]
    #
    # Kc = K - np.matmul(oneMN, K_ref) - np.matmul(K, oneNN) + \
    #     np.matmul(np.matmul(oneMN, K_ref), oneNN)

    if(ref_cmean is None):
        ref_cmean = K_ref.mean(axis=0)

    if(ref_rmean is None):
        ref_rmean = K.mean(axis=1)

    if(ref_mean is None):
        ref_mean = K_ref.mean()

    Kc = K - np.broadcast_arrays(K, ref_cmean)[1] \
           - ref_rmean.reshape((K.shape[0], 1)) \
           + np.broadcast_arrays(K, ref_mean)[1]

    return Kc
