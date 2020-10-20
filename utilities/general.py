import numpy as np
from sklearn.metrics import r2_score as calc_R2

from .kernels import gaussian_kernel, center_kernel


def eig_inv(v, rcond=1e-14):
    """ Inverse of a list (typically of eigenvalues) with thresholding à la pinv """
    thresh = v.max() * rcond
    return np.array([(1/vv if vv>thresh else 0.0) for vv in v])


def normalize_matrix(A, scale=None):
    """ Normalize a matrix so that its entries have unit variance """
    if scale is None:
        scale = np.linalg.norm(A) / np.sqrt(len(A))
    return A / scale


def center_matrix(A, center=None):
    """ Removes the mean to "center" a feature matrix """
    if (center is None):
        center = np.mean(A, axis=0)
    return A - center


def FPS(X, n=0, idx=None):
    """
        Does Farthest Point Selection on a set of points X
        Adapted from a routine by Michele Ceriotti
    """
    N = X.shape[0]

    # If desired number of points less than or equal to zero,
    # select all points
    if n <= 0:
        n = N

    # Initialize arrays to store distances and indices
    fps_idxs = np.zeros(n, dtype=np.int)
    d = np.zeros(n)

    if idx is None:
        # Pick first point at random
        idx = np.random.randint(0, N)
    fps_idxs[0] = idx

    # Compute distance from all points to the first point
    d1 = np.linalg.norm(X - X[idx], axis=1)**2

    # Loop over the remaining points...
    for i in range(1, n):

        # Get maximum distance and corresponding point
        fps_idxs[i] = np.argmax(d1)
        d[i - 1] = np.amax(d1)

        # Compute distance from all points to the selected point
        d2 = np.linalg.norm(X - X[fps_idxs[i]], axis=1)**2

        # Set distances to minimum among the last two selected points
        d1 = np.minimum(d1, d2)

        if np.abs(d1).max() == 0.0:
            print("Only {} FPS Possible".format(i))
            return fps_idxs[:i], d[:i]

    return fps_idxs, d


def quick_inverse(mat):
    """ Does a quick(er) matrix inverse """
    U_mat, v_mat = sorted_eig(mat, thresh=0)

    return np.matmul(np.matmul(U_mat, np.diagflat(eig_inv(v_mat))), U_mat.T)


def sorted_eig(mat, thresh=0.0, n=None, sps=True):
    """
        Returns n eigenvalues and vectors sorted
        from largest to smallest eigenvalue
    """

    if(sps):
        from scipy.sparse.linalg import eigs as speig
        if(n is None):
            k = mat.shape[0] - 1
        else:
            k = n
        val, vec = speig(mat, k=k, tol=thresh)
        val = np.real(val)
        vec = np.real(vec)

        idx = sorted(range(len(val)), key=lambda i: -val[i])

        val = val[idx]
        vec = vec[:, idx]

    else:
        val, vec = np.linalg.eigh(mat)
        val = np.flip(val, axis=0)
        vec = np.flip(vec, axis=1)

    if(n is not None and len(val[val<thresh]) < n):
        vec[:, val < thresh] = 0
        val[val < thresh] = 0
    else:
        vec = vec[:, val>=thresh]
        val = val[val>=thresh]


    return val[:n], vec[:, :n]


def get_stats(y=None, yp=None, x=None, t=None, xr=None, k=None, kapprox=None, **kwargs):
    """Returns available statistics given provided data"""
    stats = {}
    if y is not None and yp is not None:
        stats["Coefficient of Determination<br>($R^2$)"] = calc_R2(y, yp)
        stats[r"$\ell_{regr}$"] = np.linalg.norm(y - yp) / y.shape[0]
    if x is not None and t is not None:
        stats[r"Dataset Variance<br>$\sigma_X^2$"] = x.var(axis=0).sum()
        stats[r"Projection Variance<br>$\sigma_T^2$"] = t.var(axis=0).sum()
        error = x.var(axis=0).sum() - t.var(axis=0).sum()
        stats[r"Residual Variance<br>$\sigma_X^2 - \sigma_T^2$"] = error
    if x is not None and xr is not None:
        stats[r"$\ell_{proj}$"] = ((x - xr)**2).mean(axis=0).sum()
    if k is not None and kapprox is not None:
        error = np.linalg.norm(kapprox - k)**2 / np.linalg.norm(k)**2.0
        stats[r"$\ell_{gram}$"] = error

    # allow for manual input of statistics (for kpcovr error)
    for k in kwargs:
        stats[k] = kwargs[k]

    return stats


def split_data(N, n_train=0):
    """Returns indices for the training and tests data sets"""
    global n_test
    # Splits in train and test sets
    if n_train <= 0:  # defaults 50-50 split
        n_train = int(N / 2)
    n_test = N - n_train
    r_train = np.asarray(range(N))
    np.random.shuffle(r_train)
    i_test = list(sorted(r_train[n_train:]))
    i_train = list(sorted(r_train[:n_train]))

    return i_test, i_train


def load_variables(cache_file="../datasets/precomputed.npz", **kwargs):
    """Loads the cache holding the soap vectors for CSD"""
    return calculate_variables(**dict(np.load(cache_file)), **kwargs)


def calculate_variables(
        X,
        Y,
        indices,
        n_atoms,
        N=10,
        n_FPS=200,
        kernel_func=gaussian_kernel,
        i_train=None,
        i_test=None,
        **kwargs):
    """Loads necessary data for the tutorials"""

    print(len(indices), "frames in total.")
    print("Shape of Input Data is ", X.shape, ".")

    if n_FPS is not None:
        fps_idxs, _ = FPS(X.T, n_FPS)
        print("Taking a subsampling of ", n_FPS, "features")
        X = X[:, fps_idxs]

    try:
        print("Shape of testing data is: ", i_train.shape, ".")
    except:
        print("Splitting Data Set")
        try:
            i_test, i_train = split_data(len(Y), n_train)
        except:
            n_train = int(len(Y) / 2)
            i_test, i_train = split_data(len(Y), n_train)

    n_train = len(i_train)
    n_test = len(i_test)

    Y_train = Y[i_train]
    Y_test = Y[i_test]

    Y_center = Y_train.mean(axis=0)
    Y_scale = np.linalg.norm(Y_train - Y_center, axis=0) / np.sqrt(n_train / Y_train.shape[1])

    Y = center_matrix(Y, center=Y_center)
    Y_train = center_matrix(Y_train, center=Y_center)
    Y_test = center_matrix(Y_test, center=Y_center)
    Y_train = normalize_matrix(Y_train, scale=Y_scale)
    Y_test = normalize_matrix(Y_test, scale=Y_scale)

    if len(Y) == len(indices):
        print("Computing training/testing sets from summed environment-centered soap vectors.")
        frame_starts = [sum(nat[:i]) for i in range(len(n_atoms) + 1)]
        X_split = [
            X[frame_starts[i]:frame_starts[i + 1]]
            for i in range(len(indices))
        ]

        X = np.array([np.mean(xs, axis=0) for xs in X_split])
        X_train = X[i_train]
        X_test = X[i_test]

    else:
        X_split = X.copy()

        X_train = X[i_train]
        X_test = X[i_test]

    X_center = X_train.mean(axis=0)
    X_scale = np.linalg.norm(X_train - X_center) / np.sqrt(n_train)

    X_train = center_matrix(X_train, center=X_center)
    X_test = center_matrix(X_test, center=X_center)
    X = center_matrix(X, center=X_center)

    X_train = normalize_matrix(X_train, scale=X_scale)
    X_test = normalize_matrix(X_test, scale=X_scale)
    X = normalize_matrix(X, scale=X_scale)

    try:
        print("Shape of kernel is: ", K_train.shape, ".")
    except:
        if len(Y) == len(indices):
            print("Computing kernels from summing kernels of environment-centered soap vectors.")

            K_train = kernel_func([X_split[i] for i in i_train], 
                    [X_split[i] for i in i_train])
            K_test = kernel_func([X_split[i] for i in i_test], 
                    [X_split[i] for i in i_train])

        else:

            K_train = kernel_func(X_split[i_train], X_split[i_train])
            K_test = kernel_func(X_split[i_test], X_split[i_train])

    K_test = center_kernel(K_test, reference=K_train)
    K_train = center_kernel(K_train)

    K_scale = np.trace(K_train) / K_train.shape[0]

    K_train = normalize_matrix(K_train, scale=K_scale)
    K_test = normalize_matrix(K_test, scale=K_scale)

    n_train = len(X_train)
    n_test = len(X_test)
    n_PC = 2

    return dict(X=X, Y=Y,
                X_split=X_split,
                X_center=X_center, Y_center=Y_center,
                X_scale=X_scale, Y_scale=Y_scale,
                X_train=X_train, Y_train=Y_train,
                X_test=X_test, Y_test=Y_test,
                K_train=K_train, K_test=K_test,
                i_train=i_train, i_test=i_test,
                n_PC=n_PC, n_train=n_train, n_test=n_test)
