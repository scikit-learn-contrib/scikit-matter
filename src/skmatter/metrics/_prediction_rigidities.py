import numpy as np


def local_prediction_rigidity(X_train, X_test, alpha):
    r"""Computes the local prediction rigidity (LPR) of a linear or kernel model
    trained on a training dataset provided as input, on the local environments in the
    test set provided as a separate input. LPR is defined as follows:

    .. math::
        LPR_{i} = \frac{1}{X_i (X^{T} X + \lambda I)^{-1} X_i^{T}}

    The function assumes that the model training is undertaken in a manner where the
    global prediction targets are averaged over the number of atoms appearing in each
    training structure, and the average feature vector of each structure is hence used
    in the regression. This ensures that (1) regularization strength across structures
    with different number of atoms is kept constant per structure during model training,
    and (2) range of resulting LPR values are loosely kept between 0 and 1 for the ease
    of interpretation. This requires the user to provide the regularizer value that
    results from such training procedure. To guarantee valid comparison in the LPR
    across different models, feature vectors are scaled by a global factor based on
    standard deviation across atomic envs.

    If the model is a kernel model, K_train and K_test can be provided in lieu of
    ``X_train`` and ``X_test``, alnog with the appropriate regularizer for the trained
    model.

    Parameters
    ----------
    X_train : list of numpy.ndarray of shape (n_atoms, n_features)
        Training dataset where each training set structure is stored as a
        separate ndarray.
    X_test : list of numpy.ndarray of shape (n_atoms, n_features)
        Test dataset where each training set structure is stored as a separate
        ndarray.
    alpha : float
        Regularizer value that the linear/kernel model has been optimized to.

    Returns
    -------
    LPR : list of numpy.array of shape (n_atoms)
        Local prediction rigidity (LPR) of the test set structures. LPR is
        separately stored for each test structure, and hence list length =
        n_test_strucs.
    rank_diff : int
        integer value of the difference between cov matrix dimension and rank
    """
    # initialize a StandardFlexibleScaler and fit to train set atom envs
    X_atom = np.vstack(X_train)
    sfactor = np.sqrt(np.mean(X_atom**2, axis=0).sum())

    # prep to build covariance matrix XX, take average feat vecs per struc
    X_struc = []
    for X_i in X_train:
        X_struc.append(np.mean(X_i / sfactor, axis=0))
    X_struc = np.vstack(X_struc)

    # build XX and obtain Xinv for LPR calculation
    XX = X_struc.T @ X_struc
    Xprime = XX + alpha * np.eye(XX.shape[0])
    rank_diff = X_struc.shape[1] - np.linalg.matrix_rank(Xprime)
    Xinv = np.linalg.pinv(Xprime)

    # track test set atom indices for output
    lens = []
    for X in X_test:
        lens.append(len(X))
    test_idxs = np.cumsum([0] + lens)

    # prep and compute LPR
    num_test = len(X_test)
    X_test = np.vstack(X_test)
    atom_count = X_test.shape[0]
    LPR_np = np.zeros(X_test.shape[0])

    for ai in range(atom_count):
        Xi = X_test[ai].reshape(1, -1) / sfactor
        LPR_np[ai] = 1 / (Xi @ Xinv @ Xi.T).item()

    # separately store LPR by test struc
    LPR = []
    for i in range(num_test):
        LPR.append(LPR_np[test_idxs[i] : test_idxs[i + 1]])

    return LPR, rank_diff


def componentwise_prediction_rigidity(X_train, X_test, alpha, comp_dims):
    r"""Computes the component-wise prediction rigidity (CPR) and the local CPR
    (LCPR) of a linear or kernel model trained on a training dataset provided as input,
    on the local environments in the test set provided as a separate input. CPR and LCPR
    are defined as follows:

    .. math::
        CPR_{A,c} = \frac{1}{X_{A,c} (X^{T} X + \lambda I)^{-1} X_{A,c}^{T}}

    .. math::
        LCPR_{i,c} = \frac{1}{X_{i,c} (X^{T} X + \lambda I)^{-1} X_{i,c}^{T}}

    The function assumes that the feature vectors for the local environments and
    structures are built by concatenating the descriptors of different prediction
    components together. It also assumes, like the case of LPR, that model training is
    undertaken in a manner where the global prediction targets are averaged over the
    number of atoms appearing in each training structure, and the average feature vector
    of each structure is hence used in the regression. Likewise, to guarantee valid
    comparison in the (L)CPR across different models, feature vectors are scaled by a
    global factor based on standard deviation across atomic envs.

    If the model is a kernel model, K_train and K_test can be provided in lieu of
    X_train and X_test, alnog with the appropriate regularizer for the trained model.
    However, in computing the kernels, one must strictly keep the different components
    separate, and compute separate kernel blocks for different prediction components.

    Parameters
    ----------
    X_train : list of numpy.ndarray of shape (n_atoms, n_features)
        Training dataset where each training set structure is stored as a
        separate ndarray.
    X_test : list of numpy.ndarray of shape (n_atoms, n_features)
        Test dataset where each training set structure is stored as a separate
        ndarray.
    alpha : float
        Regularizer value that the linear/kernel model has been optimized to.
    comp_dims : numpy.ndarray of int values
        Dimensions of the feature vectors pertaining to each prediction
        component.

    Returns
    -------
    CPR : numpy.ndarray of shape (n_test_strucs, n_comps)
        Component-wise prediction rigidity computed for each prediction component,
        pertaining to the entire test structure.
    LCPR : list of ndarrays of shape (n_atoms, n_comps)
        Local component-wise prediction rigidity of the test set structures. Values are
        separately stored for each test structure, and hence list length = n_test_strucs
    rank_diff : int
        value of the difference between cov matrix dimension and rank
    """
    # initialize a StandardFlexibleScaler and fit to train set atom envs
    X_atom = np.vstack(X_train)
    sfactor = np.sqrt(np.mean(X_atom**2, axis=0).sum())

    # prep to build covariance matrix XX, take average feat vecs per struc
    X_struc = []
    for X_i in X_train:
        X_struc.append(np.mean(X_i / sfactor, axis=0))
    X_struc = np.vstack(X_struc)

    # build XX and obtain Xinv for LPR calculation
    XX = X_struc.T @ X_struc
    Xprime = XX + alpha * np.eye(XX.shape[0])
    rank_diff = X_struc.shape[1] - np.linalg.matrix_rank(Xprime)
    Xinv = np.linalg.pinv(Xprime)

    # track test set atom indices for output
    lens = []
    for X in X_test:
        lens.append(len(X))
    test_idxs = np.cumsum([0] + lens)

    # get struc average feat vecs for test set
    X_struc_test = []
    for X_i in X_test:
        X_struc_test.append(np.mean(X_i / sfactor, axis=0))
    X_struc_test = np.vstack(X_struc_test)

    # prep and compute CPR and LCPR
    num_test = len(X_test)
    num_comp = len(comp_dims)
    comp_idxs = np.cumsum([0] + comp_dims.tolist())
    X_test = np.vstack(X_test)
    atom_count = X_test.shape[0]
    CPR = np.zeros((num_test, num_comp))
    LCPR_np = np.zeros((atom_count, num_comp))

    for ci in range(num_comp):
        tot_comp_idx = np.arange(comp_dims.sum())
        mask = (
            (tot_comp_idx >= comp_idxs[ci]) & (tot_comp_idx < comp_idxs[ci + 1])
        ).astype(float)

        for ai in range(atom_count):
            Xic = np.multiply(X_test[ai].reshape(1, -1) / sfactor, mask)
            LCPR_np[ai, ci] = 1 / (Xic @ Xinv @ Xic.T).item()

        for si in range(len(X_struc_test)):
            XAc = np.multiply(X_struc_test[si].reshape(1, -1), mask)
            CPR[si, ci] = 1 / (XAc @ Xinv @ XAc.T).item()

    # separately store LCPR by test struc
    LCPR = []
    for i in range(num_test):
        LCPR.append(LCPR_np[test_idxs[i] : test_idxs[i + 1]])

    return CPR, LCPR, rank_diff
