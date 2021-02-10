import numpy as np

from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd

from ._greedy import GreedySelector
from ..utils.orthogonalizers import feature_orthogonalizer


class SimpleCUR(GreedySelector):
    """Transformer that performs Greedy Feature Selection using by choosing features
    which maximize the magnitude of the right singular vectors, consistent with
    classic CUR matrix decomposition.


    Parameters
    ----------

    n_features_to_select : int or float, default=None
        The number of features to select. If `None`, half of the features are
        selected. If integer, the parameter is the absolute number of features
        to select. If float between 0 and 1, it is the fraction of features to
        select.

    iterative : boolean
                whether to orthogonalize after each selection, defaults to `true`

    k : int
        number of eigenvectors to compute the importance score with, defaults to 1

    tol: float
         threshold below which values will be considered 0, defaults to 1E-12

    iterated_power : int or 'auto', default='auto'
         Number of iterations for the power method computed by
         svd_solver == 'randomized'.
         Must be of range [0, infinity).

    random_state : int, RandomState instance or None, default=None
         Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------

    n_features_to_select_ : int
        The number of features that were selected.

    selected_: ndarray of shape (n_features_to_select), dtype=int
               indices of the selected features

    support_ : ndarray of shape (n_features,), dtype=bool
        The mask of selected features.

    """

    def __init__(
        self,
        n_features_to_select=None,
        iterative=True,
        k=1,
        tol=1e-12,
        iterated_power="auto",
        random_state=None,
    ):

        scoring = self.score
        self.selected_ = []
        self.k = k
        self.tol = tol
        self.iterative = iterative
        self.iterated_power = iterated_power
        self.random_state = random_state

        super().__init__(scoring=scoring, n_features_to_select=n_features_to_select)

    def fit(self, X, y=None, initial=[]):
        """Learn the features to select.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
        """

        # while the super class checks this, because we are doing some pre-processing
        # before calling super().fit(), we should check as well
        tags = self._get_tags()

        if y is not None:
            X, y = self._validate_data(
                X,
                y,
                accept_sparse="csc",
                ensure_min_features=2,
                force_all_finite=not tags.get("allow_nan", True),
                multi_output=True,
            )
        else:
            X = check_array(
                X,
                accept_sparse="csc",
                ensure_min_features=2,
                force_all_finite=not tags.get("allow_nan", True),
            )

        n_features = X.shape[1]
        self.X_current = X.copy()
        self.selected_ = np.array(initial).flatten()
        self.support_ = np.zeros(shape=n_features, dtype=bool)
        self.support_[initial] = True
        if not all(self.support_):
            self.pi_ = self._recompute_pi()

        super().fit(X, y, current_mask=self.support_)

    def score(self, X, y, candidate_feature_idx):
        return self.pi_[candidate_feature_idx]

    def _recompute_pi(self):
        """
        For feature selection, the importance score :math:`\\pi` is the sum over
        the squares of the first :math:`k` components of the right singular vectors

        .. math::

            \\pi_j =
            \\sum_i^k \\left(\\mathbf{U}_\\mathbf{\\tilde{C}}\\right)_{ij}^2.

        where :math:`{\\mathbf{C} = \\mathbf{X}^T\\mathbf{X}.
        """

        new_pi = np.zeros(self.support_.shape[0])
        X_for_SVD = self.X_current[:, ~self.support_]

        random_state = check_random_state(self.random_state)

        # sign flipping is done inside
        _, _, Vt = randomized_svd(
            X_for_SVD,
            n_components=self.k,
            n_iter=self.iterated_power,
            flip_sign=True,
            random_state=random_state,
        )
        new_pi[~self.support_] = (np.real(Vt) ** 2.0).sum(axis=0)
        return new_pi

    def _postprocess(self):
        if self.iterative:
            self.X_current, _ = feature_orthogonalizer(
                self.selected_[-1:], self.X_current
            )
            self.pi_ = self._recompute_pi()
