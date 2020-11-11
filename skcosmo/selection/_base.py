from abc import abstractmethod


class _BaseSelection:
    """
    Super-class defined for selection methods

    Parameters
    ----------
    matrix : ndarray of shape (n x m)
        Data to select from -
        Feature selection will choose a subset of the `m` columns
        Samples selection will choose a subset of the `n` rows
        stored in `self.A`
    mixing : float
        mixing parameter, as described in PCovR as `alpha`
        stored in `self.mixing`
    tolerance : float
        Threshold below which values will be considered 0
        stored in `self.tol`
    precompute : int
        Number of selections to precompute
    progress_bar : bool, callable
        Option to include tqdm progress bar or a callable progress bar
        implemented in `self.progress(iterable)`
    Y (optional) : ndarray of shape (n x p)
        Array to include in biased selection when mixing < 1
        Required when mixing < 1, throws AssertionError otherwise
        stored in `self.Y`

    Attributes
    ----------
    A : ndarray of shape (n x m)
        corresponds to `matrix` passed in constructor
    mixing : float
    tol : float
        corresponds to `tolerance` passed in constructor
    Y (optional) : ndarray of shape (n x p)
    idx : list
        contains the indices of the feature or sample selections made

    #"""

    def __init__(
        self,
        matrix,
        mixing=1.0,
        tolerance=1e-12,
        precompute=None,
        progress_bar=False,
        **kwargs
    ):

        self.mixing = mixing
        self.A = matrix.copy()
        self.tol = tolerance
        self.progress_bar = progress_bar

        if mixing < 1:
            try:
                assert "Y" in kwargs
                self.Y = kwargs.get("Y")
            except AssertionError:
                print(r"For $\mixing < 1$, $Y$ must be in the constructor parameters")
        else:
            self.Y = None

        self.idx = []

        if precompute is not None:
            self.idx = self.select(precompute)

    def progress(self, iterable):
        """Option to show the progress of an iterable"""
        if callable(self.progress_bar):
            return self.progress_bar(iterable)
        if self.progress_bar:
            try:
                from tqdm import tqdm

                return tqdm(iterable)
            except ModuleNotFoundError:
                return iterable

        return iterable

    @abstractmethod
    def select(self, n, **kwargs):
        """Abstract method for select, to be implemented in subclasses

        Parameters
        ----------
        n : number of selections to make

        Returns
        -------
        idx: list of n selections
        """
        return
