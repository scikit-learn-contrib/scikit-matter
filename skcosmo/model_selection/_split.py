import sklearn.model_selection
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


def train_test_split(*arrays, **options):
    """This is an extended version of the sklearn train test split supporting
    overlapping train and test sets.
    See `sklearn.model_selection.train_test_split (external link)
    <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_ .

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int or RandomState instance, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See
        `random state glossary from sklearn (external link) <https://scikit-learn.org/stable/glossary.html#term-random-state>`_
    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.
    train_test_overlap : bool, default=False
        If True, and train and test set are both not None, the train and test
        set may overlap.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
    """
    train_test_overlap = options.pop("train_test_overlap", False)
    test_size = options.get("test_size", None)
    train_size = options.get("train_size", None)

    if train_test_overlap and train_size is not None and test_size is not None:
        # checks from sklearn
        arrays = indexable(*arrays)
        n_samples = _num_samples(arrays[0])

        if test_size == 1.0 or test_size == n_samples:
            test_sets = arrays
        else:
            options["train_size"] = None
            test_sets = sklearn.model_selection.train_test_split(*arrays, **options)[
                1::2
            ]
            options["train_size"] = train_size

        if train_size == 1.0 or train_size == n_samples:
            train_sets = arrays
        else:
            options["test_size"] = None
            train_sets = sklearn.model_selection.train_test_split(*arrays, **options)[
                ::2
            ]
            options["test_size"] = test_size

        train_test_sets = []
        for i in range(len(train_sets)):
            train_test_sets += [train_sets[i], test_sets[i]]
        return train_test_sets
    else:
        return sklearn.model_selection.train_test_split(*arrays, **options)
