def get_progress_bar():
    """Returns the appropriate version of ``tqdm``, as determined by ``tqdm.auto``.

    If ``tqdm`` is not installed, an :py:class`ImportError` is raised.
    """
    try:
        from tqdm.auto import tqdm

        return tqdm
    except ImportError:
        raise ImportError(
            "tqdm must be installed to use a progress bar. Either install tqdm or "
            "re-run with progress_bar = False"
        )


def no_progress_bar(x):
    """Identity function, same as ``lambda x:x``. It returns ``x``."""
    return x
