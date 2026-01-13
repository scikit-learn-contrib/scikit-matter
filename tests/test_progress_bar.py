import pytest

from skmatter.utils import get_progress_bar


def test_no_tqdm():
    """Check that the model cannot use a progress bar when tqdm is not installed."""
    import sys

    sys.modules["tqdm"] = None

    match = (
        "tqdm must be installed to use a progress bar. Either install tqdm or "
        "re-run with progress_bar = False"
    )
    with pytest.raises(ImportError, match=match):
        get_progress_bar()
