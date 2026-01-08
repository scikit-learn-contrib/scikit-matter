import pytest

from skmatter.utils import get_progress_bar


def test_no_tqdm():
    """Check that the model cannot use a progress bar when tqdm is not installed."""
    import sys

    sys.modules["tqdm"] = None

    with pytest.raises(ImportError) as cm:
        _ = get_progress_bar()
    assert str(cm.value) == (
        "tqdm must be installed to use a progress bar. Either install tqdm or "
        "re-run with progress_bar = False"
    )
