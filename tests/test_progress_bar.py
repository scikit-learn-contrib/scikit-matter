import unittest

from skcosmo.utils import get_progress_bar


class PBarTest(unittest.TestCase):
    def test_no_tqdm(self):
        """
        This test checks that the model cannot use a progress bar when tqdm
        is not installed
        """
        import sys

        sys.modules["tqdm"] = None

        with self.assertRaises(ImportError) as cm:
            _ = get_progress_bar()
            self.assertEqual(
                str(cm.exception),
                "tqdm must be installed to use a progress bar."
                "Either install tqdm or re-run with"
                "progress_bar = False",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
