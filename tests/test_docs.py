import doctest
import importlib
import os
import unittest

import skmatter


print(doctest.NORMALIZE_WHITESPACE)


class TestDoctests(unittest.TestCase):
    def test_doctests(self):
        for directory, _, filenames in os.walk(skmatter.__path__[0]):
            submodule = os.path.split(directory)[1]
            if submodule == "skmatter":
                submodule = ""
            else:
                submodule = f".{submodule}"

            for file in filenames:
                name, ending = os.path.splitext(file)
                if ending == ".py":
                    module_to_test = importlib.import_module(
                        f"skmatter{submodule}.{name}"
                    )
                    result = doctest.testmod(module_to_test)
                    self.assertEqual(result.failed, 0)


if __name__ == "__main__":
    unittest.main()
