import unittest
import skcosmo


class MinimalTestExample(unittest.TestCase):
    def test_version(self):
        self.assertEqual(skcosmo.__version__, "0.1.0")


if __name__ == '__main__':
    unittest.main()
