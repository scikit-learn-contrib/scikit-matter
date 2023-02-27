#!/usr/bin/env python3
from setuptools import setup
import re

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', open("skmatter/__init__.py").read()
).group(1)

if __name__ == "__main__":
    setup(version=__version__)
