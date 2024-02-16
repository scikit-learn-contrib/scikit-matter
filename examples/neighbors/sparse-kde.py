#!/usr/bin/env python
# coding: utf-8

"""
Sparse KDE examples
====================================

We start by importing our modules
"""
# %%
#

import numpy as np

from skmatter.datasets import load_roy_dataset  # TODO add a dataset


# %%
#
# After importing we start with the actual calculations and examples.

roy_data = load_roy_dataset()

a = np.array(2)
print(a)
