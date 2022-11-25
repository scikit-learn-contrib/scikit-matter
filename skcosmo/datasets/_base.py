from os.path import (
    dirname,
    join,
)

import numpy as np
from sklearn.utils import Bunch


def load_nice_dataset():
    """Load and returns NICE dataset.
    Returns
    -------
    nice_data : sklearn.utils.Bunch
      Dictionary-like object, with the following attributes:
      data : `sklearn.utils.Bunch` --
      contains the keys ``X`` and ``y``.
      Structural NICE features and energies, respectively.
      DESCR: `str` --
        The full description of the dataset.
    """

    module_path = dirname(__file__)
    target_filename = join(module_path, "data", "nice_dataset.npz")
    raw_data = np.load(target_filename)
    data = Bunch(
        X=raw_data["structural_features"],
        y=raw_data["energies"],
    )
    with open(join(module_path, "descr", "nice_dataset.rst")) as rst_file:
        fdescr = rst_file.read()
    return Bunch(data=data, DESCR=fdescr)


def load_degenerate_CH4_manifold():
    """Load and return the degenerate manifold dataset.

    Returns
    -------
    degenerate_CH4_manifold_data : sklearn.utils.Bunch
        Dictionary-like object, with the following attributes:

        data : `sklearn.utils.Bunch` --
        contains the keys ``SOAP_power_spectrum`` and ``SOAP_bispectrum``.
        Two representations of the carbon environments of the
        degenerate manifold dataset.

        DESCR: `str` --
        The full description of the dataset.
    """
    module_path = dirname(__file__)
    target_filename = join(module_path, "data", "degenerate_CH4_manifold.npz")
    raw_data = np.load(target_filename)
    data = Bunch(
        SOAP_power_spectrum=raw_data["SOAP_power_spectrum"],
        SOAP_bispectrum=raw_data["SOAP_bispectrum"],
    )
    with open(join(module_path, "descr", "degenerate_CH4_manifold.rst")) as rst_file:
        fdescr = rst_file.read()

    return Bunch(data=data, DESCR=fdescr)


def load_csd_1000r(return_X_y=False):
    """Load and return the minimal CSD dataset.

    Returns
    -------
    csd1000r : sklearn.utils.Bunch
        Dictionary-like object, with the following attributes:

        data : `sklearn.utils.Bunch` --
        contains the keys ``X`` and ``Y``, corresponding to the
        FPS-reduced SOAP vectors and local NMR chemical shielding, respectively,
        for 100 selected environments of the CSD-1000r dataset.

        DESCR: `str` --
        The full description of the dataset.
    """
    module_path = dirname(__file__)
    target_filename = join(module_path, "data", "csd-1000r.npz")
    raw_data = np.load(target_filename)
    if not return_X_y:
        data = Bunch(
            X=raw_data["X"],
            y=raw_data["Y"],
        )
        with open(join(module_path, "descr", "csd-1000r.rst")) as rst_file:
            fdescr = rst_file.read()

        return Bunch(data=data, DESCR=fdescr)
    else:
        return raw_data["X"], raw_data["Y"]
