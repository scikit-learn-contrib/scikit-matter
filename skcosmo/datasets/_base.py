from os.path import join, dirname
import numpy as np
from sklearn.utils import Bunch


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
