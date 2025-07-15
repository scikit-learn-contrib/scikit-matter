from os.path import dirname, join

import numpy as np
import sklearn


if sklearn.__version__ >= "1.5.0":
    from sklearn.utils._optional_dependencies import check_pandas_support
else:
    from sklearn.utils import check_pandas_support

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


def load_who_dataset():
    """Load and returns WHO dataset.

    Returns
    -------
    who_dataset : sklearn.utils.Bunch
      Dictionary-like object, with the following attributes:
          data : `pandas.core.frame.DataFrame` -- the WHO dataset
                  as a Pandas dataframe.
          DESCR: `str` -- The full description of the dataset.
    """
    module_path = dirname(__file__)
    target_filename = join(module_path, "data", "who_dataset.csv")
    pd = check_pandas_support("load_who_dataset")
    raw_data = pd.read_csv(target_filename)
    with open(join(module_path, "descr", "who_dataset.rst")) as rst_file:
        fdescr = rst_file.read()
    return Bunch(data=raw_data, DESCR=fdescr)


def load_roy_dataset():
    """Load and returns the ROY dataset, which contains densities,
    energies and SOAP-derived descriptors for 264 structures of polymorphs of ROY,
    from [Beran et Al, Chemical Science (2022)](https://doi.org/10.1039/D1SC06074K)
    Each structure is labeled as "Known" or "Unknown".

    Returns
    -------
    roy_dataset : sklearn.utils.Bunch
      Dictionary-like object, with the following attributes:
          densities : `np.array` -- the densities of the structures
          structure_types : `np.array` -- the type of the structures
          features : `np.array` -- SOAP-derived descriptors for the structures
          energies : `np.array` -- energies of the structures
    """
    module_path = dirname(__file__)
    target_properties = join(module_path, "data", "beran_roy_properties.npz")
    properties = np.load(target_properties)

    return Bunch(
        densities=properties["densities"],
        energies=properties["energies"],
        structure_types=properties["structure_types"],
        features=properties["feats"],
    )


def load_hbond_dataset():
    """Load and returns the hydrogen bond dataset, which contains
    a set of 3D descriptors for 27233 hydrogen bonds and corresponding
    weights, from [Gasparotto et Al, The Journal of Chemical Physics]
    (https://doi.org/10.1063/1.4900655)

    Returns
    -------
    hbond_dataset : sklearn.utils.Bunch
      Dictionary-like object, with the following attributes:
          descriptors : `numpy.ndarray` -- the descriptors of hydrogen bond dataset
          weights : `numpy.ndarray` -- the weights of each sample in the dataset
    """
    module_path = dirname(__file__)
    target_filename = join(module_path, "data", "h2o-blyp-piglet.npz")
    raw_data = np.load(target_filename)

    with open(join(module_path, "descr", "h2o-blyp-piglet.rst")) as rst_file:
        fdescr = rst_file.read()

    return Bunch(
        descriptors=raw_data["descriptors"],
        weights=raw_data["weights"],
        DESCR=fdescr,
    )
