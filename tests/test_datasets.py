from unittest import mock

import numpy as np
import pytest

from skmatter.datasets import (
    load_csd_1000r,
    load_degenerate_CH4_manifold,
    load_hbond_dataset,
    load_nice_dataset,
    load_roy_dataset,
    load_who_dataset,
)


@pytest.fixture(scope="module")
def nice_data():
    return load_nice_dataset()


@pytest.fixture(scope="module")
def degenerate_CH4_manifold():
    return load_degenerate_CH4_manifold()


@pytest.fixture(scope="module")
def csd():
    return load_csd_1000r()


@pytest.fixture(scope="module")
def who_data():
    size = 24240
    shape = (2020, 12)
    value = 5.00977993011475
    try:
        import pandas as pd  # NoQa: F401

        has_pandas = True
        who = load_who_dataset()
    except ImportError:
        has_pandas = False
        who = None
    return {
        "who": who,
        "has_pandas": has_pandas,
        "size": size,
        "shape": shape,
        "value": value,
    }


@pytest.fixture(scope="module")
def roy():
    return {"data": load_roy_dataset(), "size": 264, "shape": (264, 32)}


@pytest.fixture(scope="module")
def hbond():
    return {"data": load_hbond_dataset(), "size": 27233, "shape": (27233, 3)}


def test_load_nice_data(nice_data):
    # test if representations and properties have commensurate shape
    assert nice_data.data.X.shape[0] == nice_data.data.y.shape[0]
    assert nice_data.data.X.shape[0] == 500
    assert nice_data.data.X.shape[1] == 160
    assert len(nice_data.data.X.shape) == 2


def test_load_nice_data_descr(nice_data):
    nice_data.DESCR


def test_load_degenerate_CH4_manifold_power_spectrum_shape(degenerate_CH4_manifold):
    # test if representations have correct shape
    assert degenerate_CH4_manifold.data.SOAP_power_spectrum.shape == (162, 12)


def test_load_degenerate_CH4_manifold_bispectrum_shape(degenerate_CH4_manifold):
    assert degenerate_CH4_manifold.data.SOAP_bispectrum.shape == (162, 12)


def test_load_degenerate_CH4_manifold_access_descr(degenerate_CH4_manifold):
    degenerate_CH4_manifold.DESCR


def test_load_csd_1000r_shape(csd):
    # test if representations and properties have commensurate shape
    assert csd.data.X.shape[0] == csd.data.y.shape[0]


def test_load_csd_1000r_access_descr(csd):
    csd.DESCR


def test_load_dataset_without_pandas():
    """Check if the correct exception occurs when pandas isn't present."""
    with mock.patch.dict("sys.modules", {"pandas": None}):
        with pytest.raises(ImportError) as cm:
            _ = load_who_dataset()
        assert str(cm.value) == "load_who_dataset requires pandas."


def test_dataset_size_and_shape(who_data):
    """
    Check if the correct number of datapoints are present in the dataset.
    Also check if the size of the dataset is correct.
    """
    if who_data["has_pandas"]:
        assert who_data["who"]["data"].size == who_data["size"]
        assert who_data["who"]["data"].shape == who_data["shape"]


def test_datapoint_value(who_data):
    """Check if the value of a datapoint at a certain location is correct."""
    if who_data["has_pandas"]:
        assert np.allclose(
            who_data["who"]["data"]["SE.XPD.TOTL.GD.ZS"][1924],
            who_data["value"],
            rtol=1e-6,
        )


def test_roy_dataset_content(roy):
    """Check if the correct number of datapoints are present in the dataset.

    Also check if the size of the dataset is correct.
    """
    assert len(roy["data"]["structure_types"]) == roy["size"]
    assert roy["data"]["features"].shape == roy["shape"]
    assert len(roy["data"]["energies"]) == roy["size"]


def test_hbond_dataset_size_and_shape(hbond):
    """
    Check if the correct number of datapoints are present in the dataset.
    Also check if the size of the dataset is correct.
    """
    assert hbond["data"]["descriptors"].shape == hbond["shape"]
    assert hbond["data"]["weights"].size == hbond["size"]
