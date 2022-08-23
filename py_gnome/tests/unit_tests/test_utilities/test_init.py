"""
tests for the function in the utilities/__init__.py file
"""

import pytest
import numpy as np

from gnome.utilities import convert_longitude

L_180_180 = [-180, -179.99999, -90, 0, 90, 179.9999, 180]

L_0_360 = [0, 90, 180, 270, 359.9999, 360]


def test_convert_longitude_bad_input():
    with pytest.raises(TypeError):
        convert_longitude(180_180, 'typo')


def test_convert_longitude_360():
    lon = convert_longitude(L_180_180, "0--360")

    assert not np.any(lon >= 360.0)
    assert not np.any(lon < 0.0)


def test_convert_longitude_180():
    lon = convert_longitude(L_0_360)

    print(lon)
    assert not np.any(lon > 180)
    assert not np.any(lon <= -180)

def test_normalization_180():
    lon = convert_longitude(L_180_180)

    print(L_180_180)
    print(lon)

    # only the zeroth element should have changed
    assert np.all(np.array(L_180_180[1:] == lon[1:]))
    assert not np.any(lon > 180)
    assert not np.any(lon <= -180)


def test_normalization_360():
    lon = convert_longitude(L_0_360, "0--360")

    print(L_0_360)
    print(lon)

    # only the last element should have changed
    assert np.all(np.array(L_0_360[:-1] == lon[:-1]))
    assert not np.any(lon < 0)
    assert not np.any(lon >= 360)


def test_round_trip360():
    """
    round tripping should result in the same normalized form
    """
    norm = convert_longitude(L_0_360, "0--360")
    lon = convert_longitude(convert_longitude(L_0_360, '-180--180'), "0--360")

    assert np.all(lon == norm)

def test_round_trip360():
    """
    round tripping should result in the same normalized form
    """
    norm = convert_longitude(L_180_180, '-180--180')
    lon = convert_longitude(convert_longitude(L_180_180, "0--360"), "-180--180")

    print(L_180_180)
    print(norm)
    print(lon)
    print(lon == norm)

    assert np.allclose(lon, norm, rtol=1e-15, atol=0)

