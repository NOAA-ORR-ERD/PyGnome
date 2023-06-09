"""
tests for the function in the utilities/__init__.py file
"""

import pytest
import numpy as np

from gnome.utilities import convert_longitude, round_sf_scalar, round_sf_array

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


# the orginial codde came with tests, but a few here to make sure
# its working as expected for our use case


test_vals = [
    (1.234567891234567, 4, 1.235),
    (1.234567891234567, 6, 1.23457),
    (-1.234567891234567, 4, -1.235),
    (1.234567891234567e40, 4, 1.235e40),
    (-1.234567891234567e-40, 6, -1.23457e-40),
    (3.3456789e-20, 4, 3.346e-20 ),
    (0.0, 4, 0.0),
    (np.nan, 4, np.nan),
    (np.inf, 4, np.inf),
    (-np.inf, 4, -np.inf),
]

@pytest.mark.parametrize("test_input, sigfigs, expected", test_vals)
def test_round_sf_float(test_input, sigfigs, expected):
    result = round_sf_scalar(test_input, sigfigs=sigfigs)

    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == expected


def test_non_int_sigfigs():
    with pytest.raises(TypeError):
        round_sf_scalar(1.23, sigfigs=1.2)


def test_round_sf_array():
    # only the ones with 4 sigfigs
    tv = (t for t in test_vals if t[1] == 4)
    test_input, sigfigs, expected = zip(*tv)

    test_input = np.array(test_input, dtype=np.float64)
    expected = np.array(expected, dtype=np.float64)

    result = round_sf_array(test_input, sigfigs=4)

    print(f"{expected=}")
    print(f"{result=}")
    # print(f"{(expected - result)=}")
    nans = np.isnan(expected)
    assert np.allclose(result[~nans], expected[~nans], rtol=1e-15, atol=0.0)
    # assert np.all(result[~nans] == expected[~nans])
    assert np.alltrue(np.isnan(result[nans]))




