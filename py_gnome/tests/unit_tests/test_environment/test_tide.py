'''
Test all operations for cats mover work
'''

import os
from pathlib import Path

import pytest
from pytest import raises

from gnome.environment.tide import SHIO_YEARDATA_LIMITS
from gnome.environment import Tide
from gnome.utilities.remote_data import get_datafile

from ..conftest import testdata


shio_file = testdata['timeseries']['tide_shio']
ossm_file = testdata['timeseries']['tide_ossm']
#ossm_new_hdr_file = testdata['timeseries']['tide_ossm_new_hdr']
ossm_new_hdr_file = Path(__file__).parent / "sample_data" / "shio_ossm_new_hdr.txt"


def test_shio_data_limits():
    """
    make sure we can get the SHIO year data limits

    This is determined by the yeardata in the gnome/data/yeardata folder

    6/3/2026 : added the yeardata through 2045
    """
    print(SHIO_YEARDATA_LIMITS)

    assert SHIO_YEARDATA_LIMITS == (1980, 2045)


def test_exceptions():
    """
    Test correct exceptions are raised
    """
    bad_file = 'CLISShio.txtX'
    bad_yeardata_path = os.path.join('Data', 'yeardata')

    with raises(IOError):
        Tide(bad_file)

    with raises(IOError):
        Tide(shio_file, yeardata=bad_yeardata_path)


@pytest.mark.parametrize('filename', [shio_file, ossm_file, ossm_new_hdr_file])
def test_file(filename):
    """
    (WIP) simply tests that the file loads correctly
    """
    td = Tide(filename)
    assert td.filename == filename


@pytest.mark.parametrize('filename',
                         [shio_file, ossm_file, ossm_new_hdr_file])
def test_serialize_deserialize(filename):
    '''
        create - it creates new object after serializing original object
                 and tests equality of the two

        update - tests serialize/deserialize and update_from_dict methods
                 don't fail.  It doesn't update any properties.
    '''
    tide = Tide(filename)
    serial = tide.serialize()
    new_t = Tide.deserialize(serial)
    assert new_t is not tide
    assert new_t == tide
