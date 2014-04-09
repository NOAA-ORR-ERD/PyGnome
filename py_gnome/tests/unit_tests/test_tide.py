'''
Test all operations for cats mover work
'''

import os

import pytest
from pytest import raises

from gnome.environment import Tide
from gnome.utilities.remote_data import get_datafile

here = os.path.dirname(__file__)
data_dir = os.path.join(here, 'sample_data')
tides_dir = os.path.join(data_dir, 'tides')
lis_dir = os.path.join(data_dir, 'long_island_sound')

shio_file = get_datafile(os.path.join(tides_dir, 'CLISShio.txt'))
ossm_file = get_datafile(os.path.join(tides_dir, 'TideHdr.FINAL'))


def test_exceptions():
    """
    Test correct exceptions are raised
    """

    bad_file = os.path.join(lis_dir, 'CLISShio.txtX')
    bad_yeardata_path = os.path.join(data_dir, 'Data', 'yeardata')
    with raises(IOError):
        Tide(bad_file)

    with raises(IOError):
        Tide(shio_file, yeardata=bad_yeardata_path)


@pytest.mark.parametrize('filename', [shio_file, ossm_file])
def test_file(filename):
    """
    (WIP) simply tests that the file loads correctly
    """

    td = Tide(filename)
    assert td.filename == filename


@pytest.mark.parametrize(('filename', 'json_'),
                        [(shio_file, 'save'), (ossm_file, 'webapi')])
def test_serialize_deserialize(filename, json_):
    '''
    create - it creates new object after serializing original object
        and tests equality of the two

    update - tests serialize/deserialize and update_from_dict methods don't fail.
        It doesn't update any properties.
    '''
    tide = Tide(filename)
    serial = tide.serialize(json_)
    dict_ = Tide.deserialize(serial)
    if json_ == 'save':
        new_t = Tide.new_from_dict(dict_)
        assert new_t is not tide
        assert new_t == tide
    else:
        tide.update_from_dict(dict_)
        assert True
