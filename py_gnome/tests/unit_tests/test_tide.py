'''
Test all operations for cats mover work
'''
import os

import pytest

from gnome import environment

here = os.path.dirname(__file__)
data_dir = os.path.join(here, 'sample_data')
tides_dir = os.path.join(data_dir, 'tides')
lis_dir = os.path.join(data_dir, 'long_island_sound')

shio_file = os.path.join(tides_dir, "CLISShio.txt")
ossm_file = os.path.join(tides_dir, "TideHdr.FINAL")


def test_exceptions():
    """
    Test correct exceptions are raised
    """
    bad_file = os.path.join(lis_dir, "CLISShio.txtX")
    bad_yeardata_path = os.path.join(data_dir, "Data", "yeardata")
    with pytest.raises(IOError):
        environment.Tide(bad_file)

    with pytest.raises(IOError):
        environment.Tide(shio_file, yeardata=bad_yeardata_path)


@pytest.mark.parametrize("filename", [shio_file, ossm_file])
def test_file(filename):
    """
    (WIP) simply tests that the file loads correctly
    """
    td = environment.Tide(filename)
    assert td.filename == filename


@pytest.mark.parametrize("filename", [shio_file, ossm_file])
def test_new_from_dict(filename):
    """
    test to_dict function for Wind object
    create a new wind object and make sure it has same properties
    """
    td = environment.Tide(filename)
    td_state = td.to_dict('create')

    # this does not catch two objects with same ID
    td2 = environment.Tide.new_from_dict(td_state)

    assert td == td2


@pytest.mark.parametrize("filename", [shio_file, ossm_file])
def test_from_dict(filename):
    """
    test from_dict function for Tide object
    no values are changed so it just tests that there are no failures
    """
    wm = environment.Tide(filename)
    wm_dict = wm.to_dict()

    wm.from_dict(wm_dict)

    for key in wm_dict.keys():
        assert wm.__getattribute__(key) == wm_dict.__getitem__(key)
