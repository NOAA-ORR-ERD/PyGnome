"""
Tests for cy_shio_time.pyx class

Does some basic testing of properties.

todo: still need to add tests for functions that get the data from C++ code
"""

import os
from datetime import datetime

import pytest

import gnome
from gnome.utilities import time_utils
from gnome.cy_gnome.cy_shio_time import CyShioTime
from ..conftest import testdata

shio_file = testdata['timeseries']['tide_shio']


def test_exceptions():
    bad_file = os.path.join('./', 'CLISShio.txtX')
    bad_yeardata_path = os.path.join('./', 'Data', 'yeardata')
    with pytest.raises(IOError):
        CyShioTime(bad_file)

    with pytest.raises(IOError):
        shio = CyShioTime(shio_file)
        shio.set_shio_yeardata_path(bad_yeardata_path)


check_items = {
    'Type=': 'station_type',
    'Name=': 'station',
    'Longitude=': 'station_location',
    'Latitude=': 'station_location',
    }


def test_properties_read_from_file():
    """ read only properties """

    shio = CyShioTime(shio_file)
    print shio.filename
    with open(shio_file) as fh:
        data = fh.read()

    for (item, prop) in check_items.iteritems():
        s_idx = data.find(item)
        e_idx = data.find('\n', s_idx)
        val = getattr(shio, prop)
        if item == 'Longitude=':
            assert round(val[0], 4) == round(float(data[s_idx +
                                                        len(item):e_idx]), 4)
        elif item == 'Latitude=':

            assert round(val[1], 4) == round(float(data[s_idx
                                                        + len(item):e_idx]), 4)
        else:

            assert val == data[s_idx + len(item):e_idx]

    assert not shio.daylight_savings_off
    assert shio.path_filename == shio_file


def test_daylight_savings_off():
    shio = CyShioTime(shio_file)
    assert not shio.daylight_savings_off
    shio.daylight_savings_off = True
    assert shio.daylight_savings_off


def test_scale_factor():
    shio = CyShioTime(shio_file)
    assert shio.scale_factor == 1
    shio.scale_factor = 2
    assert shio.scale_factor == 2


def test_yeardata():
    'test default yeardata is set correctly. Also test set function.'
    shio = CyShioTime(shio_file)

    yd = os.path.join(os.path.dirname(gnome.__file__), 'data', 'yeardata')
    # shio puts a trailing slash at the end that is platform dependent
    # so compare the names without the platform dependent trailing slash
    assert shio.yeardata[:-1] == yd

    # bogus path but we just want to test that it gets set
    shio.yeardata = yd
    assert shio.yeardata[:-1] == yd


def test_get_time_value():
    'make sure get_time_value goes to correct C++ derived class function'
    shio = CyShioTime(shio_file)
    t = time_utils.date_to_sec(datetime(2012, 8, 20, 13))
    time = [t + 3600.*dt for dt in range(10)]
    vel_rec = shio.get_time_value(time)
    assert all(vel_rec['u'] != 0)
    assert all(vel_rec['v'] == 0)


def test_eq():
    shio = CyShioTime(shio_file)

    other_shio = CyShioTime(shio_file)
    assert shio == other_shio

    other_shio = CyShioTime(shio_file)
    other_shio.daylight_savings_off = True
    assert shio != other_shio

    other_shio = CyShioTime(shio_file)
    other_shio.scale_factor = 2
    assert shio != other_shio

    # TODO: need to test for inequality if read-only attributes are
    # different.  These include:
    # - station
    # - station_type
    # - station_location
    # For this, we probably need to open up sample files that have
    # different properties.  And we need to make them available via
    # get_datafile()


def test_eval_repr():
    print "in test_eval_repr: shio_file is:", shio_file
    shio = CyShioTime(shio_file)

    print "the repr is:", repr(shio)
    other_shio = eval(repr(shio))
    assert shio == other_shio

    yd = os.path.join(os.path.dirname(gnome.__file__), 'data', 'yeardata')
    shio.yeardata = yd
    other_shio = eval(repr(shio))
    assert shio == other_shio

    shio.daylight_savings_off = False
    other_shio = eval(repr(shio))
    assert shio == other_shio

    shio.scale_factor = 2
    other_shio = eval(repr(shio))
    assert shio == other_shio
