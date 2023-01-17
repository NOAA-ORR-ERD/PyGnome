"""
tests for water object
"""

from datetime import datetime
import numpy as np

import pytest

from nucos import InvalidUnitError, convert

from gnome.utilities.inf_datetime import InfDateTime, InfTime
from gnome.environment.water import (Water, Temperature, Salinity,
                                     Sediment, WaveHeight)


def test_Temperature():
    """
    not much to it
    """
    w = Water(temperature=293.0)  # always good to test not with defaults
    temp = Temperature(w)

    result = temp.at([(1, 2, 3),(4, 5, 6),(7, 8, 9)],
                     datetime(1000, 1, 1, 0) )

    assert np.array_equal(result, np.array([293., 293., 293.]))

def test_Temperature_new_units():
    """
    not much to it
    """
    t_k = 293.0
    t_c = convert("K", "C", t_k)

    w = Water(temperature=t_k)  # always good to test not with defaults

    result = w.Temperature.at([(1, 2, 3), (4, 5, 6), (7, 8, 9)],
                              datetime(1000, 1, 1, 0),
                              units='C')

    assert np.array_equal(result, np.array([t_c] * 3))


def test_unrealistic_temp():
    with pytest.warns(UserWarning):
        w = Water(70)


def test_Salinity():
    """
    not much to it
    """
    s = 33.0
    w = Water(salinity=s)  # always good to test not with defaults
    sal = Salinity(w)

    result = sal.at([(1, 2, 3), (4, 5, 6), (7, 8, 9)], datetime(1000, 1, 1, 0))

    assert np.array_equal(result, np.array([s, s, s]))


def test_Salinity_Water():
    """
    not much to it
    """
    s = 33.0
    w = Water(salinity=s)  # always good to test not with defaults

    result = w.Salinity.at([(1, 2, 3), (4, 5, 6), (7, 8, 9)],
                           datetime(1000, 1, 1, 0))

    assert np.array_equal(result, np.array([s, s, s]))


def test_Salinity_new_unit():
    """
    only psu is accepted
    """
    s = 33.0
    w = Water(salinity=s)  # always good to test not with defaults
    sal = Salinity(w)

    with pytest.raises(InvalidUnitError):
        sal.at([(1, 2, 3), (4, 5, 6), (7, 8, 9)],
               datetime(1000, 1, 1, 0),
               units='ppt')


def test_Sediment():
    w = Water(sediment=.004)
    assert w.Sediment.at([(0, 0, 0), ], InfTime) == 0.004


def test_WaveHeight():
    w = Water(wave_height=1.1)

    result = w.WaveHeight.at([(0, 0, 0), ], InfTime)
    assert len(result == 1)
    assert result[0] == 1.1


def test_psu_water():
    """
    psu is not in nucos -- so there were issues
    """
    w = Water()
    result = w.get('salinity', unit='psu')

    assert result == w.salinity




#############
# previous tests for legacy water object API
# still in use -- but we really don't need that whole API
# So these tests can be removed / cleaned up as we refactor
#############

# pytest.mark.parametrize() is really finicky about what you can use as a
# string parameter.  You can't use a list or a tuple, for example.
@pytest.mark.parametrize(('attr', 'sub_attr', 'value'),
                         [('name', None, 'NewWater'),
                          ('temperature', None, 400.0),
                          ('salinity', None, 50.0),
                          ('sediment', None, .01),
                          ('wave_height', None, 2.0),
                          ('fetch', None, 100.0),

                          ('units', 'temperature', 'C'),
                          ('units', 'salinity', 'psu'),
                          ('units', 'sediment', 'kg/m^3'),
                          ('units', 'wave_height', 'm'),
                          ('units', 'fetch', 'm'),
                          ('units', 'density', 'kg/m^3'),
                          ('units', 'kinematic_viscosity', 'm^2/s'),
                          ])
def test_Water_init(attr, sub_attr, value):
    '''
        The initial default values that an object may have is a contract.
        As such, we test this contract for a Water() object here.

        Specifically, we test:
        - that the default values are what we expect,
        - that the default values are immutable.
    '''
    w = Water()

    check_water_defaults(w)

    if sub_attr is None:
        setattr(w, attr, value)
    else:
        sub_value = getattr(w, attr)
        print('sub_value = ', sub_value)
        sub_value[sub_attr] = value
        print('sub_value = ', sub_value)

    w = Water()

    check_water_defaults(w)


def check_water_defaults(water_obj):
    assert water_obj.name == 'Water'

    print(repr(water_obj.data_start))
    print(repr(InfDateTime("-inf")))
    assert water_obj.data_start == InfDateTime("-inf")
    assert water_obj.data_stop == InfDateTime("inf")

    assert water_obj.temperature == 300.0
    assert water_obj.salinity == 35.0
    assert water_obj.sediment == .005
    assert water_obj.wave_height is None
    assert water_obj.fetch is None

    assert water_obj.units['temperature'] == 'K'
    assert water_obj.units['salinity'] == 'psu'
    assert water_obj.units['sediment'] == 'kg/m^3'
    assert water_obj.units['wave_height'] == 'm'
    assert water_obj.units['fetch'] == 'm'
    assert water_obj.units['density'] == 'kg/m^3'
    assert water_obj.units['kinematic_viscosity'] == 'm^2/s'


def test_not_implemented_in_water():
    sample_time = 60 * 60 * 24 * 365 * 30  # seconds
    w = Water()

    with pytest.raises(AttributeError):
        w.data_start = sample_time

    with pytest.raises(AttributeError):
        w.data_stop = sample_time


@pytest.mark.parametrize(("attr", "unit"), [('temperature', 'kg'),
                                            ('sediment', 'kg'),
                                            ('salinity', 'ppt'),
                                            ('wave_height', 'l'),
                                            ('fetch', 'ppt')])
def test_unit_errors(attr, unit):
    '''
        - currently salinity only has psu in there since there is
          no conversion from psu to ppt, though ppt is a valid unit.
          This needs to be fixed
        - similarly, sediment only has mg/l as units.  We need to decide
          if we want more units here
    '''
    w = Water()
    w.wave_height = 1
    w.fetch = 10000

    with pytest.raises(InvalidUnitError):
        w.get(attr, unit)

    with pytest.raises(InvalidUnitError):
        w.set(attr, 5, unit)


@pytest.mark.parametrize(("attr", "unit", "val", "si_val"),
                         [('temperature', 'C', 0, 273.15),
                          ('sediment', 'mg/l', 5, 0.005),
                          ('sediment', 'percent', 0.005, 0.05),
                          ('wave_height', 'cm', 100.0, 1.0),
                          ('fetch', 'km', 1.0, 1000.0)])
def test_Water_get(attr, unit, val, si_val):
    w = Water()
    setattr(w, attr, val)
    w.units[attr] = unit

    assert w.get(attr) == si_val
    assert w.get(attr, unit) == val


@pytest.mark.parametrize(("attr", "unit"), [('temperature', 'F'),
                                            ('sediment', 'mg/l'),
                                            ('sediment', 'part per thousand'),
                                            ('wave_height', 'km'),
                                            ('fetch', 'km')])
def test_Water_set(attr, unit):
    w = Water()
    w.set(attr, 1.0, unit)
    assert getattr(w, attr) == 1.0
    assert w.units[attr] == unit


def test_Water_density():
    '''
    for default salinity and water temp, water density is > 1000.0 kg/m^3
    '''
    w = Water()
    assert w.density > 1000.0


def test_Water_update_from_dict():
    '''
    test that the update_from_dict correctly sets fetch and wave_height to None
    if it is dropped from json payload so user chose compute from wind option.
    '''
    w = Water()
    json_ = w.serialize()
    w.fetch = 0.0
    w.wave_height = 1.0
    json_with_values = w.serialize()

    w.update_from_dict(json_)
    assert w.fetch is None
    assert w.wave_height is None

    w.update_from_dict(json_with_values)
    assert w.fetch == 0.0
    assert w.wave_height == 1.0


@pytest.mark.parametrize(("attr", "unit", "val", "exp_si"),
                         [('temperature', 'C', 0, 273.15),
                          ('sediment', 'mg/l', 5, 0.005),
                          ('wave_height', 'km', .001, 1),
                          ('fetch', 'km', .01, 10),
                          ('fetch', 'm', 0.323, 0.323)])
def test_properties_in_si(attr, unit, val, exp_si):
    '''
    set properties in non SI units and check default get() returns it in SI
    '''
    kw = {attr: val, 'units': {attr: unit}}
    w = Water(**kw)
    assert getattr(w, attr) == val
    assert w.units[attr] == unit

    assert w.get(attr) == exp_si


