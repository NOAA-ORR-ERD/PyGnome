'''
test object in environment module
'''
import pytest
from unit_conversion import InvalidUnitError
from gnome.environment import Water


def test_Water_init():
    w = Water()
    assert w.temperature == 300.0
    assert w.salinity == 35.0
    w = Water(temperature=273, salinity=0)
    assert w.temperature == 273.0
    assert w.salinity == 0.0


# currently salinity only have psu in there since there is no conversion from
# psu to ppt, though ppt is a valid unit - needs to be fixed
# similarly, sediment only has mg/l as units - decide if we want more units
# here
@pytest.mark.parametrize(("attr", "unit"), [('temperature', 'kg'),
                                            ('sediment', 'kg'),
                                            ('salinity', 'ppt'),
                                            ('wave_height', 'l'),
                                            ('fetch', 'ppt')])
def test_exceptions(attr, unit):
    w = Water()

    with pytest.raises(InvalidUnitError):
        w.get(attr, unit)

    with pytest.raises(InvalidUnitError):
        w.set(attr, 5, unit)


@pytest.mark.parametrize(("attr", "unit", "val", "si_val"),
                         [('temperature', 'C', 0, 273.16),
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

    w.update_from_dict(Water.deserialize(json_))
    assert w.fetch is None
    assert w.wave_height is None

    w.update_from_dict(Water.deserialize(json_with_values))
    assert w.fetch == 0.0
    assert w.wave_height == 1.0


@pytest.mark.parametrize(("attr", "unit", "val", "exp_si"),
                         [('temperature', 'C', 0, 273.16),
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
