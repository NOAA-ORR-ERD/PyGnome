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
                                            ('sediment', 'kg/m^3'),
                                            ('salinity', 'ppt'),
                                            ('wave_height', 'l'),
                                            ('fetch', 'ppt')])
def test_exceptions(attr, unit):
    w = Water()

    with pytest.raises(InvalidUnitError):
        w.get(attr, unit)

    with pytest.raises(InvalidUnitError):
        w.set(attr, 5, unit)


def test_Water_get():
    w = Water(temperature=273.16)
    w.sediment = 10.0
    w.wave_height = 5.0

    assert w.get('temperature', 'K') == 273.16
    assert w.get('temperature', 'C') == 0.0
    assert w.get('wave_height', 'km') == w.wave_height / 1000.0


@pytest.mark.parametrize(("attr", "unit"), [('temperature', 'F'),
                                            ('sediment', 'mg/l'),
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
