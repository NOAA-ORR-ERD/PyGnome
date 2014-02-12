'''
Tests for oil_props module in gnome.db.oil_library
'''

import pytest

from gnome.db.oil_library.oil_props import (OilProps, OilPropsFromDensity)
from hazpy import unit_conversion


def test_OilProps_exceptions():
    from sqlalchemy.orm.exc import NoResultFound
    with pytest.raises(NoResultFound):
        OilProps('test')
    with pytest.raises(unit_conversion.InvalidUnitError):
        OilPropsFromDensity(density=.9,units='kg/m**3')

# just double check values for _sample_oil are entered correctly

oil_density_units = [
    ('oil_gas', 0.75, 'g/cm^3'),
    ('oil_jetfuels', 0.81, 'g/cm^3'),
    ('oil_4', 0.90, 'g/cm^3'),
    ('oil_crude', 0.90, 'g/cm^3'),
    ('oil_6', 0.99, 'g/cm^3'),
    ('oil_conservative', 1, 'g/cm^3'),
    ('chemical', 1, 'g/cm^3'),
    ]


@pytest.mark.parametrize(('oil', 'density', 'units'), oil_density_units)
def test_OilProps_sample_oil(oil, density, units):
    """ compare expected values with values stored in OilProps - make sure
    data entered correctly and unit conversion is correct """

    o = OilProps(oil)
    assert o.get_density(units) == density
    assert o.name == oil

@pytest.mark.parametrize(('oil', 'density', 'units'), [('my_oil',.98,'g/cm^3')])
def test_OilPropsFromDensity(oil, density, units):
    """ make sure data entered correctly and unit conversion is correct """

    o = OilPropsFromDensity(density,oil,units)
    assert o.get_density(units) == density
    assert o.name == oil

@pytest.mark.parametrize(('oil', 'api'), [('FUEL OIL NO.6', 12.3)])
def test_OilProps_DBquery(oil, api):
    """ test dbquery worked for an example like FUEL OIL NO.6 """
    o = OilProps(oil)
    assert o.oil.api == api

def test_set_properties():
    """
    test setting / getting properties
    """
    o = OilProps('oil_conservative')
    o.name = 'my_oil'
    assert o.name == 'my_oil'
    o.density = 950
    assert o.density == 950
