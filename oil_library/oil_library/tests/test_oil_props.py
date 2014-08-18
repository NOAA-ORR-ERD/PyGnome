'''
Tests for oil_props module in gnome.db.oil_library
'''

import pytest

from oil_library import get_oil, oil_from_density

from hazpy import unit_conversion


def test_OilProps_exceptions():
    from sqlalchemy.orm.exc import NoResultFound
    with pytest.raises(NoResultFound):
        get_oil('test')
    with pytest.raises(unit_conversion.InvalidUnitError):
        oil_from_density(density=.9, units='kg/m**3')

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

    o = get_oil(oil)
    assert o.get_density(units) == density
    assert o.name == oil


@pytest.mark.parametrize(('oil', 'density', 'units'),
                         [('my_oil', .98, 'g/cm^3')])
def test_OilPropsFromDensity(oil, density, units):
    """ make sure data entered correctly and unit conversion is correct """

    o = oil_from_density(density, oil, units)
    assert o.get_density(units) == density
    assert o.name == oil


@pytest.mark.parametrize(('oil', 'api'), [('FUEL OIL NO.6', 12.3)])
def test_OilProps_DBquery(oil, api):
    """ test dbquery worked for an example like FUEL OIL NO.6 """
    o = get_oil(oil)
    assert o.api == api
