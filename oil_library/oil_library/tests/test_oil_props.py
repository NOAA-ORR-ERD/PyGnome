'''
Tests for oil_props module in gnome.db.oil_library
'''

import pytest

from numpy import isclose

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
    assert isclose(o.get_density(units), density, atol=1e-3)
    assert o.name == oil


@pytest.mark.parametrize(('oil', 'density', 'units'),
                         [('my_oil', .98, 'g/cm^3')])
def test_OilPropsFromDensity(oil, density, units):
    """ make sure data entered correctly and unit conversion is correct """

    o = oil_from_density(density, oil, units)

    # the temperature must be the same to get a computed density that
    # is close to our input
    o.temperature = 273.15 + 15

    assert isclose(o.get_density(units), density)
    assert o.name == oil


@pytest.mark.parametrize(('oil', 'api'), [('FUEL OIL NO.6', 12.3)])
def test_OilProps_DBquery(oil, api):
    """ test dbquery worked for an example like FUEL OIL NO.6 """
    o = get_oil(oil)
    assert o.api == api


@pytest.mark.parametrize(('oil', 'temp', 'viscosity'),
                         [('FUEL OIL NO.6', 311.15, 0.000383211),
                          ('FUEL OIL NO.6', 288.15, 0.045808748),
                          ('FUEL OIL NO.6', 280.0, 0.045808749)
                          ])
def test_OilProps_Viscosity(oil, temp, viscosity):
    """
        test dbquery worked for an example like FUEL OIL NO.6
        Here are the measured viscosities:
           [<KVis(meters_squared_per_sec=1.04315461221, ref_temp=273.15, weathering=0.0)>,
            <KVis(meters_squared_per_sec=0.0458087487284, ref_temp=288.15, weathering=0.0)>,
            <KVis(meters_squared_per_sec=0.000211, ref_temp=323.15, weathering=0.0)>]
    """
    o = get_oil(oil)
    o.temperature = temp
    assert isclose(o.viscosity, viscosity)
