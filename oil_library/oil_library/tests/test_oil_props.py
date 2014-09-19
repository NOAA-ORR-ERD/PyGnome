'''
Tests for oil_props module in gnome.db.oil_library
'''

import pytest

from oil_library import get_oil_props
from oil_library.oil_props import boiling_point


def test_OilProps_exceptions():
    from sqlalchemy.orm.exc import NoResultFound
    with pytest.raises(NoResultFound):
        get_oil_props('test')

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


@pytest.mark.xfail
@pytest.mark.parametrize(('oil', 'density', 'units'), oil_density_units)
def test_OilProps_sample_oil(oil, density, units):
    """ compare expected values with values stored in OilProps - make sure
    data entered correctly and unit conversion is correct """

    o = get_oil_props(oil)
    assert abs(o.get_density(units)-density) < 1e-3
    assert o.name == oil


@pytest.mark.parametrize(('oil', 'api'), [('FUEL OIL NO.6', 12.3)])
def test_OilProps_DBquery(oil, api):
    """ test dbquery worked for an example like FUEL OIL NO.6 """
    o = get_oil_props(oil)
    assert o.api == api


@pytest.mark.xfail
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
    o = get_oil_props(oil)
    o.temperature = temp
    assert abs((o.viscosity - viscosity)/viscosity) < 1e-5  # < 0.001 %


@pytest.mark.parametrize("max_cuts", (1, 2, 3, 4, 5))
def test_boiling_point(max_cuts):
    '''
    some basic testing of boiling_point function
    - checks the expected BP for 0th component for api=1
    - checks len(bp) == max_cuts * 2
    - also checks the BP for saturates == BP for aromatics
    '''
    api = 1
    slope = 1356.7
    intercept = 457.16 - 3.3447

    exp_bp_0 = 1./(max_cuts * 2) * slope + intercept
    bp = boiling_point(max_cuts, api)
    print '\nBoiling Points: '
    print bp
    assert len(bp) == max_cuts * 2
    assert ([bp[ix] - bp[ix + 1] for ix in range(0, max_cuts * 2, 2)] ==
            [0.0] * max_cuts)
    assert bp[:2] == [exp_bp_0] * 2


@pytest.mark.xfail
def test_get_density():
    'test get_density uses temp given as input'
    o = get_oil_props('FUEL OIL NO.6')
    assert o.get_density() != o.get_density(temp=273)
