'''
Tests for oil_props module in gnome.db.oil_library
'''
import copy

import numpy as np
import pytest
import unit_conversion as uc

from oil_library import get_oil_props, get_oil


def test_OilProps_exceptions():
    from sqlalchemy.orm.exc import NoResultFound
    with pytest.raises(NoResultFound):
        get_oil_props('test')


@pytest.mark.parametrize("search", ['FUEL OIL NO.6', 51])
def test_get_oil(search):
    o = get_oil(search)
    if isinstance(search, basestring):
        assert o.name == search
    else:
        # cannot search by adios ID yet
        assert o.imported_record_id == search
        assert o.imported.id == search


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
    d = uc.convert('density', units, 'kg/m^3', density)
    assert abs(o.get_density(273.16 + 15) - d) < 1e-3
    assert abs(o.get_density() - d) < 1e-3
    assert o.name == oil


@pytest.mark.parametrize(('oil', 'api'), [('FUEL OIL NO.6', 12.3)])
def test_OilProps_DBquery(oil, api):
    """ test dbquery worked for an example like FUEL OIL NO.6 """
    o = get_oil_props(oil)
    assert o.api == api


class TestProperties:
    op = get_oil_props(u'ALASKA NORTH SLOPE')
    s_comp = sorted(op._r_oil.sara_fractions, key=lambda s: s.ref_temp_k)
    s_dens = sorted(op._r_oil.sara_densities, key=lambda s: s.ref_temp_k)
    #s_cuts = sorted(op._r_oil.cuts, key=lambda s: s.vapor_temp_k)

    def test_sara(self):
        # boiling points
        assert np.all(self.op.boiling_point ==
                      [comp.ref_temp_k for comp in self.s_comp])

        # mass fraction
        assert np.all(self.op.mass_fraction ==
                      [comp.fraction for comp in self.s_comp])

        # sara type
        assert np.all(self.op._sara['type'] ==
                      [comp.sara_type for comp in self.s_comp])

        # density
        assert np.all(self.op.boiling_point ==
                      [comp.ref_temp_k for comp in self.s_dens])

        assert np.all(self.op.component_density ==
                      dens for dens in self.s_dens)

        assert np.allclose(self.op.mass_fraction.sum(), 1.0)


def test_eq():
    op = get_oil_props(10)
    op1 = get_oil_props(10)
    assert op == op1


def test_ne():
    assert get_oil_props(10) != get_oil_props(11)


class TestCopy():
    def test_copy(self):
        '''
        do a shallow copy and test that it is a shallow copy
        '''
        op = get_oil_props(10)
        cop = copy.copy(op)
        assert op == cop
        assert op is not cop
        assert op._r_oil is cop._r_oil

        for item in op.__dict__:
            try:
                assert getattr(op, item) == getattr(cop, item)
            except ValueError:
                assert np.all(getattr(op, item) == getattr(cop, item))

            assert getattr(op, item) is getattr(cop, item)

    def test_deepcopy(self):
        '''
        do a shallow copy and test that it is a shallow copy
        '''
        op = get_oil_props(10)
        dcop = copy.deepcopy(op)

        assert op == dcop
        assert op is not dcop

        for item in op.__dict__:
            print "item checking:", item
            try:
                assert getattr(op, item) == getattr(dcop, item)
            except ValueError:
                assert np.all(getattr(op, item) == getattr(dcop, item))

            if item == '_r_oil' or getattr(op, item) is None:
                assert getattr(op, item) is getattr(dcop, item)
            else:
                assert getattr(op, item) is not getattr(dcop, item)
