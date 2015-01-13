'''
Tests for oil_props module in gnome.db.oil_library
'''
import copy

import pytest
import unit_conversion as uc

from oil_library import get_oil_props


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
            assert getattr(op, item) == getattr(cop, item)
            assert getattr(op, item) is getattr(cop, item)

        # shallow copy means cop.mass_fraction list is a reference to original so
        # changing it in op, also changes it for cop
        op.mass_fraction[0] = 0
        assert op.mass_fraction == cop.mass_fraction

    def _assert_deepcopy(self, op, dcop):
        assert op == dcop
        assert op is not dcop

        for item in op.__dict__:
            print "item checking:", item
            assert getattr(op, item) == getattr(dcop, item)
            if item == '_r_oil' or getattr(op, item) is None:
                assert getattr(op, item) is getattr(dcop, item)
            else:
                assert getattr(op, item) is not getattr(dcop, item)

    def test_deepcopy(self):
        '''
        do a shallow copy and test that it is a shallow copy
        '''
        op = get_oil_props(10)
        dcop = copy.deepcopy(op)
        self._assert_deepcopy(op, dcop)

        # deepcopy means dcop.mass_fraction list is a new list as opposed to a
        # reference so changing it in 'op' doesn't effect the list in 'dcop'
        op.mass_fraction[0] = 0
        assert op.mass_fraction != dcop.mass_fraction
