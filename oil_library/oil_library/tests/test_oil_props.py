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
    sa = [ix for ix in op._r_oil.sara_fractions
          if ix.sara_type not in ('Resins', 'Asphaltenes')]
    rho_sa = [ix for ix in op._r_oil.sara_densities
              if ix.sara_type not in ('Resins', 'Asphaltenes')]
    sa.sort(key=lambda s: s.ref_temp_k)
    rho_sa.sort(key=lambda s: s.ref_temp_k)

    ra = [ix for ix in op._r_oil.sara_fractions
          if ix.sara_type in ('Resins', 'Asphaltenes')]
    rho_ra = [ix for ix in op._r_oil.sara_densities
              if ix.sara_type in ('Resins', 'Asphaltenes')]
    ra.sort(key=lambda s: s.sara_type, reverse=True)
    rho_ra.sort(key=lambda s: s.sara_type, reverse=True)

    def test_sara(self):
        # boiling points
        assert np.all(self.op.boiling_point[:-2] ==
                      [comp.ref_temp_k for comp in self.sa])
        np.all(self.op.boiling_point[-2:] == float('inf'))
        # resins and asphaltenes mass_fractions
        np.all(self.op.mass_fraction[-2:] ==
               [ix.fraction for ix in self.ra])

        # resins and asphaltenes density
        np.all(self.op.component_density[-2:] ==
               [ix.density for ix in self.rho_ra])

        # saturates + aromatics density + mass_fractions
        for ix in xrange(len(self.sa)/2):
            assert self.op._sara['type'][ix*2] == 'S'
            assert self.op._sara['type'][ix*2 + 1] == 'A'

            # Make no assumptions about order of sara_fraction and
            # sara_densities
            sa_frac = sorted(self.sa[ix*2:ix*2 + 2], key=lambda s: s.sara_type,
                             reverse=True)
            rho_sa = sorted(self.rho_sa[ix*2:ix*2 + 2],
                            key=lambda s: s.sara_type, reverse=True)
            # Saturates
            assert sa_frac[0].sara_type == 'Saturates'
            assert sa_frac[0].fraction == self.op.mass_fraction[ix*2]
            assert rho_sa[0].density == self.op.component_density[ix*2]

            # Aromatics
            assert sa_frac[1].sara_type == 'Aromatics'
            assert sa_frac[1].fraction == self.op.mass_fraction[ix*2 + 1]
            assert rho_sa[1].density == self.op.component_density[ix*2 + 1]



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
