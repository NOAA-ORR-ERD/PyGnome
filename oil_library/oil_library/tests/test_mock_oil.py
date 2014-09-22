'''
test dict_to_oil functions
'''
from hazpy import unit_conversion as uc

from oil_library import _sample_oils, get_oil_props
from oil_library.mock_oil import sample_oil_to_mock_oil

so = _sample_oils['conservative_oil']


def test_dict_to_oil_obj():
    oil_ = sample_oil_to_mock_oil(so, max_cuts=2)
    for key, val in so.iteritems():
        assert val == getattr(oil_, key)


def test_get_oil_props():
    op = get_oil_props(so)
    assert sum(op.mass_fraction) == 1.0
    assert all(op.mass_fraction > 0)
    assert op.api == \
        uc.convert('density', 'kg/m^3', 'API', op.get_density(273.16 + 15))
