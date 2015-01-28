'''
test dict_to_oil functions
'''
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)

import pytest
import unit_conversion as uc

from oil_library import get_oil_props
from oil_library.mock_oil import sample_oil_to_mock_oil

sample_oil = 'oil_conservative'
so = {'name': 'oil_conservative',
      'api': uc.convert('Density',
                        'gram per cubic centimeter',
                        'API degree', 1)}


def test_sample_oil_to_mock_oil():
    oil_ = sample_oil_to_mock_oil(max_cuts=2, **so)
    for key, val in so.iteritems():
        assert val == getattr(oil_, key)


@pytest.mark.xfail
def test_get_oil_props():
    op = get_oil_props(sample_oil)
    assert abs(sum(op.mass_fraction) - 1.0) < 1e-10
    assert op.mass_fraction > 0
    assert op.api == \
        uc.convert('density', 'kg/m^3', 'API', op.get_density(273.16 + 15))
