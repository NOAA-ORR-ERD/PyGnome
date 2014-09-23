'''
test dict_to_oil functions
'''
import pytest
from hazpy import unit_conversion as uc

from oil_library import _sample_oils, get_oil_props
from oil_library.mock_oil import sample_oil_to_mock_oil, boiling_point

sample_oil = 'oil_conservative'
so = _sample_oils[sample_oil]


def test_sample_oil_to_mock_oil():
    oil_ = sample_oil_to_mock_oil(max_cuts=2, **so)
    for key, val in so.iteritems():
        assert val == getattr(oil_, key)


def test_get_oil_props():
    op = get_oil_props(sample_oil)
    assert abs(sum(op.mass_fraction) - 1.0) < 1e-10
    assert op.mass_fraction > 0
    assert op.api == \
        uc.convert('density', 'kg/m^3', 'API', op.get_density(273.16 + 15))


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
