'''
Test Langmuir() - very simple object with only one method
'''
from datetime import datetime

import pytest

from gnome.environment import constant_wind, Langmuir


rel_buoy = 0.2
thick = 1e-4
model_time = datetime(2015, 1, 1, 12, 0)

wind = constant_wind(5, 0)


def test_init():
    l = Langmuir(wind)
    assert l.wind is wind


l = Langmuir(wind)
(vmin, vmax) = l._wind_speed_bound(rel_buoy, thick)


@pytest.mark.parametrize(("l", "speed", "exp_bound"),
                         [(l, vmin - 0.01 * vmin, 1.0),
                          (l, vmax + 0.01 * vmax, 0.1)])
def test_speed_bounds(l, speed, exp_bound):
    '''
    Check that the input speed for Langmuir object returns frac_coverage
    within bounds:
        0.1 <= frac_cov <= 1.0
    '''
    l.wind.timeseries = (l.wind.timeseries['time'][0], (speed, 0.0))
    frac_cov = l.get_value(model_time, rel_buoy, thick)
    assert frac_cov == exp_bound


def test_update_from_dict():
    '''
    just a simple test to ensure schema/serialize/deserialize is correclty
    setup
    '''
    j = l.serialize()
    j['wind']['timeseries'][0] = \
        (j['wind']['timeseries'][0][0],
         (j['wind']['timeseries'][0][1][0] + 1, 0))
    updated = l.update_from_dict(Langmuir.deserialize(j))
    assert updated
    assert l.serialize() == j
