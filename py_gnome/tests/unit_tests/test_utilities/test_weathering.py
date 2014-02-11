#!/usr/bin/env python

"""
tests of the comp_volume code

designed to be run with nose

"""

import sys
sys.path.append("../lib")

from pytest import raises

import numpy
np = numpy


## Testing the weather curve calculation
#from tap_comp_volume import weather_curve, comp_volume
from gnome.utilities.weathering import weather_curve, WeatheringComponent

# we will probably get our oil types from a different module
# at some later date
from gnome.utilities.weathering import OilTypes


def test_weather_all_add_up():
    with raises(ValueError):
        wc = weather_curve(((0.5, 12),
                            (0.5, 24),
                            (0.5, 36)),
                           )
        print wc


def test_weather_all_add_up2():
    wc = weather_curve(((0.333333, 12),
                        (0.333333, 24),
                        (0.333334, 36)),
                       )
    print wc


def test_weather_single_component1():
    wc = weather_curve(((1.0, 12),))
    time = 12
    assert wc.weather(100, time) == 50.0


def test_weather_single_component2():
    wc = weather_curve(((1.0, 24),))
    time = 48
    assert wc.weather(100, time) == 25.0


def test_weather_single_component_0():
    wc = weather_curve(((1.0, 36),))
    time = 36 * 0
    assert wc.weather(128, time) == 128


def test_weather_single_component_1():
    wc = weather_curve(((1.0, 36),))
    time = 36 * 1
    assert wc.weather(128, time) == 64


def test_weather_single_component_2():
    wc = weather_curve(((1.0, 36),))
    time = 36 * 2
    assert wc.weather(128, time) == 32


def test_weather_single_component_3():
    wc = weather_curve(((1.0, 36),))
    time = 36 * 3
    assert wc.weather(128, time) == 16


def test_weather_single_component_4():
    wc = weather_curve(((1.0, 36),))
    time = 36 * 4
    assert wc.weather(128, time) == 8


def test_weather_single_component_10():
    wc = weather_curve(((1.0, 36),))
    time = 36 * 10
    assert wc.weather(128, time) == 128. / (2. ** 10)


def test_weather_two_component():
    wc = weather_curve(((.5, 50),
                        (.5, 25)),
                       )
    time = 50
    assert np.allclose(wc.weather(100, time), 37.5)


def test_weather_four_component():
    '''
       this one with different half-lives for each component
    '''
    wc = weather_curve(((.2, 120),
                        (.2, 60),
                        (.2, 40),
                        (.2, 30),
                        (.2, 24)),
                       )
    # num_half-lives: 1 2 3 4 5
    # 120 is divisible by 2, 3, 4, 5, 6
    M_0 = 100
    time = 120
    expected = (M_0 * .2) * (1.0 / 2 + 1.0 / 4 + 1.0 / 8 + 1.0 / 16 + 1.0 / 32)
    print  (M_0 * .25 / 2,
            M_0 * .25 / 4,
            M_0 * .25 / 8,
            M_0 * .25 / 16)
    assert np.allclose(wc.weather(M_0, time), expected)


def test_weather_five_component2():
    wc = weather_curve(((.2, 50),
                        (.2, 50),
                        (.2, 50),
                        (.2, 50),
                        (.2, 50)),
                       )
    time = 50
    print wc.weather(100, time)
    assert np.allclose(wc.weather(100, time), 50.0)


def test_weather_ten_component():
    wc = weather_curve(((.1, 25),
                        (.1, 25),
                        (.1, 25),
                        (.1, 25),
                        (.1, 25),
                        (.1, 50),
                        (.1, 50),
                        (.1, 50),
                        (.1, 50),
                        (.1, 50)),
                       )
    time = 50
    assert np.allclose(wc.weather(100, time), 37.5)


def test_weather_long_time():
    wc = weather_curve(((0.3, 12),
                        (0.3, 24),
                        (0.4, 1e10)),
                       )
    time = 1e10
    assert np.allclose(wc.weather(100, time), 0.4 * 50.0)


def test_weather_array():
    # this also tests all components the same
    wc = weather_curve(((0.3, 24),
                        (0.3, 24),
                        (0.4, 24)),
                       )
    time = np.array([24, 48, 72])
    mass = np.array([100, 1000, 10000])

    # This is old behavior which now raises an exception
    #result = np.array([50, 250, 1250], np.float32)

    with raises(ValueError):
        wc.weather(mass, time)


## test using the same code in a time-step by time-step approach:
def test_individual_timesteps_one():
    '''
       tests to see if we get the same results calling weather many times
       for a number of time steps, rather than all at once

       NOTE: this is only expected to work with one component!
             More than one, and the fractions will change as the components
             decay at different rates.
    '''
    # for a single component
    wc = weather_curve(((1.0, 24),))
    dt = 1.0
    num_steps = 10

    m_0 = 100.0
    m = m_0
    for i in range(num_steps):
        print i,
        m = (wc.weather(m, dt))
    print
    assert np.allclose(m, wc.weather(m_0, dt * num_steps))


def test_weathering_component():
    componentlist = [WeatheringComponent(c, f)
                     for c, f
                     in zip((0.3, 0.3, 0.4), (24, 24, 24))]
    assert [c.fraction for c in componentlist] == [0.3, 0.3, 0.4]
    assert [c.factor for c in componentlist] == [24, 24, 24]


def test_multiple_initial_masses():
    wc = weather_curve(((0.333333, 12),
                        (0.333333, 12),
                        (0.333334, 12)))

    print '\nTesting if we can pass multiple initial masses',
    print 'into our weather() function'
    res = wc.weather((100, 200, 300), 12)
    print res
    assert np.allclose(res, (50., 100., 150.))

    res = wc.weather((100, 200, 300), 24)
    print res
    assert np.allclose(res, (25., 50., 75.))


def test_multiple_decay_times():
    wc = weather_curve(((0.333333, 12),
                        (0.333333, 12),
                        (0.333334, 12)))

    print '\nTesting if we can pass multiple decay times',
    print ' into our weather() function.'
    with raises(ValueError):
        res = wc.weather(100, (12, 24, 36))


def test_mean_lifetime_method():
    # Test out our mean lifetime method
    # Basically our function is M_0 * exp(-time/tau)
    #     half-life = tau * ln(2)
    #     tau = half-life / ln(2)
    # So if our half life is 12 hrs,
    #     tau = (12 / np.log(2)) = 17.312340490667562
    print '\nTesting our mean lifetime method'
    wc = weather_curve(((0.333333, (12 / np.log(2))),
                        (0.333333, (12 / np.log(2))),
                        (0.333334, (12 / np.log(2)))),
                       method='mean-lifetime'
                       )
    res = wc.weather((100, 200, 300), 12)
    print res
    assert np.allclose(res, (50., 100., 150.))


def test_decay_constant_method():
    # Test out our decay constant method
    # Basically our function is M_0 * exp(-time * lambda)
    #     half-life = ln(2) / lambda
    #     lambda * half-life = ln(2)
    #     lambda = ln(2) / half-life
    # So if our half life is 12 hrs,
    #     lambda = (np.log(2) / 12) = 0.057762265046662105
    print '\nTesting our decay constant method'
    wc = weather_curve(((0.333333, (np.log(2) / 12)),
                        (0.333333, (np.log(2) / 12)),
                        (0.333334, (np.log(2) / 12))),
                       method='decay-constant'
                       )
    res = wc.weather((100, 200, 300), 12)
    print res
    assert np.allclose(res, (50., 100., 150.))


def test_multiple_weathering_steps():
    print '\nTesting if we can perform multiple sequential weathering steps'
    wc = weather_curve(((.5, 12),
                        (.5, 24)))

    print 'step 1'
    res = wc.weather(100, 24, update_fractions=True)
    assert np.allclose(res, (12.5 + 25.,))

    print 'step 2'
    expected = np.asarray(((res.sum() / 3) / 4,
                           (res.sum() * 2 / 3) / 2,
                           ),
                          dtype=np.float64)
    res = wc.weather(res, 24, update_fractions=True)
    assert np.allclose(res, expected.sum())

    print '\nTesting if we can perform multiple sequential weathering steps ',
    print 'using the explicit method'
    wc = weather_curve(((.5, 12),
                        (.5, 24)))

    print 'step 1'
    res = wc.weather(100, 24)
    assert np.allclose(res, (12.5 + 25.,))

    print 'step 2'
    expected = np.asarray(((res.sum() / 3) / 4,
                           (res.sum() * 2 / 3) / 2,
                           ),
                          dtype=np.float64)

    wc.update_fractions(24)
    res = wc.weather(res, 24)
    assert np.allclose(res, expected.sum())
