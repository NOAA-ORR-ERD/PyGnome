#!/usr/bin/env python

"""
tests of the comp_volume code

designed to be run with nose

"""

import sys

sys.path.append("../lib")

import numpy as np
import nose

## Testing the weather curve calculation
#from tap_comp_volume import weather_curve, comp_volume
from oil_weathering import weather_curve, OilTypes

@nose.tools.raises(ValueError)
def test_weather_all_add_up():
    wc = weather_curve( (0.5, 0.5, 0.5),
                        ( 12,  24,  36))
        
def test_weather_all_add_up2():
    wc = weather_curve((0.333333, 0.333333, 0.333334),
                       (12,       24,       36))

@nose.tools.raises(ValueError)
def test_weather_same_num_components():
    wc = weather_curve( (0.333333, 0.333333, 0.333334),
                        ( 12,  24,  36, 14))
        
        
def test_weather_single_component1():
    wc = weather_curve((1.0,), (12,))
    time = 12
    assert wc.weather(100, time) == 50.0 
           
def test_weather_single_component2():
    wc = weather_curve( (1.0,), (24,))
    time = 48
    assert wc.weather(100, time) == 25.0

def test_weather_single_component_0():
    wc = weather_curve( 1.0, 36)
    time = 36*0
    assert wc.weather(128, time) == 128

def test_weather_single_component_1():
    wc = weather_curve( 1.0, 36)
    time = 36*1
    assert wc.weather(128, time) == 64

def test_weather_single_component_2():
    wc = weather_curve( 1.0, 36)
    time = 36*2
    assert wc.weather(128, time) == 32

def test_weather_single_component_3():
    wc = weather_curve( 1.0, 36)
    time = 36*3
    assert wc.weather(128, time) == 16

def test_weather_single_component_4():
    wc = weather_curve( 1.0, 36)
    time = 36*4
    assert wc.weather(128, time) == 8

def test_weather_single_component_10():
    wc = weather_curve( 1.0, 36)
    time = 36*10
    assert wc.weather(128, time) == 128. / (2.**10)

def test_weather_two_component():
    wc = weather_curve( (.5, .5),
                        (50, 25)
                        ) 
    time = 50
    print wc.weather(100, time)
    assert np.allclose(wc.weather(100, time), 37.5)

def test_weather_four_component():
    """
    this one with different half-lives for each component
    """
    wc = weather_curve( (.2 , .2,  .2,  .2,  .2 ),
                        (120,  60,  40,  30, 24) )
    # num_half-lives:     1     2    3    4   5
    # 120 is divisible by 2,3,4,5,6
    M_0 = 100
    time = 120
    expected = (M_0*.2) * (1.0/2 + 1.0/4 + 1.0/8 + 1.0/16 + 1.0/32) 
    print  M_0*.25/2,  M_0*.25/4,  M_0*.25/8, M_0*.25/16

    assert np.allclose(wc.weather(M_0, time), expected)

def test_weather_five_component2():
    wc = weather_curve( (.2, .2, .2, .2, .2),
                        (50, 50, 50, 50, 50) )
    time = 50
    print wc.weather(100, time)
    assert np.allclose(wc.weather(100, time), 50.0)


def test_weather_ten_component():
    wc = weather_curve( (.1, .1, .1, .1, .1, .1, .1, .1, .1, .1),
                        (25, 25, 25, 25, 25, 50, 50, 50, 50, 50) )
    time = 50
    print wc.weather(100, time)
    assert np.allclose(wc.weather(100, time), 37.5)

def test_weather_long_time():
    wc = weather_curve( (0.3, 0.3, 0.4), (12, 24, 1e10))
    time = 1e10
    assert np.allclose(wc.weather(100, time), 0.4 * 50.0) 

def test_weather_array():
    # this also tests all components the same
    wc = weather_curve( (0.3, 0.3, 0.4), ( 24, 24, 24 ))
    time = np.array([24, 48, 72])
    mass = np.array([100, 1000, 10000])
    result = np.array([50, 250, 1250], np.float32)
    assert np.array_equal( wc.weather(mass, time), result )

## test using the same code in a time-step by time-step approach:
def test_individual_timesteps_one():
    """
    tests to see if we get the same results calling weather many times
    for a number of time steps, rather than all at once

    NOTE: this is only expected to work with one component!
    More than one, and the fractiosn wil change as the components decay at different rates.
    """
    # for a single component
    wc = weather_curve( (1.0,), (24,))
    dt = 1.0
    num_steps = 10

    m_0 = 100.0
    m = m_0
    for i in range(num_steps):
        m = (wc.weather(m, dt))        
    assert np.allclose(m, wc.weather(m_0, dt * num_steps) )

