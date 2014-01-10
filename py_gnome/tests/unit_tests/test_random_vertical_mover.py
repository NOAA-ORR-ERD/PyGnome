#!/usr/bin/env python

"""
tests for the RandomVerticalMover in random_movers.py
"""

import datetime
import numpy as np

from gnome import basic_types
from gnome.movers.random_movers import RandomVerticalMover

from conftest import sample_sc_release
import pytest
# some helpful parameters:

model_time = datetime.datetime(2014, 1, 8, 12)
time_step = 600 # 10 minutes in seconds

def test_init():
    """
    test the varios ways it can be initilized
    """
    mv = RandomVerticalMover()

    mv = RandomVerticalMover(vertical_diffusion_coef_above_ml=10.0) # cm^2/s

    mv = RandomVerticalMover(vertical_diffusion_coef_below_ml=2.0) # cm^2/s

    mv = RandomVerticalMover(mixed_layer_depth=20) # m

    assert True

def test_horizontal_zero():
    """
    checks that there is no horizontal movement
    """
    mv = RandomVerticalMover() # all defaults

    num_elements = 100

    sc = sample_sc_release(num_elements=num_elements,
                           start_pos=(0.0, 0.0, 0.0),
                           release_time=model_time,
                           )
    # set z positions:
    sc['positions'][:,2] = np.linspace(0, 50, num_elements)
    
    delta = mv.get_move(sc,
                        time_step,
                        model_time,
                        )

    print delta

    assert np.alltrue(delta[:,0:2] == 0.0)

def test_one_layer():
    """
    tests a zero-thickness mixed mixed_layer_depth

    elements should vertically spread according to the diffusion coef of the bottom layer
    """
    mv = RandomVerticalMover(mixed_layer_depth=0.0) # m

    num_elements = 10
    sc = sample_sc_release(num_elements=num_elements,
                           start_pos=(0.0, 0.0, 0.0),
                           release_time=model_time,
                           )
    # re-set z positions:
    sc['positions'][ :num_elements/2 , 2] =  5.0 # near top
    sc['positions'][  num_elements/2:, 2] = 50.0 # down deep

    # call get_move a bunch of times
    for i in range(10):
        print "positions:\n", sc['positions']
        delta = mv.get_move(sc,
                            time_step,
                            model_time,
                            )
        print "delta:\n", delta
        sc['positions'] += delta

    print sc['positions']

    assert False





