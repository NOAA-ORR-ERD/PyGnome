#!/usr/bin/env python

"""
tests for the RandomMover3D in random_movers.py
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import datetime
import numpy as np
import pytest

from gnome.movers.random_movers import RandomMover3D

from ..conftest import sample_sc_release
# some helpful parameters:

model_time = datetime.datetime(2014, 1, 8, 12)
time_step = 600 # 10 minutes in seconds


def test_init():
    """
    test the varios ways it can be initilized
    """
    mv = RandomMover3D()

    mv = RandomMover3D(vertical_diffusion_coef_above_ml=10.0) # cm^2/s

    mv = RandomMover3D(vertical_diffusion_coef_below_ml=2.0) # cm^2/s

    mv = RandomMover3D(mixed_layer_depth=20) # m

    assert True


def test_horizontal_zero():
    """
    checks that there is no horizontal movement
    """
    mv = RandomMover3D() # all defaults

    num_elements = 100

    sc = sample_sc_release(num_elements=num_elements,
                           start_pos=(0.0, 0.0, 0.0),
                           release_time=model_time,
                           )
    # set z positions:
    sc['positions'][:, 2] = np.linspace(0, 50, num_elements)

    mv.horizontal_diffusion_coef_above_ml = 0;
    mv.horizontal_diffusion_coef_below_ml = 0;

    delta = mv.get_move(sc,
                        time_step,
                        model_time,
                        )

    print(delta)

    assert np.alltrue(delta[:, 0:2] == 0.0)


@pytest.mark.skipif(True, reason="changed algorithm, needs update")
def test_vertical_zero():
    """
    checks that there is no vertical movement
    """
    mv = RandomMover3D(mixed_layer_depth=10.0) # mostly defaults

    num_elements = 100

    sc = sample_sc_release(num_elements=num_elements,
                           start_pos=(0.0, 0.0, 0.0),
                           release_time=model_time,
                           )
    # set z positions:
    sc['positions'][:, 2] = np.linspace(0, 50, num_elements)
    print(sc['positions'])

    mv.vertical_diffusion_coef_above_ml = 0.0
    mv.vertical_diffusion_coef_below_ml = 0.0

    delta = mv.get_move(sc,
                        time_step,
                        model_time,
                        )

    print(delta)

    assert not np.alltrue(delta[:, 2] == 0.0)


def test_bottom_layer():
    """
    tests the bottom layer

    elements should vertically spread according to the diffusion coef of the bottom layer
    """
    D_lower = .11 #m^2/s (or cm^2/s?)
    time_step = 60
    total_time = 60000

    num_elements = 100
    num_timesteps = total_time // time_step

    mv = RandomMover3D(vertical_diffusion_coef_below_ml=D_lower) # m

    sc = sample_sc_release(num_elements=num_elements,
                           start_pos=(0.0, 0.0, 0.0),
                           release_time=model_time,
                           )
    # re-set z positions:
    sc['positions'][:, 2] = 1000.0 # far from the  top
#    sc['positions'][ :num_elements/2 , 2] =  5.0 # near top
#    sc['positions'][  num_elements/2:, 2] = 50.0 # down deep

    # call get_move a bunch of times
    for i in range(num_timesteps):
#        print "positions:\n", sc['positions']
        delta = mv.get_move(sc,
                            time_step,
                            model_time,
                            )
#        print "delta:\n", delta
        sc['positions'] += delta

#    print sc['positions']

    # expected variance:
    exp_var = 2 * num_timesteps*time_step * D_lower # in cm^2?
    exp_var /= 10**4 # convert to m
    var = sc['positions'][:,2].var()
    print("expected_var:", exp_var, "var:", var)

    assert np.allclose(exp_var, var, rtol=0.25)


def test_mixed_layer():
    """
    tests the top layer

    elements should vertically spread according to the diffusion coef of the
    top layer. This version uses a really thick mixed layer to stay away from
    boundary effects

    """
    D_mixed = 10.0 # cm^2/s
    time_step = 60
    total_time = 600

    num_elements = 1000
    num_timesteps = total_time // time_step

    mv = RandomMover3D(vertical_diffusion_coef_above_ml=D_mixed,
                             mixed_layer_depth=1000, # HUGE to avoid surface effects.
                             ) # m

    sc = sample_sc_release(num_elements=num_elements,
                           start_pos=(0.0, 0.0, 0.0),
                           release_time=model_time,
                           )
    # re-set z positions:
    sc['positions'][:, 2] =  500.0 # far from the  top

    # call get_move a bunch of times
    for i in range(num_timesteps):
        # print "positions:\n", sc['positions']
        delta = mv.get_move(sc,
                            time_step,
                            model_time,
                            )
#        print "delta:\n", delta
        sc['positions'] += delta

#    print sc['positions']

    # expected variance:
    exp_var = 2 * num_timesteps*time_step * D_mixed # in cm^2?
    exp_var /= 10**4 # convert to m
    var = sc['positions'][:,2].var()
    print("expected_var:", exp_var, "var:", var)

    assert np.allclose(exp_var, var, rtol=0.1)


def test_mixed_layer2():
    """
    tests the top layer

    elements should end up fairly evenly distributed by the end of along run

    """
    D_mixed = 10.0 # cm^2/s
    mixed_layer_depth = 10.0 # m
    time_step = 60
    total_time = 6000

    num_elements = 1000
    num_timesteps = total_time // time_step

    mv = RandomMover3D(vertical_diffusion_coef_above_ml=D_mixed,
                             vertical_diffusion_coef_below_ml=0.0,
                             mixed_layer_depth=mixed_layer_depth,
                             ) # m

    sc = sample_sc_release(num_elements=num_elements,
                           start_pos=(0.0, 0.0, 0.0),
                           release_time=model_time,
                           )
    # re-set z positions:
    sc['positions'][:, 2] =  5.0 # middle of the layer

    # call get_move a bunch of times
    for i in range(num_timesteps):
        # print "positions:\n", sc['positions']
        delta = mv.get_move(sc,
                            time_step,
                            model_time,
                            )
#        print "delta:\n", delta
        sc['positions'] += delta

#    print sc['positions']

    # expected mean
    # exp_mean = 5.0 # middle of layer

    # expected variance:
    exp_var = mixed_layer_depth**2 / 12.0
    var = sc['positions'][:, 2].var()
    print("expected_var:", exp_var, "var:", var)

    assert np.allclose(exp_var, var, rtol=0.18)



