#!/usr/bin/env python

"""
tests for the on_tideflat status code

winds and diffusion shouldn't move the elements

NOTE: this doesn't test whether the on_tideflat code is
      set (or unset) properly, but rather than it is used
      properly
"""

from datetime import datetime

import numpy as np

from gnome.basic_types import oil_status

from gnome.movers import (WindMover,
                          RandomMover,
                          constant_wind_mover,
                          wind_mover_from_file)

from .conftest import sample_sc_release

# sample_sc_release(num_elements=10,
#                       start_pos=(0.0, 0.0, 0.0),
#                       release_time=datetime(2000, 1, 1, 1),
#                       uncertain=False,
#                       time_step=360,
#                       spill=None,
#                       element_type=None,
#                       current_time=None,
#                       arr_types=None,
#                       windage_range=None,
#                       units='g',
#                       amount_per_element=1.0)


def run_one_timestep(pSpill, mover, time_step, model_time):
    mover.prepare_for_model_run()
    mover.prepare_for_model_step(pSpill, time_step, model_time)
    delta = mover.get_move(pSpill, time_step, model_time)
    mover.model_step_is_done()

    return delta


def test_constant_wind():
    """
    make sure wind doesn't move the LEs marked as on_tideflat
    """
    wm = constant_wind_mover(10, 270, units='m/s')

    windage = 0.03
    sc = sample_sc_release(10,  # ten elements
                           windage_range=(windage, windage),
                           )
    time_step = 1000
    model_time = datetime(2018, 1, 1, 0)

    # make sure all is "normal"
    assert np.all(sc['status_codes'] == oil_status.in_water)

    delta = run_one_timestep(sc, wm, time_step, model_time)

    assert np.all(delta[:, 0] > 0.0)
    assert np.all(delta[:, 1] == 0.0)

    # now set the on_tideflat code on half the elements
    sc['status_codes'][5:] = oil_status.on_tideflat

    delta = run_one_timestep(sc, wm, time_step, model_time)

    print delta
    # west wind
    # only the first 5 should move
    assert np.all(delta[:5, 0] > 0.0)
    assert np.all(delta[5:, 0] == 0.0)
    assert np.all(delta[:, 1] == 0.0)


def test_random_mover():
    """
    Make sure diffusion doesn't move the LEs marked as on_tideflat
    """
    start_time = datetime(2012, 11, 10, 0)
    sc = sample_sc_release(10,
                           (0.0, 0.0, 0.0),
                           start_time)
    D = 100000

    rand = RandomMover(diffusion_coef=D)


    model_time = start_time
    time_step = 1000  # quite random

    sc.release_elements(time_step, model_time)

    print "status codes"
    print sc['status_codes']

    delta = run_one_timestep(sc, rand, time_step, model_time)

    print "delta:", delta

    assert np.all(delta[:, 0] != 0.0)
    assert np.all(delta[:, 1] != 0.0)
    assert np.all(delta[:, 2] == 0.0)

    # now set the on_tideflat code on half the elements
    sc['status_codes'][5:] = oil_status.on_tideflat

    delta = run_one_timestep(sc, rand, time_step, model_time)

    # only the first 5 should move
    assert np.all(delta[:5, 0] != 0.0)
    assert np.all(delta[:5, 1] != 0.0)

    assert np.all(delta[5:, 0] == 0.0)
    assert np.all(delta[5:, 1] == 0.0)

    # no change to depth
    assert np.all(delta[:, 2] == 0.0)

