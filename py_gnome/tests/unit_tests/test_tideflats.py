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

import gnome.scripting as sc

from gnome.basic_types import oil_status

from gnome.model import Model

from gnome.movers import (c_GridWindMover,
                          RandomMover,
                          constant_point_wind_mover,
                          point_wind_mover_from_file)

from gnome.spills import surface_point_line_spill

from .conftest import sample_sc_release, testdata

wind_file = testdata['c_GridWindMover']['wind_curv']
topology_file = testdata['c_GridWindMover']['top_curv']




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
    wm = constant_point_wind_mover(10, 270, units='m/s')

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

    print(delta)
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

    print("status codes")
    print(sc['status_codes'])

    delta = run_one_timestep(sc, rand, time_step, model_time)

    print("delta:", delta)

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


def test_gridded_wind():
    """
    does a gridded wind mover move the elements?
    """
    start_pos = (-123.57152, 37.369436, 0.0)
    rel_time = datetime(2006, 3, 31, 21, 0)
    time_step = 30 * 60  # seconds
    model_time = rel_time

    sc = sample_sc_release(10,
                           start_pos=start_pos,
                           release_time=rel_time,
                           windage_range=(0.01, 0.01),
                           )

    wm = c_GridWindMover(wind_file, topology_file)

    delta = run_one_timestep(sc, wm, time_step, model_time)

    assert np.all(delta[:, 0] == delta[0, 0])  # lat move matches for all LEs
    assert np.all(delta[:, 1] == delta[0, 1])  # lon move matches for all LEs

    # set the on_tideflat flag

    # now set the on_tideflat code on half the elements
    sc['status_codes'][5:] = oil_status.on_tideflat

    delta = run_one_timestep(sc, wm, time_step, model_time)

    # only the first 5 should move
    assert np.all(delta[:5, 0] != 0.0)
    assert np.all(delta[:5, 1] != 0.0)

    assert np.all(delta[5:, 0] == 0.0)
    assert np.all(delta[5:, 1] == 0.0)

    # no change to depth
    assert np.all(delta[:, 2] == 0.0)


def test_full_model_run():
    """
    This will have a couple movers, and make sure that the
    on_tideflat status code stops all movers.

    only winds and currents for now
    """
    start_time = "2018-09-10T12:00"

    model = Model(start_time=start_time,
                  time_step=sc.minutes(10),
                  )

#                   start_time=round_time(datetime.now(), 3600),

#                   duration=timedelta(days=1),
#                  weathering_substeps=1,
#                  map=None,
#                  uncertain=False,
#                  cache_enabled=False,
#                  mode=None,
#                  location=[],
#                  environment=[],
#                  outputters=[],
#                  movers=[],
#                  weatherers=[],
#                  spills=[],
#                  uncertain_spills=[],
#                  **kwargs):
# )

    model.movers += constant_point_wind_mover(speed=10,
                                        direction=225,
                                        units='m/s')

    model.movers += RandomMover()  # defaults are fine

    model.spills += surface_point_line_spill(num_elements=10,
                                             start_position=(0.0, 0.0, 0.0),
                                             release_time=start_time,
                                             )

    # fixme: should maybe add currents -- though if the currents are smart,
    #        they should be zero on the tideflats

    # run one step:
    print(model.step())
    # step zero -- should be released, but not yet moved
    positions = model.get_spill_property('positions').copy()
    assert np.all(positions == 0.0)
    prev_positions = positions

    print(model.step())
    # step one -- all elements should have moved in horizontal
    positions = model.get_spill_property('positions').copy()
    assert np.all(positions[:, 0:1] != prev_positions[:, 0:1])
    assert np.all(positions[:, 2] == prev_positions[:, 2])
    prev_positions = positions


    # Now set the flags for half of the elements
    status_codes = model.get_spill_property('status_codes')
    new_status_codes = np.array([oil_status.on_tideflat] * 5 +
                                [oil_status.in_water] * 5)
    status_codes[:] = new_status_codes

    # make sure it took
    assert np.all(model.get_spill_property('status_codes') ==
                  new_status_codes)

    # step the model again
    model.step()
    positions = model.get_spill_property('positions').copy()
    delta = positions - prev_positions
    # first five should not have moved
    assert np.all(delta[:5, 0:1] == 0.0)
    assert np.all(delta[5:, 0:1] != 0.0)
    prev_positions = positions

    # reset status codes to in_water:
    model.get_spill_property('status_codes')[:] = oil_status.in_water

    # step the model again
    print(model.step())
    positions = model.get_spill_property('positions').copy()
    delta = positions - prev_positions
    # they all should have moved again
    assert np.all(delta[:, 0:1] != 0.0)



