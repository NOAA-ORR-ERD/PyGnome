'''
Test classes in vertical_movers.py module
'''

from datetime import datetime

import pytest
import numpy as np

from gnome.movers import RiseVelocityMover
from gnome.array_types import rise_vel
from gnome.spill.elements import ElementType, InitRiseVelFromDist

from conftest import sample_sc_release


def test_init():
    """
    test initializes correctly
    """

    r = RiseVelocityMover()
    #assert r.water_density == 1020
    #assert r.water_viscosity == 1.e-6


def test_props():
    """
    test properties can be set
    """

    r = RiseVelocityMover()
    #r.water_density = 1
    #r.water_viscosity = 1.1e-6

    #assert r.water_density == 1
    #assert r.water_viscosity == 1.1e-6


time_step = 15 * 60  # seconds
rel_time = datetime(2012, 8, 20, 13)  # yyyy/month/day/hr/min/sec
sc = sample_sc_release(5, (3., 6., 0.),
                       rel_time,
                       uncertain=False,
                       arr_types={'rise_vel': rise_vel},
                       element_type=ElementType({'rise_vel':
                                                 InitRiseVelFromDist()}))
u_sc = sample_sc_release(5, (3., 6., 0.),
                         rel_time,
                         uncertain=True,
                         arr_types={'rise_vel': rise_vel},
                         element_type=ElementType({'rise_vel':
                                                 InitRiseVelFromDist()}))
model_time = rel_time


@pytest.mark.parametrize("test_sc", [sc, u_sc])
def test_one_move(test_sc):
    """
    calls one step for movement - just checks that it doesn't fail for any step
    Placeholder - get_move does not currently work since all data_arrays are
    not yet defined
    """

    r_vel = RiseVelocityMover()
    r_vel.prepare_for_model_run()

    r_vel.prepare_for_model_step(test_sc, time_step, model_time)
    delta_pos = r_vel.get_move(test_sc, time_step, model_time)
    r_vel.model_step_is_done()
    print "\ndelta_pos: "
    print delta_pos
    assert np.all(delta_pos[:, :2] == 0)
    assert np.all(delta_pos[:, 2] != 0)
