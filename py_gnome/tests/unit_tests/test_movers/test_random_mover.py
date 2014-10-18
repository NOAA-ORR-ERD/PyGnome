"""
unittests for random mover

designed to be run with py.test
"""

import datetime
import numpy as np
import os

from gnome.movers import RandomMover

from gnome.utilities.time_utils import sec_to_date, date_to_sec
from gnome.utilities.projections import FlatEarthProjection
from gnome.persist import load
from conftest import sample_sc_release

import pytest


def test_exceptions():
    """
    Test ValueError exception thrown if improper input arguments
    """

    #with pytest.raises(ValueError):
        #RandomMover(diffusion_coef=0)

    with pytest.raises(ValueError):
        RandomMover(diffusion_coef=-1000)

    with pytest.raises(ValueError):
        RandomMover(uncertain_factor=0)


class TestRandomMover:

    """
    gnome.RandomMover() test

    """

    num_le = 5

    # start_pos = np.zeros((num_le,3), dtype=basic_types.world_point_type)

    start_pos = (0., 0., 0.)
    rel_time = datetime.datetime(2012, 8, 20, 13)  # yyyy/month/day/hr/min/sec
    model_time = sec_to_date(date_to_sec(rel_time) + 1)
    time_step = 15 * 60  # seconds

    mover = RandomMover()

    def reset_pos(self):
        self.pSpill['positions'] = (0., 0., 0.)
        print self.pSpill['positions']

    def test_string_representation_matches_repr_method(self):
        """
        Just print repr and str
        """

        print
        print repr(self.mover)
        print str(self.mover)
        assert True

    def test_id_matches_builtin_id(self):

        # It is not a good assumption that the obj.id property
        # will always contain the id(obj) value.  For example it could
        # have been overloaded with, say, a uuid1() generator.
        # assert id(self.mover) == self.mover.id

        pass

    def test_change_diffusion_coef(self):
        self.mover.diffusion_coef = 200000
        assert self.mover.diffusion_coef == 200000

    def test_change_uncertain_factor(self):
        self.mover.uncertain_factor = 3
        assert self.mover.uncertain_factor == 3

    def test_prepare_for_model_step(self):
        """
        Simply tests the method executes without exceptions
        """

        pSpill = sample_sc_release(self.num_le, self.start_pos)
        self.mover.prepare_for_model_step(pSpill, self.time_step,
                self.model_time)
        assert True


start_locs = [(0., 0., 0.), (30.0, 30.0, 30.0), (-45.0, -60.0, 30.0)]

timesteps = [36, 360, 3600]

# timesteps = [36, ]

test_cases = [(loc, step) for loc in start_locs for step in timesteps]


@pytest.mark.parametrize(('start_loc', 'time_step'), test_cases)
def test_variance1(start_loc, time_step):
    """
    After a few timesteps the variance of the particle positions should be
    similar to the computed value: var = Dt
    """

    num_le = 1000
    start_time = datetime.datetime(2012, 11, 10, 0)
    sc = sample_sc_release(num_le, start_loc, start_time)
    D = 100000
    num_steps = 10

    rand = RandomMover(diffusion_coef=D)

    model_time = start_time
    for i in range(num_steps):
        model_time += datetime.timedelta(seconds=time_step)
        sc.release_elements(time_step, model_time)
        rand.prepare_for_model_step(sc, time_step, model_time)
        delta = rand.get_move(sc, time_step, model_time)

        # print "delta:", delta

        sc['positions'] += delta

        # print sc['positions']

    # compute the variances:
    # convert to meters

    pos = FlatEarthProjection.lonlat_to_meters(sc['positions'],
            start_loc)
    var = np.var(pos, axis=0)

    # D converted to meters^s/s

    expected = 2.0 * (D * 1e-4) * num_steps * time_step

    assert np.allclose(var, (expected, expected, 0.), rtol=0.1)


if __name__ == '__main__':
    tw = TestRandomMover()
    tw.test_prepare_for_model_step()
    tw.test_change_diffusion_coef()
