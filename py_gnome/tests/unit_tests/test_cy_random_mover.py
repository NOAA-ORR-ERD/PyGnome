"""
unit tests cython wrapper

designed to be run with py.test
"""

import numpy as np

from gnome.basic_types import spill_type, world_point, world_point_type

from gnome.cy_gnome.cy_helpers import srand
from gnome.cy_gnome.cy_random_mover import CyRandomMover
import cy_fixtures

import pytest


def test_exceptions():
    """
    Test ValueError exception thrown if improper input arguments
    """

    with pytest.raises(ValueError):
        CyRandomMover(diffusion_coef=-.0001)

    with pytest.raises(ValueError):
        CyRandomMover(uncertain_factor=.5)


class TestRandom:

    cm = cy_fixtures.CyTestMove()
    rm = CyRandomMover(diffusion_coef=100000)

    def move(self, delta):
        self.rm.prepare_for_model_run()

        self.rm.prepare_for_model_step(self.cm.model_time,
                self.cm.time_step)
        self.rm.get_move(
            self.cm.model_time,
            self.cm.time_step,
            self.cm.ref,
            delta,
            self.cm.status,
            spill_type.forecast,
            )

    def test_move(self):
        """
        test that it moved
        """

        self.move(self.cm.delta)
        np.set_printoptions(precision=4)

        print
        print 'diffusion_coef = {0:0.1f}'.format(self.rm.diffusion_coef),
        print 'get_move output:'
        print self.cm.delta.view(dtype=np.float64).reshape(-1, 3)

        assert np.all(self.cm.delta['lat'] != 0)
        assert np.all(self.cm.delta['long'] != 0)

    def test_zero_coef(self):
        """
        ensure no move for 0 diffusion coefficient
        """

        self.rm.diffusion_coef = 0
        new_delta = np.zeros((self.cm.num_le, ), dtype=world_point)
        self.move(new_delta)
        self.rm.diffusion_coef = 100000
        assert np.all(new_delta.view(dtype=np.double).reshape(1, -1)
                      == 0)

    def test_update_coef(self):
        """
        Test that the move is different from original move since
        diffusion coefficient is different
        Use the py.test -s flag to view the difference between the two
        """

        np.set_printoptions(precision=6)
        delta = np.zeros((self.cm.num_le, ), dtype=world_point)
        self.move(delta)  # get the move before changing the coefficient

        print
        print 'diffusion_coef = {0.diffusion_coef}'.format(self.rm),
        print 'get_move output:'
        print delta.view(dtype=np.float64).reshape(-1, 3)
        self.rm.diffusion_coef = 10
        assert self.rm.diffusion_coef == 10

        srand(1)
        new_delta = np.zeros((self.cm.num_le, ), dtype=world_point)
        self.move(new_delta)  # get the move after changing coefficient
        print
        print 'diffusion_coef = {0.diffusion_coef}'.format(self.rm),
        print 'get_move output:'
        print new_delta.view(dtype=np.float64).reshape(-1, 3)
        print
        print '-- Norm of difference between movement vector --'
        print self._diff(delta, new_delta).reshape(-1, 1)
        assert np.all(delta['lat'] != new_delta['lat'])
        assert np.all(delta['long'] != new_delta['long'])

        self.rm.diffusion_coef = 100000  # reset it

    def test_seed(self):
        """
        Since seed is not reset, the move should be repeatable
        """

        delta = np.zeros((self.cm.num_le, ), dtype=world_point)
        self.move(delta)

        srand(1)
        new_delta = np.zeros((self.cm.num_le, ), dtype=world_point)
        self.move(new_delta)

        print
        print '-- Do not reset seed and call get move again to get identical results --'
        print 'get_move results 1st time:'
        print delta.view(dtype=np.float64).reshape(-1, 3)
        print 'get_move results 2nd time - same seed:'
        print new_delta.view(dtype=np.float64).reshape(-1, 3)
        print
        print '-- Norm of difference between movement vector --'
        print self._diff(delta, new_delta)
        print delta['lat'], new_delta['lat']
        assert np.all(delta['lat'] == new_delta['lat'])
        assert np.all(delta['long'] == new_delta['long'])

    def _diff(self, delta, new_delta):
        """
        gives the norm of the (delta-new_delta)
        """

        diff = delta.view(dtype=world_point_type).reshape(-1, 3) \
            - new_delta.view(dtype=world_point_type).reshape(-1, 3)
        return np.sum(diff ** 2, axis=1) ** .5


if __name__ == '__main__':
    tr = TestRandom()

    # tr.test_move()
    # tr.test_update_coef()

    tr.test_seed()
