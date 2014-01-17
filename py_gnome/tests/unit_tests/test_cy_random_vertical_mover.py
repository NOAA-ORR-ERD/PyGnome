"""
unit tests cython wrapper

designed to be run with py.test
"""

import numpy as np

from gnome.basic_types import spill_type, world_point, world_point_type

from gnome.cy_gnome.cy_helpers import srand
from gnome.cy_gnome.cy_random_vertical_mover import CyRandomVerticalMover
import cy_fixtures

import pytest


def test_exceptions():
    """
    Test ValueError exception thrown if improper input arguments
    """

    with pytest.raises(ValueError):
        CyRandomVerticalMover(vertical_diffusion_coef_above_ml=-1000)


class TestRandomVertical:

    msg = 'vertical_diffusion_coef_above_ml = {0.vertical_diffusion_coef_above_ml},vertical_diffusion_coef_below_ml = {0.vertical_diffusion_coef_below_ml}'
    cm = cy_fixtures.CyTestMove()
    cm.ref['z'][:]=.1
    rm = CyRandomVerticalMover(vertical_diffusion_coef_above_ml=5,vertical_diffusion_coef_below_ml=.11)

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
        print self.msg.format(self.rm),
        print 'get_move output:'
        print self.cm.delta.view(dtype=np.float64).reshape(-1, 3)
        assert np.all(self.cm.delta['z'] != 0)

    def test_zero_coef(self):
        """
        ensure no move for 0 vertical diffusion coefficient
        """

        self.rm.vertical_diffusion_coef_above_ml = 0
        self.rm.vertical_diffusion_coef_below_ml = 0
        new_delta = np.zeros((self.cm.num_le, ), dtype=world_point)
        self.move(new_delta)
        self.rm.vertical_diffusion_coef_above_ml = 5
        self.rm.vertical_diffusion_coef_below_ml = .11
        assert np.all(new_delta.view(dtype=np.double).reshape(1, -1)
                      == 0)

    def test_surface_particles(self):
        """
        ensure no move for surface particles
        """

    	self.cm.ref['z'][:]=0
        new_delta = np.zeros((self.cm.num_le, ), dtype=world_point)
        self.move(new_delta)
    	self.cm.ref['z'][:]=.1
        assert np.all(new_delta.view(dtype=np.double).reshape(1, -1)
                      == 0)

    def test_mixed_layer_zero(self):
        """
        Test that the move for z above the mixed layer
        is different from move for z below the mixed layer
        and the same if the mixed layer is zero
        Use the py.test -s flag to view the differences
        """

        srand(1)
        np.set_printoptions(precision=6)
        delta = np.zeros((self.cm.num_le, ), dtype=world_point)
        self.move(delta)  # get the move before changing the coefficient
        print
        print self.msg.format(self.rm) + ' get_move output:'
        print delta.view(dtype=np.float64).reshape(-1, 3)

        srand(1)
        self.cm.ref['z'][:]=20
        new_delta = np.zeros((self.cm.num_le, ), dtype=world_point)
        self.move(new_delta)  # get the move after changing coefficient
        print
        print self.msg.format(self.rm) + ' get_move output:'
        print new_delta.view(dtype=np.float64).reshape(-1, 3)
        print
        assert np.all(delta['z'] != new_delta['z'])

        print '-- Norm of difference between movement vector --'
        print self._diff(delta, new_delta).reshape(-1, 1)

        self.rm.mixed_layer_depth = 0

        srand(1)
    	self.cm.ref['z'][:]=.1
        newer_delta = np.zeros((self.cm.num_le, ), dtype=world_point)
        self.move(newer_delta)  # get the move after changing mld
        print
        print self.msg.format(self.rm) + ' get_move output:'
        print newer_delta.view(dtype=np.float64).reshape(-1, 3)
        print

        msg = r"{0} move is not within a tolerance of {1}"
        tol = 1e-10
        np.testing.assert_allclose(
            new_delta['z'],
            newer_delta['z'],
            tol,
            tol,
            msg.format('random_vertical', tol),
            0,
            )

        print '-- Norm of difference between movement vector --'
        print self._diff(new_delta, newer_delta).reshape(-1, 1)

        self.rm.mixed_layer_depth = 10

    def test_update_coef(self):
        """
        Test that the move is different from original move since
        vertical diffusion coefficient is different
        Use the py.test -s flag to view the difference between the two
        """

        np.set_printoptions(precision=6)
        delta = np.zeros((self.cm.num_le, ), dtype=world_point)
        self.move(delta)  # get the move before changing the coefficient
        print
        print self.msg.format(self.rm) + ' get_move output:'
        print delta.view(dtype=np.float64).reshape(-1, 3)
        self.rm.vertical_diffusion_coef_above_ml = 10
        assert self.rm.vertical_diffusion_coef_above_ml == 10

        srand(1)
        new_delta = np.zeros((self.cm.num_le, ), dtype=world_point)
        self.move(new_delta)  # get the move after changing coefficient
        print
        print self.msg.format(self.rm) + ' get_move output:'
        print new_delta.view(dtype=np.float64).reshape(-1, 3)
        print
        print '-- Norm of difference between movement vector --'
        print self._diff(delta, new_delta).reshape(-1, 1)

        assert np.all(delta['z'] != new_delta['z'])

        self.rm.vertical_diffusion_coef_above_ml = 5  # reset it

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
        print '-- Do not reset seed and call get move again',
        print 'to get identical results --'
        print 'get_move results 1st time:'
        print delta.view(dtype=np.float64).reshape(-1, 3)
        print 'get_move results 2nd time - same seed:'
        print new_delta.view(dtype=np.float64).reshape(-1, 3)
        print
        print '-- Norm of difference between movement vector --'
        print self._diff(delta, new_delta)
        assert np.all(delta['lat'] == new_delta['lat'])
        assert np.all(delta['long'] == new_delta['long'])

    def _diff(self, delta, new_delta):
        """
        gives the norm of the (delta-new_delta)
        """

        diff = delta.view(dtype=world_point_type).reshape(-1, 3)
        diff -= new_delta.view(dtype=world_point_type).reshape(-1, 3)
        return np.sum(diff ** 2, axis=1) ** .5


if __name__ == '__main__':
    tr = TestRandomVertical()

    # tr.test_move()
    # tr.test_update_coef()

    tr.test_seed()
