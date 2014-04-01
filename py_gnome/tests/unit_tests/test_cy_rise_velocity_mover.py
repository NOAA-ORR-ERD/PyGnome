"""
unit tests cython wrapper

designed to be run with py.test
"""

import numpy as np
import pytest

from gnome.basic_types import world_point, world_point_type, spill_type

from gnome.cy_gnome.cy_helpers import srand
from gnome.cy_gnome.cy_rise_velocity_mover import CyRiseVelocityMover
from cy_fixtures import CyTestMove


# def test_exceptions():
#     """
#     Test ValueError exception thrown if improper input arguments
#     """
# 
#     with pytest.raises(ValueError):
#         CyRiseVelocityMover(water_density=0, water_viscosity=0)


class TestRiseVelocity:

    cm = CyTestMove()
    #rv = CyRiseVelocityMover(water_density=1020,
    #                        water_viscosity=.000001)
    rv = CyRiseVelocityMover()

    # set these values and try with NaNs
    # rise_velocity = np.zeros((cm.num_le,), dtype=np.double)

    rise_velocity = np.linspace(0.01, 0.04, cm.num_le)

    def move(self, delta):
        self.rv.prepare_for_model_run()

        self.rv.prepare_for_model_step(self.cm.model_time,
                self.cm.time_step)
        self.rv.get_move(
            self.cm.model_time,
            self.cm.time_step,
            self.cm.ref,
            delta,
            self.rise_velocity,
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

        # print '''rise_velocity  = {0:0.1f} get_move output:
        #      '''.format(self.rise_velocity)

        print 'rise_velocity  = '
        print self.rise_velocity
        print 'get_move output:'
        print self.cm.delta.view(dtype=np.float64).reshape(-1, 3)
        assert np.all(self.cm.delta['z'] != 0)

    def test_update_coef(self):
        """
        Test that the move is different from original move
        since rise velocity is different
        Use the py.test -s flag to view the difference between the two
        """

        # cy_helpers.srand(1)  # this happens in conftest.py before every test

        np.set_printoptions(precision=6)
        delta = np.zeros((self.cm.num_le, ), dtype=world_point)
        self.move(delta)  # get the move before changing the coefficient

        print
        print 'rise_velocity  = '
        print self.rise_velocity
        print 'get_move output:'
        print delta.view(dtype=np.float64).reshape(-1, 3)
        self.rise_velocity = np.linspace(0.02, 0.05, self.cm.num_le)
        print 'rise_velocity  = '
        print self.rise_velocity

        srand(1)
        new_delta = np.zeros((self.cm.num_le, ), dtype=world_point)
        self.move(new_delta)  # get the move after changing coefficient
        print

        print 'get_move output:'
        print new_delta.view(dtype=np.float64).reshape(-1, 3)
        print
        print '-- Norm of difference between movement vector --'
        print self._diff(delta, new_delta).reshape(-1, 1)
        assert np.all(delta['z'] != new_delta['z'])

    def test_zero_rise_velocity(self):
        """
        ensure no move for 0 rise velocity
        """

        self.rise_velocity = np.zeros((self.cm.num_le, ),
                dtype=np.double)
        print 'rise_velocity  = '
        print self.rise_velocity
        new_delta = np.zeros((self.cm.num_le, ), dtype=world_point)
        self.move(new_delta)
        print new_delta.view(dtype=np.float64).reshape(-1, 3)
        assert np.all(new_delta.view(dtype=np.double).reshape(1, -1)
                      == 0)

    def _diff(self, delta, new_delta):
        """
        gives the norm of the (delta-new_delta)
        """

        diff = delta.view(dtype=world_point_type).reshape(-1, 3)
        diff -= new_delta.view(dtype=world_point_type).reshape(-1, 3)
        return np.sum(diff ** 2, axis=1) ** .5


if __name__ == '__main__':
    tr = TestRiseVelocity()

    # tr.test_move()
    # tr.test_update_coef()
