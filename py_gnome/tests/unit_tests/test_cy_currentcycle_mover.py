"""
unit tests cython wrapper

designed to be run with py.test
"""

import os
import numpy as np

import gnome
from gnome.basic_types import world_point, status_code_type, \
    spill_type, oil_status

from gnome.cy_gnome.cy_currentcycle_mover import CyCurrentCycleMover
from gnome.cy_gnome import cy_shio_time

from gnome.utilities import time_utils
from gnome.utilities.remote_data import get_datafile

import datetime

import pytest

here = os.path.dirname(__file__)
cur_dir = os.path.join(here, 'sample_data', 'currents')
tide_dir = os.path.join(here, 'sample_data', 'tides')


# def test_exceptions():
#     """
#     Test ValueError exception thrown if improper input arguments
#     """
#     with pytest.raises(ValueError):
#         cy_currentcycle_mover.CyCurrentCycleMover()
#

class Common:

    """
    test setting up and moving four particles

    Base class that initializes stuff that is common for multiple
    cy_currentcycle_mover objects
    """

    # ################
    # create arrays #
    # ################

    num_le = 4  # test on 4 LEs
    ref = np.zeros((num_le, ), dtype=world_point)
    delta = np.zeros((num_le, ), dtype=world_point)
    delta_uncertainty = np.zeros((num_le, ), dtype=world_point)
    status = np.empty((num_le, ), dtype=status_code_type)

    time_step = 900

    def __init__(self):
        time = datetime.datetime(2014, 6, 9, 0)
        self.model_time = time_utils.date_to_sec(time)
        self.time_step = 360       

        # ###############
        # init. arrays #
        # ###############

        self.ref[:] = 1.
        self.ref[:]['z'] = 0  # on surface by default
        self.status[:] = oil_status.in_water


@pytest.mark.slow
class TestCurrentCycleMover:

    cm = Common()
    ccm = CyCurrentCycleMover()

    # delta = np.zeros((cm.num_le,), dtype=world_point)

    def move(self):
        self.ccm.prepare_for_model_run()

        print "Certain move"
        self.ccm.prepare_for_model_step(self.cm.model_time,
                self.cm.time_step)
        self.ccm.get_move(
            self.cm.model_time,
            self.cm.time_step,
            self.cm.ref,
            self.cm.delta,
            self.cm.status,
            spill_type.forecast,
            )

    def move_uncertain(self):
        self.ccm.prepare_for_model_run()
        spill_size = np.zeros((1, ), dtype=np.int32)  # number of LEs in 1 uncertainty spill - simple test   
        spill_size[0] = self.cm.num_le  # for uncertainty spills
        start_pos=(-76.149368,37.74496,0)

        print "Uncertain move"
        self.ccm.prepare_for_model_step(self.cm.model_time,
                self.cm.time_step, 1, spill_size)
        self.ccm.get_move(
            self.cm.model_time,
            self.cm.time_step,
            self.cm.ref,
            self.cm.delta_uncertainty,
            self.cm.status,
            spill_type.uncertainty,
            )

    def check_move(self):
        self.move()
        print self.cm.delta
        assert np.all(self.cm.delta['lat'] != 0)
        assert np.all(self.cm.delta['long'] != 0)

    def check_move_uncertain(self):
        self.move_uncertain()
        print self.cm.delta_uncertainty
        assert np.all(self.cm.delta_uncertainty['lat'] != 0)
        assert np.all(self.cm.delta_uncertainty['long'] != 0)

    def check_move_certain_uncertain(self,uncertain_time_delay=0):
        self.check_move()
        self.check_move_uncertain()
        tol = 1e-5
        msg = r"{0} move is not within a tolerance of {1}"
        if uncertain_time_delay == 0:
            assert np.all(self.cm.delta_uncertainty['lat'] != self.cm.delta['lat'])
            assert np.all(self.cm.delta_uncertainty['long'] != self.cm.delta['long'])
        if uncertain_time_delay > 0:
            np.testing.assert_allclose(
            self.cm.delta['lat'],
            self.cm.delta_uncertainty['lat'],
            tol,
            tol,
            msg.format('current_cycle.nc', tol),
            0,
            )
            np.testing.assert_allclose(
            self.cm.delta['long'],
            self.cm.delta_uncertainty['long'],
            tol,
            tol,
            msg.format('current_cycle.nc', tol),
            0,
            )

    def test_move_tri(self):
        """
        test move for a triangular grid (first time in file)
        """

        time = datetime.datetime(2014, 6, 9, 0)
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = get_datafile(os.path.join(cur_dir, 'PQBayCur.nc4'))
        topology_file = get_datafile(os.path.join(cur_dir,
                r'PassamaquoddyTOP.dat'))

        self.ccm.text_read(time_grid_file, topology_file)

        # self.ccm.export_topology(topology_file2)

        self.cm.ref[:]['long'] = -66.991344  # for Passamaquoddy
        self.cm.ref[:]['lat'] = 45.059316
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = .00020319
        actual[:]['long'] = -.0001276599
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(
            self.cm.delta['lat'],
            actual['lat'],
            tol,
            tol,
            msg.format('PQBayCur.nc4', tol),
            0,
            )
        np.testing.assert_allclose(
            self.cm.delta['long'],
            actual['long'],
            tol,
            tol,
            msg.format('PQBayCur.nc4', tol),
            0,
            )

    def test_move_tri_tide(self):
        """
        test move for a triangular grid (first time in file)
        """

        time = datetime.datetime(2014, 6, 9, 0)
        self.cm.model_time = time_utils.date_to_sec(time)
        self.cm.uncertain = True

        time_grid_file = get_datafile(os.path.join(cur_dir, 'PQBayCur.nc4'
                ))
        topology_file = get_datafile(os.path.join(cur_dir, 'PassamaquoddyTOP.dat'
                ))

        tide_file = get_datafile(os.path.join(tide_dir, 'EstesHead.txt'
                ))

        yeardata_path = os.path.join(os.path.dirname(gnome.__file__),
                'data/yeardata/')

        self.shio = cy_shio_time.CyShioTime(tide_file)
        self.ccm.set_shio(self.shio)
        self.ccm.text_read(time_grid_file, topology_file)
        self.shio.set_shio_yeardata_path(yeardata_path)
        self.cm.ref[:]['long'] = -66.991344  # for Passamaquoddy
        self.cm.ref[:]['lat'] = 45.059316
        #self.check_move()
        self.check_move_certain_uncertain(self.ccm.uncertain_time_delay)

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = -.000440779
        actual[:]['long'] = .00016611
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(
            self.cm.delta['lat'],
            actual['lat'],
            tol,
            tol,
            msg.format('ches_bay', tol),
            0,
            )
        np.testing.assert_allclose(
            self.cm.delta['long'],
            actual['long'],
            tol,
            tol,
            msg.format('ches_bay', tol),
            0,
            )
        #check that certain and uncertain are the same if uncertainty is time delayed
        #self.ccm.uncertain_time_delay = 3
        self.ccm.uncertain_time_delay = 10800 # cython expects time_delay in seconds
        self.check_move_certain_uncertain(self.ccm.uncertain_time_delay)



if __name__ == '__main__':
    tcc = TestCurrentCycleMover()
    tcc.test_move_tri()
    tcc.test_move_tri_tide()
