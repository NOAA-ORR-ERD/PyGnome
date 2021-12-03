"""
unit tests cython wrapper

designed to be run with py.test
"""





import os
import datetime

import numpy as np
import pytest

from gnome.basic_types import (world_point, status_code_type, spill_type,
                               oil_status)

from gnome.cy_gnome.cy_gridcurrent_mover import CyGridCurrentMover

from gnome.utilities import time_utils
from ..conftest import testdata


class Common(object):

    """
    test setting up and moving four particles

    Base class that initializes stuff that is common for multiple
    cy_gridcurrent_mover objects
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
        time = datetime.datetime(2012, 8, 20, 13)
        self.model_time = time_utils.date_to_sec(time)

        # ###############
        # init. arrays #
        # ###############

        self.ref[:] = 1.
        self.ref[:]['z'] = 0  # on surface by default
        self.status[:] = oil_status.in_water


def test_init():
    kwargs = {'current_scale': 2,
              'uncertain_duration': 10*3600,
              'uncertain_time_delay': 900,
              'uncertain_along': 0.75,
              'uncertain_cross': 0.5}
    gcm = CyGridCurrentMover(**kwargs)
    for key, val in kwargs.items():
        assert getattr(gcm, key) == val


@pytest.mark.slow
class TestGridCurrentMover(object):

    cm = Common()
    gcm = CyGridCurrentMover()

    # delta = np.zeros((cm.num_le,), dtype=world_point)

    def move(self):
        self.gcm.prepare_for_model_run()

        print("Certain move")
        self.gcm.prepare_for_model_step(self.cm.model_time, self.cm.time_step)
        self.gcm.get_move(
            self.cm.model_time,
            self.cm.time_step,
            self.cm.ref,
            self.cm.delta,
            self.cm.status,
            spill_type.forecast,
            )

    def move_uncertain(self):
        self.gcm.prepare_for_model_run()

        # number of LEs in 1 uncertainty spill - simple test
        spill_size = np.zeros((1, ), dtype=np.int32)

        spill_size[0] = self.cm.num_le  # for uncertainty spills
        start_pos = (-76.149368, 37.74496, 0)

        print("Uncertain move")
        self.gcm.prepare_for_model_step(self.cm.model_time, self.cm.time_step,
                                        1, spill_size)

        self.gcm.get_move(self.cm.model_time,
                          self.cm.time_step,
                          self.cm.ref,
                          self.cm.delta_uncertainty,
                          self.cm.status,
                          spill_type.uncertainty)

    def check_move(self):
        self.move()
        print(self.cm.delta)
        assert np.all(self.cm.delta['lat'] != 0)
        assert np.all(self.cm.delta['long'] != 0)

    def check_move_uncertain(self):
        self.move_uncertain()
        print(self.cm.delta_uncertainty)
        assert np.all(self.cm.delta_uncertainty['lat'] != 0)
        assert np.all(self.cm.delta_uncertainty['long'] != 0)

    def check_move_certain_uncertain(self, uncertain_time_delay=0):
        self.check_move()
        self.check_move_uncertain()
        tol = 1e-5
        msg = r"{0} move is not within a tolerance of {1}"

        if uncertain_time_delay == 0:
            assert np.all(self.cm.delta_uncertainty['lat'] != self.cm.delta['lat'])
            assert np.all(self.cm.delta_uncertainty['long'] != self.cm.delta['long'])

        if uncertain_time_delay > 0:
            np.testing.assert_allclose(self.cm.delta['lat'],
                                       self.cm.delta_uncertainty['lat'],
                                       tol, tol,
                                       msg.format('grid_current.nc', tol),
                                       0)

            np.testing.assert_allclose(self.cm.delta['long'],
                                       self.cm.delta_uncertainty['long'],
                                       tol, tol,
                                       msg.format('grid_current.nc', tol),
                                       0)

    def test_move_reg(self):
        """
        test move for a regular grid (first time in file)
        """

        time = datetime.datetime(1999, 11, 29, 21)
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = testdata['c_GridCurrentMover']['curr_reg']

        self.gcm.text_read(time_grid_file)
        self.cm.ref[:]['long'] = 3.104588  # for simple example
        self.cm.ref[:]['lat'] = 52.016468
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = .003354610952486354
        actual[:]['long'] = .0010056182923228838
        actual[:]['z'] = 0.
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'],
                                   tol, tol,
                                   msg.format('test.cdf', tol),
                                   0)

        np.testing.assert_allclose(self.cm.delta['long'], actual['long'],
                                   tol, tol,
                                   msg.format('test.cdf', tol),
                                   0)

        # np.testing.assert_equal(self.cm.delta['z'], actual['z'],
        #                        "test_move_reg() failed", 0)

        np.all(self.cm.delta['z'] == 0)

    def test_move_curv(self):
        """
        test move for a curvilinear grid (first time in file)
        """

        time = datetime.datetime(2008, 1, 29, 17)
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = testdata['c_GridCurrentMover']['curr_curv']
        topology_file = testdata['c_GridCurrentMover']['top_curv']

        self.gcm.text_read(time_grid_file, topology_file)

        # self.gcm.export_topology(topology_file2)

        self.cm.ref[:]['long'] = -74.03988  # for NY
        self.cm.ref[:]['lat'] = 40.536092
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = .000911
        actual[:]['long'] = -.001288
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'],
                                   tol, tol,
                                   msg.format('ny_cg.nc', tol),
                                   0)

        np.testing.assert_allclose(self.cm.delta['long'], actual['long'],
                                   tol, tol,
                                   msg.format('ny_cg.nc', tol),
                                   0)

    def test_move_curv_no_top(self):
        """
        test move for a curvilinear grid (first time in file)
        """

        time = datetime.datetime(2008, 1, 29, 17)
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = testdata['c_GridCurrentMover']['curr_curv']
        self.gcm.text_read(time_grid_file, topology_file=None)

        topology_file2 = os.path.join(os.path.split(time_grid_file)[0],
                                      'NYTopologyNew.dat')

        self.gcm.export_topology(topology_file2)
        self.cm.ref[:]['long'] = -74.03988  # for NY
        self.cm.ref[:]['lat'] = 40.536092
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = .000911
        actual[:]['long'] = -.001288
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'],
                                   tol, tol,
                                   msg.format('ny_cg.nc', tol),
                                   0)

        np.testing.assert_allclose(self.cm.delta['long'], actual['long'],
                                   tol, tol,
                                   msg.format('ny_cg.nc', tol),
                                   0)

    def test_move_curv_series(self):
        """
        Test a curvilinear file series
        - time in first file
        - time in second file
        """

        # time = datetime.datetime(2009, 8, 2, 0)  # first file

        time = datetime.datetime(2009, 8, 9, 0)  # second file
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = testdata['c_GridCurrentMover']['series_curv']
        topology_file = testdata['c_GridCurrentMover']['series_top']

        self.gcm.text_read(time_grid_file, topology_file)
        #self.gcm.text_read(time_grid_file)
        self.cm.ref[:]['long'] = -157.795728  # for HiROMS
        self.cm.ref[:]['lat'] = 21.069288
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        # actual[:]['lat'] = -.003850193  # file 2
        # actual[:]['long'] = .000152012

        # updated to new curvilinear algorithm
        actual[:]['lat'] = .00292  # file 2
        actual[:]['long'] = .00051458
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'],
                                   tol, tol,
                                   msg.format('HiROMS', tol),
                                   0)

        np.testing.assert_allclose(self.cm.delta['long'], actual['long'],
                                   tol, tol,
                                   msg.format('HiROMS', tol),
                                   0)

    def test_move_tri(self):
        """
        test move for a triangular grid (first time in file)
        """

        time = datetime.datetime(2004, 12, 31, 13)
        self.cm.model_time = time_utils.date_to_sec(time)
        self.cm.uncertain = True

        time_grid_file = testdata['c_GridCurrentMover']['curr_tri']
        topology_file = testdata['c_GridCurrentMover']['top_tri']

        self.gcm.text_read(time_grid_file, topology_file)
        self.cm.ref[:]['long'] = -76.149368  # for ChesBay
        self.cm.ref[:]['lat'] = 37.74496

        # self.check_move()
        self.check_move_certain_uncertain(self.gcm.uncertain_time_delay)

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = .00148925
        actual[:]['long'] = .00088789
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'],
                                   tol, tol,
                                   msg.format('ches_bay', tol),
                                   0)

        np.testing.assert_allclose(self.cm.delta['long'], actual['long'],
                                   tol, tol,
                                   msg.format('ches_bay', tol),
                                   0)

        # check that certain and uncertain are the same
        # if uncertainty is time delayed
        # self.gcm.uncertain_time_delay = 3

        # cython expects time_delay in seconds
        self.gcm.uncertain_time_delay = 10800

        self.check_move_certain_uncertain(self.gcm.uncertain_time_delay)

    def test_move_ptcur(self):
        """
        test move for a ptCur grid (first time in file)
        """

        time = datetime.datetime(2000, 2, 14, 10)
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = testdata['c_GridCurrentMover']['ptCur']

        self.gcm.text_read(time_grid_file)
        self.cm.ref[:]['long'] = -124.686928  # for ptCur test
        self.cm.ref[:]['lat'] = 48.401124
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = .0161987
        actual[:]['long'] = -.02439887
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'],
                                   tol, tol,
                                   msg.format('ptcur', tol),
                                   0)

        np.testing.assert_allclose(self.cm.delta['long'], actual['long'],
                                   tol, tol,
                                   msg.format('ptcur', tol),
                                   0)

    def test_move_ptcur_extrapolate(self):
        """
        test move for a ptCur grid (first time in file)
        """

        # time before first time in file
        time = datetime.datetime(2000, 2, 14, 8)
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = testdata['c_GridCurrentMover']['ptCur']

        self.gcm.text_read(time_grid_file)

        # result of move should be same as first step for ptCur test
        self.gcm.extrapolate_in_time(True)
        self.cm.ref[:]['long'] = -124.686928
        self.cm.ref[:]['lat'] = 48.401124
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = .0161987
        actual[:]['long'] = -.02439887
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'],
                                   tol, tol,
                                   msg.format('ptcur', tol),
                                   0)

        np.testing.assert_allclose(self.cm.delta['long'], actual['long'],
                                   tol, tol,
                                   msg.format('ptcur', tol),
                                   0)

    def test_move_gridcurtime(self):
        """
        test move for a gridCurTime, grid current time series, file (first time
        in file)
        """

        time = datetime.datetime(2002, 1, 30, 1)
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = testdata['c_GridCurrentMover']['grid_ts']

        # for gridCur test
        self.gcm.text_read(time_grid_file)
        self.cm.ref[:]['long'] = -119.933264
        self.cm.ref[:]['lat'] = 34.138736
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = -0.0034527536849574456
        actual[:]['long'] = 0.005182449331779978
        actual[:]['z'] = 0.
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'],
                                   tol, tol,
                                   msg.format('gridcurtime', tol),
                                   0)

        np.testing.assert_allclose(self.cm.delta['long'], actual['long'],
                                   tol, tol,
                                   msg.format('gridcurtime', tol),
                                   0)

        # np.testing.assert_equal(self.cm.delta, actual,
        #                        "test_move_gridcurtime() failed", 0)

        np.all(self.cm.delta['z'] == 0)

    def test_move_gridcur_series(self):
        """
        test move for a gridCur file series (first time in first file)
        """

        time = datetime.datetime(2002, 1, 30, 1)
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = testdata['c_GridCurrentMover']['series_gridCur']
        topology_file = r""

        self.gcm.text_read(time_grid_file, topology_file)
        self.cm.ref[:]['long'] = -119.933264  # for gridCur test
        self.cm.ref[:]['lat'] = 34.138736
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = -0.0034527536849574456
        actual[:]['long'] = 0.005182449331779978
        actual[:]['z'] = 0.
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'],
                                   tol, tol,
                                   msg.format('gridcur series', tol),
                                   0)

        np.testing.assert_allclose(self.cm.delta['long'], actual['long'],
                                   tol, tol,
                                   msg.format('gridcur series', tol),
                                   0)

        # np.testing.assert_equal(self.cm.delta, actual,
        #                        "test_move_gridcur_series() failed", 0)

        np.all(self.cm.delta['z'] == 0)


if __name__ == '__main__':
    tgc = TestGridCurrentMover()
    tgc.test_move_reg()
    tgc.test_move_curv()
    tgc.test_move_curv_no_top()
    tgc.test_move_curv_series()
    tgc.test_move_tri()
    tgc.test_move_ptcur()
    tgc.test_move_gridcurtime()
    tgc.test_move_gridcur_series()
