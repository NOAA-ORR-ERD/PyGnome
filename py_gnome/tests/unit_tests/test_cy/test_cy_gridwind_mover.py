"""
unit tests cython wrapper

designed to be run with py.test
"""

import os
import datetime
import numpy as np

import pytest

from gnome.basic_types import world_point, status_code_type, \
    oil_status, spill_type

from gnome.cy_gnome.cy_gridwind_mover import CyGridWindMover
from gnome.utilities.time_utils import date_to_sec
from gnome.utilities.remote_data import get_datafile

here = os.path.dirname(__file__)
winds_dir = os.path.join(here, 'sample_data', 'winds')
cur_file = os.path.join(here, 'sample_data', 'currents')


# def test_exceptions():
#     """
#     Test ValueError exception thrown if improper input arguments
#     """
#     with pytest.raises(ValueError):
#         CyGridWindMover()
#

class Common:

    """
    test setting up and moving four particles

    Base class that initializes stuff that is common for
    multiple cy_gridcurrent_mover objects
    """

    # ################
    # create arrays #
    # ################

    num_le = 4  # test on 4 LEs
    ref = np.zeros((num_le, ), dtype=world_point)
    delta = np.zeros((num_le, ), dtype=world_point)
    delta_uncertainty = np.zeros((num_le, ), dtype=world_point)
    status = np.empty((num_le, ), dtype=status_code_type)
    wind = np.zeros((num_le, ), dtype=np.double)  # windage

    time_step = 900

    def __init__(self):
        time = datetime.datetime(2012, 8, 20, 13)
        self.model_time = date_to_sec(time)

        # ###############
        # init. arrays #
        # ###############

        self.ref[:] = 1.
        self.ref[:]['z'] = 0  # on surface by default
        self.status[:] = oil_status.in_water
        self.wind[:] = .03


@pytest.mark.slow
class TestGridWindMover:

    cm = Common()
    gcm = CyGridWindMover()

    # delta = np.zeros((cm.num_le,), dtype=basic_types.world_point)

    def move(self):
        self.gcm.prepare_for_model_run()

        print "Certain move"
        self.gcm.prepare_for_model_step(self.cm.model_time,
                self.cm.time_step)
        self.gcm.get_move(
            self.cm.model_time,
            self.cm.time_step,
            self.cm.ref,
            self.cm.delta,
            self.cm.wind,
            self.cm.status,
            spill_type.forecast,
            )

    def move_uncertain(self):
        self.gcm.prepare_for_model_run()
        spill_size = np.zeros((1, ), dtype=np.int32)  # number of LEs in 1 uncertainty spill - simple test   
        spill_size[0] = self.cm.num_le  # for uncertainty spills
        start_pos=(-122.934656,38.27594,0)

        print "Uncertain move"
        self.gcm.prepare_for_model_step(self.cm.model_time,
                self.cm.time_step, 1, spill_size)
        self.gcm.get_move(
            self.cm.model_time,
            self.cm.time_step,
            self.cm.ref,
            self.cm.delta_uncertainty,
            self.cm.wind,
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
            msg.format('grid_wind.nc', tol),
            0,
            )
            np.testing.assert_allclose(
            self.cm.delta['long'],
            self.cm.delta_uncertainty['long'],
            tol,
            tol,
            msg.format('grid_wind.nc', tol),
            0,
            )

    def test_move_reg(self):
        """
        test move for a regular grid (first time in file)
        """

        time = datetime.datetime(1999, 11, 29, 21)
        self.cm.model_time = date_to_sec(time)

        time_grid_file = get_datafile(os.path.join(winds_dir,
                'test_wind.cdf'))

        self.gcm.text_read(time_grid_file)
        self.cm.ref[:]['long'] = 3.104588  # for simple example
        self.cm.ref[:]['lat'] = 52.016468
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = .00010063832857459063
        actual[:]['long'] = 3.0168548769686512e-05
        actual[:]['z'] = 0.
        tol = 1e-5

        msg = '{0} move is not within a tolerance of {1}'
        np.testing.assert_allclose(
            self.cm.delta['lat'],
            actual['lat'],
            tol,
            tol,
            msg.format('test_wind.cdf', tol),
            0,
            )
        np.testing.assert_allclose(
            self.cm.delta['long'],
            actual['long'],
            tol,
            tol,
            msg.format('test_wind.cdf', tol),
            0,
            )

    def test_move_reg_extrapolate(self):
        """
        test move for a regular grid (first time in file)
        """

        time = datetime.datetime(1999, 11, 29, 20)	# before first time in file
        self.cm.model_time = date_to_sec(time)

        time_grid_file = get_datafile(os.path.join(winds_dir,
                'test_wind.cdf'))

        self.gcm.text_read(time_grid_file)
        self.gcm.extrapolate_in_time(True)
        self.cm.ref[:]['long'] = 3.104588  # for simple example
        self.cm.ref[:]['lat'] = 52.016468
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = .00010063832857459063
        actual[:]['long'] = 3.0168548769686512e-05
        actual[:]['z'] = 0.
        tol = 1e-5

        msg = '{0} move is not within a tolerance of {1}'
        np.testing.assert_allclose(
            self.cm.delta['lat'],
            actual['lat'],
            tol,
            tol,
            msg.format('test_wind.cdf', tol),
            0,
            )
        np.testing.assert_allclose(
            self.cm.delta['long'],
            actual['long'],
            tol,
            tol,
            msg.format('test_wind.cdf', tol),
            0,
            )

    def test_move_curv(self):
        """
        test move for a curvilinear grid (first time in file)
        """

        time = datetime.datetime(2006, 3, 31, 21)
        self.cm.model_time = date_to_sec(time)
        self.cm.uncertain = True

        time_grid_file = get_datafile(os.path.join(winds_dir,
                'WindSpeedDirSubset.nc'))
        topology_file = get_datafile(os.path.join(winds_dir,
                'WindSpeedDirSubsetTop.dat'))

        self.gcm.text_read(time_grid_file, topology_file)
        self.cm.ref[:]['long'] = -122.934656  # for NWS off CA
        self.cm.ref[:]['lat'] = 38.27594
        #self.check_move()
        self.check_move_certain_uncertain(self.gcm.uncertain_time_delay)

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = 0.0009890068148185598
        actual[:]['long'] = 0.0012165959734995123
        actual[:]['z'] = 0.
        tol = 1e-5

        msg = '{0} move is not within a tolerance of {1}'
        np.testing.assert_allclose(
            self.cm.delta['lat'],
            actual['lat'],
            tol,
            tol,
            msg.format('WindSpeedDirSubset.nc', tol),
            0,
            )
        np.testing.assert_allclose(
            self.cm.delta['long'],
            actual['long'],
            tol,
            tol,
            msg.format('WindSpeedDirSubset.nc', tol),
            0,
            )

        #check that certain and uncertain are the same if uncertainty is time delayed
        #self.gcm.uncertain_time_delay = 3
        self.gcm.uncertain_time_delay = 10800 # cython expects time_delay in seconds
        self.check_move_certain_uncertain(self.gcm.uncertain_time_delay)
        # np.testing.assert_equal(self.cm.delta, actual,
        #                        "test_move_curv() failed", 0)

        np.all(self.cm.delta['z'] == 0)

    def test_move_curv_no_top(self):
        """
        test move for a curvilinear grid (first time in file)
        """

        time = datetime.datetime(2006, 3, 31, 21)
        self.cm.model_time = date_to_sec(time)

        time_grid_file = get_datafile(os.path.join(winds_dir,
                'WindSpeedDirSubset.nc'))
        self.gcm.text_read(time_grid_file)

        topology_file2 = os.path.join(winds_dir,
                'WindSpeedDirSubsetTopNew.dat')
        self.gcm.export_topology(topology_file2)
        self.cm.ref[:]['long'] = -122.934656  # for NWS off CA
        self.cm.ref[:]['lat'] = 38.27594
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = 0.0009890068148185598
        actual[:]['long'] = 0.0012165959734995123
        actual[:]['z'] = 0.
        tol = 1e-5

        msg = '{0} move is not within a tolerance of {1}'
        np.testing.assert_allclose(
            self.cm.delta['lat'],
            actual['lat'],
            tol,
            tol,
            msg.format('WindSpeedDirSubset.nc', tol),
            0,
            )
        np.testing.assert_allclose(
            self.cm.delta['long'],
            actual['long'],
            tol,
            tol,
            msg.format('WindSpeedDirSubset.nc', tol),
            0,
            )

        # np.testing.assert_equal(self.cm.delta, actual,
        #                        "test_move_curv() failed", 0)

        np.all(self.cm.delta['z'] == 0)

#     def test_move_curv_series(self):
#         """
#         Test a curvilinear file series
#         - time in first file
#         - time in second file
#         """
#         time = datetime.datetime(2009, 8, 9, 0) #second file
#         self.cm.model_time = time_utils.date_to_sec(time)
#
#         time_grid_file = os.path.join(winds_dir, 'file_series', 'flist2.txt')
#         topology_file = os.path.join(cur_file, 'file_series',
#                                      'HiROMSTopology.dat')
#
#         self.gcm.text_read(time_grid_file,topology_file)
#         self.cm.ref[:]['long'] = (-157.795728) #for HiROMS
#         self.cm.ref[:]['lat'] = (21.069288)
#         self.check_move()
#
#         actual = np.empty((self.cm.num_le,), dtype=basic_types.world_point)
#         actual[:]['lat'] = (-.003850193) #file 2
#         actual[:]['long'] = (.000152012)
#         tol = 1e-5
#
#         msg = "{0} move is not within a tolerance of {1}"
#         np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'],
#                                    tol, tol,
#                                    msg.format("HiROMS", tol), 0)
#         np.testing.assert_allclose(self.cm.delta['long'], actual['long'],
#                                    tol, tol,
#                                    msg.format("HiROMS", tol), 0)

    def test_move_gridwindtime(self):
        """
        test move for a gridCurTime file (first time in file)
        """

        # time = datetime.datetime(2002, 11, 19, 1)

        time = datetime.datetime(2002, 1, 30, 1)
        self.cm.model_time = date_to_sec(time)

        time_grid_file = get_datafile(os.path.join(winds_dir,
                'gridwind_ts.wnd'))

        self.gcm.text_read(time_grid_file)
        self.cm.ref[:]['long'] = -119.861328  # for gridWind test
        self.cm.ref[:]['lat'] = 34.130412
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = -0.0001765253714478036
        actual[:]['long'] = 0.00010508690731670587
        actual[:]['z'] = 0.
        tol = 1e-5

        msg = '{0} move is not within a tolerance of {1}'
        np.testing.assert_allclose(
            self.cm.delta['lat'],
            actual['lat'],
            tol,
            tol,
            msg.format('gridwindtime', tol),
            0,
            )
        np.testing.assert_allclose(
            self.cm.delta['long'],
            actual['long'],
            tol,
            tol,
            msg.format('gridwindtime', tol),
            0,
            )


        # np.testing.assert_equal(self.cm.delta, actual,
        #                        "test_move_gridcurtime() failed", 0)

#     def test_move_gridwind_series(self):
#         """
#         test move for a gridCur file series (first time in first file)
#         """
#         time = datetime.datetime(2002, 1, 30, 1)
#         self.cm.model_time = time_utils.date_to_sec(time)
#
#         time_grid_file = r"sample_data/winds/gridcur_ts_hdr2.cur"
#         topology_file = r"sample_data/winds/ChesBay.dat"
#
#         self.gcm.text_read(time_grid_file,topology_file)
#         self.cm.ref[:]['long'] = (-119.933264) #for gridCur test
#         self.cm.ref[:]['lat'] = (34.138736)
#         self.check_move()
#
#         actual = np.empty((self.cm.num_le,), dtype=basic_types.world_point)
#         actual[:]['lat'] = (-0.0034527536849574456)
#         actual[:]['long'] = (0.005182449331779978)
#         actual[:]['z'] = (0.)
#         tol = 1e-5
#
#         msg = "{0} move is not within a tolerance of {1}"
#         np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'],
#                                    tol, tol,
#                                    msg.format("gridwind series", tol), 0)
#         np.testing.assert_allclose(self.cm.delta['long'], actual['long'],
#                                    tol, tol,
#                                    msg.format("gridwind series", tol), 0)
#         np.testing.assert_equal(self.cm.delta, actual,
#                                 "test_move_gridwind_series() failed", 0)

if __name__ == '__main__':
    tgc = TestGridWindMover()
    tgc.test_move_reg()
    tgc.test_move_curv()
    tgc.test_move_curv_no_top()
    tgc.test_move_gridwindtime()
