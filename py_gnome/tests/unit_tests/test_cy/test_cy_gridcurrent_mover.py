"""
unit tests cython wrapper

designed to be run with py.test
"""

import os
import numpy as np

from gnome.basic_types import world_point, status_code_type, \
    spill_type, oil_status

from gnome.cy_gnome.cy_gridcurrent_mover import CyGridCurrentMover

from gnome.utilities import time_utils
from gnome.utilities.remote_data import get_datafile

import datetime

import pytest

here = os.path.dirname(__file__)
cur_dir = os.path.join(here, 'sample_data', 'currents')
cur_series_dir = os.path.join(cur_dir, 'file_series')


# def test_exceptions():
#     """
#     Test ValueError exception thrown if improper input arguments
#     """
#     with pytest.raises(ValueError):
#         cy_gridcurrent_mover.CyGridCurrentMover()
#

class Common:

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


def get_datafiles_in_flist(file_):
    """
    read flist1.txt, flist2.txt, gridcur_ts_hdr2.cur and download the netcdf datafiles from server if required
    
    helper function that reads each line of the file_ and gets an files that are defined by 
    [FILE] or [File] from the server
    
    NOTE: reading the text file and getting the names of the datafiles on the fly works fine. 
          It maybe simpler to define a dict containing all files required by these tests including
          the files listed in flist1.txt, flist2.txt ..currently this is the only place where this 
          needs to be done
    """

    flist = get_datafile(file_)
    with open(flist, 'r') as fh:
        while True:
            line = fh.readline()
            if len(line) == 0:
                break
            elif line[:6].lower() == '[file]':

                # read filename and download it from server if it required

                get_datafile(os.path.join(os.path.split(flist)[0],
                             line[6:].strip()))

    return flist


@pytest.mark.slow
class TestGridCurrentMover:

    cm = Common()
    gcm = CyGridCurrentMover()

    # delta = np.zeros((cm.num_le,), dtype=world_point)

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
            self.cm.status,
            spill_type.forecast,
            )

    def move_uncertain(self):
        self.gcm.prepare_for_model_run()
        spill_size = np.zeros((1, ), dtype=np.int32)  # number of LEs in 1 uncertainty spill - simple test   
        spill_size[0] = self.cm.num_le  # for uncertainty spills
        start_pos=(-76.149368,37.74496,0)

        print "Uncertain move"
        self.gcm.prepare_for_model_step(self.cm.model_time,
                self.cm.time_step, 1, spill_size)
        self.gcm.get_move(
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
            msg.format('grid_current.nc', tol),
            0,
            )
            np.testing.assert_allclose(
            self.cm.delta['long'],
            self.cm.delta_uncertainty['long'],
            tol,
            tol,
            msg.format('grid_current.nc', tol),
            0,
            )

    def test_move_reg(self):
        """
        test move for a regular grid (first time in file)
        """

        time = datetime.datetime(1999, 11, 29, 21)
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = get_datafile(os.path.join(cur_dir, 'test.cdf'))

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
        np.testing.assert_allclose(
            self.cm.delta['lat'],
            actual['lat'],
            tol,
            tol,
            msg.format('test.cdf', tol),
            0,
            )
        np.testing.assert_allclose(
            self.cm.delta['long'],
            actual['long'],
            tol,
            tol,
            msg.format('test.cdf', tol),
            0,
            )

        # np.testing.assert_equal(self.cm.delta['z'], actual['z'],
        #                        "test_move_reg() failed", 0)

        np.all(self.cm.delta['z'] == 0)

    def test_move_curv(self):
        """
        test move for a curvilinear grid (first time in file)
        """

        time = datetime.datetime(2008, 1, 29, 17)
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = get_datafile(os.path.join(cur_dir, 'ny_cg.nc'))
        topology_file = get_datafile(os.path.join(cur_dir,
                r'NYTopology.dat'))

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
        np.testing.assert_allclose(
            self.cm.delta['lat'],
            actual['lat'],
            tol,
            tol,
            msg.format('ny_cg.nc', tol),
            0,
            )
        np.testing.assert_allclose(
            self.cm.delta['long'],
            actual['long'],
            tol,
            tol,
            msg.format('ny_cg.nc', tol),
            0,
            )

    def test_move_curv_no_top(self):
        """
        test move for a curvilinear grid (first time in file)
        """

        time = datetime.datetime(2008, 1, 29, 17)
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = get_datafile(os.path.join(cur_dir, 'ny_cg.nc'))
        self.gcm.text_read(time_grid_file, topology_file=None)

        topology_file2 = os.path.join(cur_dir, 'NYTopologyNew.dat')
        self.gcm.export_topology(topology_file2)
        self.cm.ref[:]['long'] = -74.03988  # for NY
        self.cm.ref[:]['lat'] = 40.536092
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = .000911
        actual[:]['long'] = -.001288
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(
            self.cm.delta['lat'],
            actual['lat'],
            tol,
            tol,
            msg.format('ny_cg.nc', tol),
            0,
            )
        np.testing.assert_allclose(
            self.cm.delta['long'],
            actual['long'],
            tol,
            tol,
            msg.format('ny_cg.nc', tol),
            0,
            )

    def test_move_curv_series(self):
        """
        Test a curvilinear file series
        - time in first file
        - time in second file
        """

        # time = datetime.datetime(2009, 8, 2, 0)  # first file

        time = datetime.datetime(2009, 8, 9, 0)  # second file
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = \
            get_datafiles_in_flist(os.path.join(cur_series_dir,
                                   'flist2.txt'))
        topology_file = get_datafile(os.path.join(cur_series_dir,
                'HiROMSTopology.dat'))

        self.gcm.text_read(time_grid_file, topology_file)
        self.cm.ref[:]['long'] = -157.795728  # for HiROMS
        self.cm.ref[:]['lat'] = 21.069288
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = -.003850193  # file 2
        actual[:]['long'] = .000152012
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(
            self.cm.delta['lat'],
            actual['lat'],
            tol,
            tol,
            msg.format('HiROMS', tol),
            0,
            )
        np.testing.assert_allclose(
            self.cm.delta['long'],
            actual['long'],
            tol,
            tol,
            msg.format('HiROMS', tol),
            0,
            )

    def test_move_tri(self):
        """
        test move for a triangular grid (first time in file)
        """

        time = datetime.datetime(2004, 12, 31, 13)
        self.cm.model_time = time_utils.date_to_sec(time)
        self.cm.uncertain = True

        time_grid_file = get_datafile(os.path.join(cur_dir, 'ChesBay.nc'
                ))
        topology_file = get_datafile(os.path.join(cur_dir, 'ChesBay.dat'
                ))

        self.gcm.text_read(time_grid_file, topology_file)
        self.cm.ref[:]['long'] = -76.149368  # for ChesBay
        self.cm.ref[:]['lat'] = 37.74496
        #self.check_move()
        self.check_move_certain_uncertain(self.gcm.uncertain_time_delay)

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = -.00170908
        actual[:]['long'] = -.0003672
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
        #self.gcm.uncertain_time_delay = 3
        self.gcm.uncertain_time_delay = 10800 # cython expects time_delay in seconds
        self.check_move_certain_uncertain(self.gcm.uncertain_time_delay)

    def test_move_ptcur(self):
        """
        test move for a ptCur grid (first time in file)
        """

        time = datetime.datetime(2000, 2, 14, 10)
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = get_datafile(os.path.join(cur_dir,
                'ptCurNoMap.cur'))

        self.gcm.text_read(time_grid_file)
        self.cm.ref[:]['long'] = -124.686928  # for ptCur test
        self.cm.ref[:]['lat'] = 48.401124
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = .0161987
        actual[:]['long'] = -.02439887
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(
            self.cm.delta['lat'],
            actual['lat'],
            tol,
            tol,
            msg.format('ptcur', tol),
            0,
            )
        np.testing.assert_allclose(
            self.cm.delta['long'],
            actual['long'],
            tol,
            tol,
            msg.format('ptcur', tol),
            0,
            )

    def test_move_ptcur_extrapolate(self):
        """
        test move for a ptCur grid (first time in file)
        """

        time = datetime.datetime(2000, 2, 14, 8)	# time before first time in file
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = get_datafile(os.path.join(cur_dir,
                'ptCurNoMap.cur'))

        self.gcm.text_read(time_grid_file)
        self.gcm.extrapolate_in_time(True)	# result of move should be same as first step
        self.cm.ref[:]['long'] = -124.686928  # for ptCur test
        self.cm.ref[:]['lat'] = 48.401124
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = .0161987
        actual[:]['long'] = -.02439887
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(
            self.cm.delta['lat'],
            actual['lat'],
            tol,
            tol,
            msg.format('ptcur', tol),
            0,
            )
        np.testing.assert_allclose(
            self.cm.delta['long'],
            actual['long'],
            tol,
            tol,
            msg.format('ptcur', tol),
            0,
            )

    def test_move_gridcurtime(self):
        """
        test move for a gridCurTime file (first time in file)
        """

        time = datetime.datetime(2002, 1, 30, 1)
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = get_datafile(os.path.join(cur_dir,
                'gridcur_ts.cur'))

        self.gcm.text_read(time_grid_file)
        self.cm.ref[:]['long'] = -119.933264  # for gridCur test
        self.cm.ref[:]['lat'] = 34.138736
        self.check_move()

        actual = np.empty((self.cm.num_le, ), dtype=world_point)
        actual[:]['lat'] = -0.0034527536849574456
        actual[:]['long'] = 0.005182449331779978
        actual[:]['z'] = 0.
        tol = 1e-5

        msg = r"{0} move is not within a tolerance of {1}"
        np.testing.assert_allclose(
            self.cm.delta['lat'],
            actual['lat'],
            tol,
            tol,
            msg.format('gridcurtime', tol),
            0,
            )
        np.testing.assert_allclose(
            self.cm.delta['long'],
            actual['long'],
            tol,
            tol,
            msg.format('gridcurtime', tol),
            0,
            )

        # np.testing.assert_equal(self.cm.delta, actual,
        #                        "test_move_gridcurtime() failed", 0)

        np.all(self.cm.delta['z'] == 0)

    def test_move_gridcur_series(self):
        """
        test move for a gridCur file series (first time in first file)
        """

        time = datetime.datetime(2002, 1, 30, 1)
        self.cm.model_time = time_utils.date_to_sec(time)

        time_grid_file = get_datafiles_in_flist(os.path.join(cur_dir,
                'gridcur_ts_hdr2.cur'))
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
        np.testing.assert_allclose(
            self.cm.delta['lat'],
            actual['lat'],
            tol,
            tol,
            msg.format('gridcur series', tol),
            0,
            )
        np.testing.assert_allclose(
            self.cm.delta['long'],
            actual['long'],
            tol,
            tol,
            msg.format('gridcur series', tol),
            0,
            )

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
