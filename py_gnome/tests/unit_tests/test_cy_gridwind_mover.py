"""
unit tests cython wrapper

designed to be run with py.test
"""

import os
import numpy as np

from gnome import basic_types
from gnome.cy_gnome import cy_gridwind_mover
from gnome.utilities import time_utils

import datetime

import pytest

here = os.path.dirname(__file__)

# def test_exceptions():
#     """
#     Test ValueError exception thrown if improper input arguments
#     """
#     with pytest.raises(ValueError):
#         cy_gridwind_mover.CyGridWindMover()
# 
class Common():
    """
    test setting up and moving four particles
    
    Base class that initializes stuff that is common for multiple cy_gridcurrent_mover objects
    """
    
    #################
    # create arrays #
    #################
    num_le = 4  # test on 4 LEs
    ref  =  np.zeros((num_le,), dtype=basic_types.world_point)   # LEs - initial locations
    delta = np.zeros((num_le,), dtype=basic_types.world_point)
    status = np.empty((num_le,), dtype=basic_types.status_code_type)
    wind = np.zeros((num_le,), dtype=np.double) # windage
    
    time_step = 900
    
    def __init__(self):
        time = datetime.datetime(2012, 8, 20, 13)
        self.model_time = time_utils.date_to_sec( time)
        ################
        # init. arrays #
        ################
        self.ref[:] = 1.
        self.ref[:]['z'] = 0 # on surface by default
        self.status[:] = basic_types.oil_status.in_water
        self.wind[:] = .03
    
@pytest.mark.slow
class TestGridWindMover():
    cm = Common()
    gcm = cy_gridwind_mover.CyGridWindMover()    
   # delta = np.zeros((cm.num_le,), dtype=basic_types.world_point)
    def move(self): 
        self.gcm.prepare_for_model_run()
        
        self.gcm.prepare_for_model_step(self.cm.model_time, self.cm.time_step)
        self.gcm.get_move( self.cm.model_time,
                          self.cm.time_step, 
                          self.cm.ref,
                          self.cm.delta,
                          self.cm.wind,
                          self.cm.status,
                          basic_types.spill_type.forecast,
                          0)
        
    def check_move(self):
        self.move()
        print self.cm.delta
        assert np.all(self.cm.delta['lat'] != 0)
        assert np.all(self.cm.delta['long'] != 0)
        
    def test_move_reg(self):
        """
        test move for a regular grid (first time in file)
        """
        time = datetime.datetime(1999, 11, 29, 21)
        self.cm.model_time = time_utils.date_to_sec(time)
        #time_grid_file = r"sample_data/winds/test_wind.cdf"
        time_grid_file = os.path.join(here, r'sample_data',r'winds',r'test_wind.cdf')
        #topology_file = r"sample_data/winds/WindSpeedDirSubsetTop.dat"	# will want a null default
        topology_file = r""	# will want a null default
        self.gcm.text_read(time_grid_file,topology_file)
        self.cm.ref[:]['long'] = (3.104588) #for simple example
        self.cm.ref[:]['lat'] = (52.016468)
        self.check_move()
        actual = np.empty((self.cm.num_le,), dtype=basic_types.world_point)
        actual[:]['lat'] = (.00010063832857459063)
        actual[:]['long'] = (3.0168548769686512e-05)
        actual[:]['z'] = (0.)
        tol = 1e-5
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'], tol, tol, 
                                   "test_wind.cdf move is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(self.cm.delta['long'], actual['long'], tol, tol, 
                                   "test_wind.cdf move is not within a tolerance of "+str(tol), 0)
        #np.testing.assert_equal(self.cm.delta, actual, "test_move_reg() failed", 0)
        
    def test_move_curv(self):
        """
        test move for a curvilinear grid (first time in file)
        """
        time = datetime.datetime(2006, 3, 31, 21)
        self.cm.model_time = time_utils.date_to_sec(time)
        #time_grid_file = r"sample_data/winds/WindSpeedDirSubset.nc"
        #topology_file = r"sample_data/winds/WindSpeedDirSubsetTop.dat"	
        time_grid_file = os.path.join(here, r'sample_data',r'winds',r'WindSpeedDirSubset.nc')
        topology_file = os.path.join(here, r'sample_data',r'winds',r'WindSpeedDirSubsetTop.dat')
        self.gcm.text_read(time_grid_file,topology_file)
        self.cm.ref[:]['long'] = (-122.934656) #for NWS off CA
        self.cm.ref[:]['lat'] = (38.27594)

        self.check_move()
        actual = np.empty((self.cm.num_le,), dtype=basic_types.world_point)
        actual[:]['lat'] = (0.0009890068148185598)
        actual[:]['long'] = (0.0012165959734995123)
        actual[:]['z'] = (0.)
        tol = 1e-5
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'], tol, tol, 
                                   "WindSpeedDirSubset.nc move is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(self.cm.delta['long'], actual['long'], tol, tol, 
                                   "WindSpeedDirSubset.nc move is not within a tolerance of "+str(tol), 0)
        #np.testing.assert_equal(self.cm.delta, actual, "test_move_curv() failed", 0)
        np.all(self.cm.delta['z'] == 0)
        
    def test_move_curv_no_top(self):
        """
        test move for a curvilinear grid (first time in file)
        """
        time = datetime.datetime(2006, 3, 31, 21)
        self.cm.model_time = time_utils.date_to_sec(time)
        #time_grid_file = r"sample_data/winds/WindSpeedDirSubset.nc"
        #topology_file = r"sample_data/winds/WindSpeedDirSubsetTop.DAT"	
        time_grid_file = os.path.join(here, r'sample_data',r'winds',r'WindSpeedDirSubset.nc')
        #topology_file = os.path.join(here, r'sample_data',r'winds',r'WindSpeedDirSubsetTop.DAT')
        topology_file = None
        topology_file2 = os.path.join(here, r'sample_data',r'winds',r'WindSpeedDirSubsetTopNew.dat')
        self.gcm.text_read(time_grid_file,topology_file)
        self.gcm.export_topology(topology_file2)
        self.cm.ref[:]['long'] = (-122.934656) #for NWS off CA
        self.cm.ref[:]['lat'] = (38.27594)
        self.check_move()
        actual = np.empty((self.cm.num_le,), dtype=basic_types.world_point)
        actual[:]['lat'] = (0.0009890068148185598)
        actual[:]['long'] = (0.0012165959734995123)
        actual[:]['z'] = (0.)
        tol = 1e-5
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'], tol, tol, 
                                   "WindSpeedDirSubset.nc move is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(self.cm.delta['long'], actual['long'], tol, tol, 
                                   "WindSpeedDirSubset.nc move is not within a tolerance of "+str(tol), 0)
        #np.testing.assert_equal(self.cm.delta, actual, "test_move_curv() failed", 0)
        np.all(self.cm.delta['z'] == 0)
        
#     def test_move_curv_series(self):
#         """
#         Test a curvilinear file series - time in first file, time in second file
#         """
#         #time = datetime.datetime(2009, 8, 2, 0) #first file
#         time = datetime.datetime(2009, 8, 9, 0) #second file
#         self.cm.model_time = time_utils.date_to_sec(time)
#         #time_grid_file = r"sample_data/winds/file_series/flist1.txt"
#         time_grid_file = r"sample_data/winds/file_series/flist2.txt"
#         topology_file = r"sample_data/currents/file_series/HiROMSTopology.dat"
#         self.gcm.text_read(time_grid_file,topology_file)
#         self.cm.ref[:]['long'] = (-157.795728) #for HiROMS
#         self.cm.ref[:]['lat'] = (21.069288)
#         self.check_move()
#         actual = np.empty((self.cm.num_le,), dtype=basic_types.world_point)
#         #actual[:]['lat'] = (.0011565) #file 1
#         #actual[:]['long'] = (.00013127) 
#         actual[:]['lat'] = (-.003850193) #file 2
#         actual[:]['long'] = (.000152012)
#         tol = 1e-5
#         np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'], tol, tol, 
#                                    "HiROMS move is not within a tolerance of "+str(tol), 0)
#         np.testing.assert_allclose(self.cm.delta['long'], actual['long'], tol, tol, 
#                                    "HiROMS move is not within a tolerance of "+str(tol), 0)
        
    def test_move_gridwindtime(self):
        """
        test move for a gridCurTime file (first time in file)
        """
       # time = datetime.datetime(2002, 11, 19, 1)
        time = datetime.datetime(2002, 1, 30, 1)
        self.cm.model_time = time_utils.date_to_sec(time) 
        #time_grid_file = r"sample_data/winds/gridWindTime.wnd"
        #time_grid_file = r"sample_data/winds/gridwind_ts.wnd"
        #topology_file = r"sample_data/winds/WindSpeedDirSubsetTop.dat"	
        time_grid_file = os.path.join(here, r'sample_data',r'winds',r'gridwind_ts.wnd')
        topology_file = r""	# will want a null default
        self.gcm.text_read(time_grid_file,topology_file)
        #self.cm.ref[:]['long'] = (-9.936358) #for gridWind test
        #self.cm.ref[:]['lat'] = (42.801036)
        self.cm.ref[:]['long'] = (-119.861328) #for gridWind test
        self.cm.ref[:]['lat'] = (34.130412)
        self.check_move()
        actual = np.empty((self.cm.num_le,), dtype=basic_types.world_point)
        #actual[:]['lat'] = (-1.943844444793926e-05)
        #actual[:]['long'] = (0.00010266066357533313)
        actual[:]['lat'] = (-0.0001765253714478036)
        actual[:]['long'] = (0.00010508690731670587)
        actual[:]['z'] = (0.)
        tol = 1e-5
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'], tol, tol, 
                                   "gridwindtime move is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(self.cm.delta['long'], actual['long'], tol, tol, 
                                   "gridwindtime move is not within a tolerance of "+str(tol), 0)
        #np.testing.assert_equal(self.cm.delta, actual, "test_move_gridcurtime() failed", 0)
               
#     def test_move_gridwind_series(self):
#         """
#         test move for a gridCur file series (first time in first file)
#         """
#         time = datetime.datetime(2002, 1, 30, 1)
#         self.cm.model_time = time_utils.date_to_sec(time)
#         time_grid_file = r"sample_data/winds/gridcur_ts_hdr2.cur"
#         topology_file = r"sample_data/winds/ChesBay.dat"	
#         self.gcm.text_read(time_grid_file,topology_file)
#         self.cm.ref[:]['long'] = (-119.933264) #for gridCur test
#         self.cm.ref[:]['lat'] = (34.138736)
#         self.check_move()
#         actual = np.empty((self.cm.num_le,), dtype=basic_types.world_point)
#         actual[:]['lat'] = (-0.0034527536849574456)
#         actual[:]['long'] = (0.005182449331779978)
#         actual[:]['z'] = (0.)
#         tol = 1e-5
#         np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'], tol, tol, 
#                                    "gridwind series move is not within a tolerance of "+str(tol), 0)
#         np.testing.assert_allclose(self.cm.delta['long'], actual['long'], tol, tol, 
#                                    "gridwind series move is not within a tolerance of "+str(tol), 0)
#         np.testing.assert_equal(self.cm.delta, actual, "test_move_gridwind_series() failed", 0)
               
    
if __name__ == "__main__":
    """
    This makes it easy to use this file to debug the lib_gnome DLL 
    through Visual Studio
    """
    tgc = TestGridWindMover()
    tgc.test_move()
    tgc.test_move_curv()
