"""
unit tests cython wrapper

designed to be run with py.test
"""

import os
import numpy as np

from gnome import basic_types
from gnome.cy_gnome import cy_gridcurrent_mover
from gnome.utilities import time_utils

import datetime

import pytest

here = os.path.dirname(__file__)

# def test_exceptions():
#     """
#     Test ValueError exception thrown if improper input arguments
#     """
#     with pytest.raises(ValueError):
#         cy_gridcurrent_mover.CyGridCurrentMover()
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
   
@pytest.mark.slow
class TestGridCurrentMover():
    cm = Common()
    gcm = cy_gridcurrent_mover.CyGridCurrentMover()    
   # delta = np.zeros((cm.num_le,), dtype=basic_types.world_point)
    def move(self): 
        self.gcm.prepare_for_model_run()
        
        self.gcm.prepare_for_model_step(self.cm.model_time, self.cm.time_step)
        self.gcm.get_move( self.cm.model_time,
                          self.cm.time_step, 
                          self.cm.ref,
                          self.cm.delta,
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
        time_grid_file = os.path.join(here, r'SampleData/currents/test.cdf')
        
        
        #f_name = u'SampleData/currents/an_\xe1ccent.cdf'    # but this is
        #f_name = u'SampleData/currents/an_‡ccent.cdf'
        #print "file: ", filename
        #time_grid_file = os.path.join(here, filename)
        
        
        #topology_file = r"SampleData/currents/NYTopology.dat"	# will want a null default
        topology_file = r""	# will want a null default
        self.gcm.text_read(time_grid_file,topology_file)
        self.cm.ref[:]['long'] = (3.104588) #for simple example
        self.cm.ref[:]['lat'] = (52.016468)
        self.check_move()
        actual = np.empty((self.cm.num_le,), dtype=basic_types.world_point)
        actual[:]['lat'] = (.003354610952486354)
        actual[:]['long'] = (.0010056182923228838)
        actual[:]['z'] = (0.)
        tol = 1e-5
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'], tol, tol, 
                                   "test.cdf move is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(self.cm.delta['long'], actual['long'], tol, tol, 
                                   "test.cdf move is not within a tolerance of "+str(tol), 0)
        np.testing.assert_equal(self.cm.delta, actual, "test_move_reg() failed", 0)
        
    def test_move_curv(self):
        """
        test move for a curvilinear grid (first time in file)
        """
        time = datetime.datetime(2008, 1, 29, 17)
        self.cm.model_time = time_utils.date_to_sec(time)
        time_grid_file = os.path.join(here, r'SampleData/currents/ny_cg.nc')
        topology_file = os.path.join(here, r'SampleData/currents/NYTopology.dat')

        self.gcm.text_read(time_grid_file,topology_file)
        self.cm.ref[:]['long'] = (-74.03988) #for NY
        self.cm.ref[:]['lat'] = (40.536092)
        self.check_move()
        actual = np.empty((self.cm.num_le,), dtype=basic_types.world_point)
        actual[:]['lat'] = (.000911)
        actual[:]['long'] = (-.001288)
        tol = 1e-5
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'], tol, tol, 
                                   "ny_cg.nc move is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(self.cm.delta['long'], actual['long'], tol, tol, 
                                   "ny_cg.nc move is not within a tolerance of "+str(tol), 0)
        
    def test_move_curv_series(self):
        """
        Test a curvilinear file series - time in first file, time in second file
        """
        #time = datetime.datetime(2009, 8, 2, 0) #first file
        time = datetime.datetime(2009, 8, 9, 0) #second file
        self.cm.model_time = time_utils.date_to_sec(time)
        #time_grid_file = r"SampleData/currents/file_series/flist1.txt"
        time_grid_file = os.path.join(here, r'SampleData/currents/file_series/flist2.txt')
        topology_file = os.path.join(here, r'SampleData/currents/file_series/HiROMSTopology.dat')
        self.gcm.text_read(time_grid_file,topology_file)
        self.cm.ref[:]['long'] = (-157.795728) #for HiROMS
        self.cm.ref[:]['lat'] = (21.069288)
        self.check_move()
        actual = np.empty((self.cm.num_le,), dtype=basic_types.world_point)
        #actual[:]['lat'] = (.0011565) #file 1
        #actual[:]['long'] = (.00013127) 
        actual[:]['lat'] = (-.003850193) #file 2
        actual[:]['long'] = (.000152012)
        tol = 1e-5
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'], tol, tol, 
                                   "HiROMS move is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(self.cm.delta['long'], actual['long'], tol, tol, 
                                   "HiROMS move is not within a tolerance of "+str(tol), 0)
        
    def test_move_tri(self):
        """
        test move for a triangular grid (first time in file)
        """
        time = datetime.datetime(2004, 12, 31, 13)
        self.cm.model_time = time_utils.date_to_sec(time)
        time_grid_file = os.path.join(here, r'SampleData/currents/ChesBay.nc')
        topology_file = os.path.join(here, r'SampleData/currents/ChesBay.dat')
        self.gcm.text_read(time_grid_file,topology_file)
        self.cm.ref[:]['long'] = (-76.149368) #for ChesBay
        self.cm.ref[:]['lat'] = (37.74496)
        self.check_move()
        actual = np.empty((self.cm.num_le,), dtype=basic_types.world_point)
        actual[:]['lat'] = (-.00170908)
        actual[:]['long'] = (-.0003672)
        tol = 1e-5
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'], tol, tol, 
                                   "ches_bay move is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(self.cm.delta['long'], actual['long'], tol, tol, 
                                   "ches_bay move is not within a tolerance of "+str(tol), 0)
        
    def test_move_ptcur(self):
        """
        test move for a ptCur grid (first time in file)
        """
        time = datetime.datetime(2000, 2, 14, 10)
        self.cm.model_time = time_utils.date_to_sec(time)
        time_grid_file = os.path.join(here, r'SampleData/currents/ptCurNoMap.cur')
        #topology_file = r"SampleData/currents/ChesBay.dat"	
        topology_file = r""	
        self.gcm.text_read(time_grid_file,topology_file)
        self.cm.ref[:]['long'] = (-124.686928) #for ptCur test
        self.cm.ref[:]['lat'] = (48.401124)
        self.check_move()
        actual = np.empty((self.cm.num_le,), dtype=basic_types.world_point)
        actual[:]['lat'] = (.0161987)
        actual[:]['long'] = (-.02439887)
        tol = 1e-5
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'], tol, tol, 
                                   "ptcur move is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(self.cm.delta['long'], actual['long'], tol, tol, 
                                   "ptcur move is not within a tolerance of "+str(tol), 0)
               
    def test_move_gridcurtime(self):
        """
        test move for a gridCurTime file (first time in file)
        """
        time = datetime.datetime(2002, 1, 30, 1)
        self.cm.model_time = time_utils.date_to_sec(time)
        time_grid_file = os.path.join(here, r'SampleData/currents/gridcur_ts.cur')
        #topology_file = r"SampleData/currents/ChesBay.dat"	
        topology_file = r""	
        self.gcm.text_read(time_grid_file,topology_file)
        self.cm.ref[:]['long'] = (-119.933264) #for gridCur test
        self.cm.ref[:]['lat'] = (34.138736)
        self.check_move()
        actual = np.empty((self.cm.num_le,), dtype=basic_types.world_point)
        actual[:]['lat'] = (-0.0034527536849574456)
        actual[:]['long'] = (0.005182449331779978)
        actual[:]['z'] = (0.)
        tol = 1e-5
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'], tol, tol, 
                                   "gridcurtime move is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(self.cm.delta['long'], actual['long'], tol, tol, 
                                   "gridcurtime move is not within a tolerance of "+str(tol), 0)
        np.testing.assert_equal(self.cm.delta, actual, "test_move_gridcurtime() failed", 0)
               
    def test_move_gridcur_series(self):
        """
        test move for a gridCur file series (first time in first file)
        """
        time = datetime.datetime(2002, 1, 30, 1)
        self.cm.model_time = time_utils.date_to_sec(time)
        time_grid_file = os.path.join(here, r'SampleData/currents/gridcur_ts_hdr2.cur')
        #topology_file = r"SampleData/currents/ChesBay.dat"	
        topology_file = r""	
        self.gcm.text_read(time_grid_file,topology_file)
        self.cm.ref[:]['long'] = (-119.933264) #for gridCur test
        self.cm.ref[:]['lat'] = (34.138736)
        self.check_move()
        actual = np.empty((self.cm.num_le,), dtype=basic_types.world_point)
        actual[:]['lat'] = (-0.0034527536849574456)
        actual[:]['long'] = (0.005182449331779978)
        actual[:]['z'] = (0.)
        tol = 1e-5
        np.testing.assert_allclose(self.cm.delta['lat'], actual['lat'], tol, tol, 
                                   "gridcur series move is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(self.cm.delta['long'], actual['long'], tol, tol, 
                                   "gridcur series move is not within a tolerance of "+str(tol), 0)
        np.testing.assert_equal(self.cm.delta, actual, "test_move_gridcur_series() failed", 0)
               
    
if __name__ == "__main__":
    """
    This makes it easy to use this file to debug the lib_gnome DLL 
    through Visual Studio
    """
    tgc = TestGridCurrentMover()
    tgc.test_move_reg()
    tgc.test_move_curv()
