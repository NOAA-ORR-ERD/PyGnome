#!/usr/bin/env python

"""
Unit tests for CyOSSMTime class
"""

# import basic_types and subsequently lib_gnome
import numpy as np
 
from gnome import basic_types

from gnome.cy_gnome import cy_ossm_time
 
def test_init_no_input():
    """
    Test exceptions during __init__
    - no inputs 
    """
    try:
        ossmT2 = cy_ossm_time.CyOSSMTime()
    except ValueError as e:
        print(e)
        assert True
            
def test_init_bad_path():
    """
    Test exceptions during __init__
    - bad path
    """
    try:
        file = r"SampleData/WindDataFromGnome.WNDX"
        ossmT2 = cy_ossm_time.CyOSSMTime(path=file, file_contains=basic_types.file_contains.magnitude_direction)
    except IOError as e:
        print(e)
        assert True
        
def test_init_no_units():
    """
    Test __init__
    - correct path but no user units 
    Updated so the user units default to meters_per_sec unless specified in the file
    """
    file = r"SampleData/WindDataFromGnome.WND"
    ossmT2 = cy_ossm_time.CyOSSMTime(path=file, file_contains=basic_types.file_contains.magnitude_direction)
    assert ossmT2.user_units == basic_types.velocity_units.meters_per_sec
 
def test_init_missing_info():
    """
    Test exceptions during __init__
    - correct path but no file_contains 
    """
    try:
        file = r"SampleData/WindDataFromGnome.WND"
        ossmT2 = cy_ossm_time.CyOSSMTime(path=file)
    except ValueError as e:
        print(e)
        assert True 


class TestTimeSeriesInit():
   """
   Test __init__ method and the exceptions it throws for CyOSSMTime
   """
   tval = np.empty((2,), dtype=basic_types.time_value_pair)
   tval['time'][0] = 0
   tval['value'][0]=(1,2)
       
   tval['time'][1] = 1
   tval['value'][1]=(2,3)
   
   def test_init_no_units(self):
       """ timeseries defaults user_units to meters_per_sec """
       ossm = cy_ossm_time.CyOSSMTime(timeseries=self.tval)
       assert ossm.user_units == basic_types.velocity_units.meters_per_sec

   def test_init_time_series(self):
       """
       Sets the time series in OSSMTimeValue_c equal to the externally supplied numpy
       array containing time_value_pair data
       It then reads it back to make sure data was set correctly
       """
       ossm = cy_ossm_time.CyOSSMTime(timeseries=self.tval)
       t_val = ossm.time_series
       
       np.testing.assert_array_equal(t_val, self.tval, 
                                     "CyOSSMTime.get_time_value did not return expected numpy array", 
                                     0)
       
   def test_get_time_value(self):
       ossm = cy_ossm_time.CyOSSMTime(timeseries=self.tval)
       
       actual = np.array(self.tval['value'], dtype=basic_types.velocity_rec)
       time = np.array(self.tval['time'], dtype=basic_types.seconds)
       vel_rec = ossm.get_time_value(time)
       tol = 1e-6
       np.testing.assert_allclose(vel_rec['u'], actual['u'], tol, tol, 
                                  "get_time_value is not within a tolerance of "+str(tol), 0)
       np.testing.assert_allclose(vel_rec['v'], actual['v'], tol, tol, 
                                  "get_time_value is not within a tolerance of "+str(tol), 0)
        
       
        
class TestGetTimeValues():
    """
    Test get_time_value method for CyOSSMTime
    """
    # sample data generated and stored via Gnome GUI
    file = r"SampleData/WindDataFromGnome.WND"
    ossmT = cy_ossm_time.CyOSSMTime(path=file,
                                      file_contains=basic_types.file_contains.magnitude_direction)
    
    
    
    
    def test_get_time_value(self):
        """Test get_time_values method. It gets the time value pairs for the model times
        stored in the data file. 
        For each line in the data file, the ReadTimeValues method creates one time value pair
            This test just gets the time series that was created from the file. It then invokes
        get_time_value for times in the time series.          
        """
        # Let's see what is stored in the Handle to expected result
        t_val = self.ossmT.time_series 
        #print t_val
        #assert False
        
        actual = np.array(t_val['value'], dtype=basic_types.velocity_rec)
        time = np.array(t_val['time'], dtype=basic_types.seconds)
        
        vel_rec = self.ossmT.get_time_value(time)
        # TODO: Figure out why following fails??
        #np.testing.assert_allclose(vel_rec, actual, 1e-3, 1e-3, 
        #                          "get_time_value is not within a tolerance of 1e-3", 0)
        tol = 1e-6
        np.testing.assert_allclose(vel_rec['u'], actual['u'], tol, tol, 
                                  "get_time_value is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(vel_rec['v'], actual['v'], tol, tol, 
                                  "get_time_value is not within a tolerance of "+str(tol), 0)
        
    def test__set_time_value_handle_none(self):
        """Check TypeError exception for private method"""
        try:
            self.ossmT._set_time_value_handle(None)
        except TypeError:
            assert True
        
        
class TestReadFileWithConstantWind():
    """
    Read contents for a file that contains a constant wind, this will be just 1 line in the text file.
    """
    file = r"SampleData/WindDataFromGnomeConstantWind.WND"
    ossmT = cy_ossm_time.CyOSSMTime(path=file,
                                      file_contains=basic_types.file_contains.magnitude_direction)
    
    def test_get_time_value(self):
        """Test get_time_values method. It gets the time value pair for the constant wind
        per the data file. 
            This test just gets the time value pair that was created from the file. It then invokes
        get_time_value for that time in the time series and also looks at the velocity 100 sec later.
        Since wind is constant, the value should be unchanged          
        """
        # Let's see what is stored in the Handle to expected result
        t_val = self.ossmT.time_series 
        
        actual = np.array(t_val['value'], dtype=basic_types.velocity_rec)
        time = np.array(t_val['time']+(0, 100), dtype=basic_types.seconds)
        
        vel_rec = self.ossmT.get_time_value(time)
        
        tol = 1e-6

        for vel in vel_rec:
            np.testing.assert_allclose(vel['u'], actual['u'], tol, tol, 
                                      "get_time_value is not within a tolerance of "+str(tol), 0)
            np.testing.assert_allclose(vel['v'], actual['v'], tol, tol, 
                                      "get_time_value is not within a tolerance of "+str(tol), 0)