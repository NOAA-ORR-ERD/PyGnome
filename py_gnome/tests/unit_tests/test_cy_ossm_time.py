#!/usr/bin/env python

"""
unit test ossmtime. This is the python access to the cython wrapper cy_ossm_time

"""

# import basic_types and subsequently lib_gnome
import numpy as np
 
from gnome import basic_types
from gnome.cy_gnome import cy_ossm_time
 
class TestCy_ossm_timeInit():
    """
    Test __init__ method and the exceptions it throws for Cy_ossm_time
    """
    tval = np.empty((2,), dtype=basic_types.time_value_pair)
    tval['time'][0] = 0
    tval['value']['u'][0]=1
    tval['value']['v'][0]=2
    
    tval['time'][1] = 1
    tval['value']['u'][1]=2
    tval['value']['v'][1]=3
    
    def test_initNoInputException(self):
        """Test exceptions during __init__ """
        
        # no inputs
        try:
            ossmT2 = cy_ossm_time.Cy_ossm_time()
        except ValueError as e:
            print(e)
            assert True
            
    def test_initBadPathException(self):
        # bad path
        try:
            file = r"SampleData/WindDataFromGnome.WNDX"
            ossmT2 = cy_ossm_time.Cy_ossm_time(path=file, file_contains=basic_types.file_contains.magnitude_direction)
        except IOError as e:
            print(e)
            assert True
            
    def test_initNoUnitsWithFileException(self):
        # correct path but no file_contains
        try:
            file = r"SampleData/WindDataFromGnome.WND"
            ossmT2 = cy_ossm_time.Cy_ossm_time(path=file, file_contains=basic_types.file_contains.magnitude_direction)
        except ValueError as e:
            print(e)
            assert True
     
    def test_initMissingfile_containsException(self):
        # correct path but no file_contains
        try:
            file = r"SampleData/WindDataFromGnome.WND"
            ossmT2 = cy_ossm_time.Cy_ossm_time(path=file)
        except ValueError as e:
            print(e)
            assert True
    
    def test_initNoUnitsWithTimeseriesException(self):
        # timeseries requires units
        try:
            ossm = cy_ossm_time.Cy_ossm_time(timeseries=self.tval)
        except ValueError as e:
            print(e)
            assert True
            


    def test_initFromTimeSeries(self):
        """
        Sets the time series in OSSMTimeValue_c equal to the externally supplied numpy
        array containing time_value_pair data
        It then reads it back to make sure data was set correctly
        """
        ossm = cy_ossm_time.Cy_ossm_time(timeseries=self.tval, units=basic_types.velocity_units.knots)
        t_val = ossm.Timeseries()
        
        np.testing.assert_array_equal(t_val, self.tval, 
                                      "cy_ossm_time.GetTimeValue did not return expected numpy array", 
                                      0)


class TestCy_ossm_timeGetTimeValue():
    """
    Test GetTimeValue method for Cy_ossm_time
    """
    # sample data generated and stored via Gnome GUI
    file = r"SampleData/WindDataFromGnome.WND"
    ossmT = cy_ossm_time.Cy_ossm_time(path=file,
                                      file_contains=basic_types.file_contains.magnitude_direction,
                                      units=basic_types.velocity_units.knots)
    
    
    
    
    def test_TimeValuesAtDataPointsReadFromFile(self):
        """Test GetTimeValues method. It gets the time value pairs for the model times
        stored in the data file. 
        For each line in the data file, the ReadTimeValues method creates one time value pair
            This test just gets the time series that was created from the file. It then invokes
        GetTimeValue for times in the time series.          
        """
        # Let's see what is stored in the Handle to expected result
        t_val = self.ossmT.Timeseries() 
        #print t_val
        #assert False
        
        actual = np.array(t_val['value'], dtype=basic_types.velocity_rec)
        time = np.array(t_val['time'], dtype=basic_types.seconds)
        
        vel_rec = self.ossmT.GetTimeValue(time)
        # TODO: Figure out why following fails??
        #np.testing.assert_allclose(vel_rec, actual, 1e-3, 1e-3, 
        #                          "GetTimeValue is not within a tolerance of 1e-3", 0)
        tol = 1e-6
        np.testing.assert_allclose(vel_rec['u'], actual['u'], tol, tol, 
                                  "GetTimeValue is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(vel_rec['v'], actual['v'], tol, tol, 
                                  "GetTimeValue is not within a tolerance of "+str(tol), 0)
        #assert np.all( np.abs( vel_rec['u']-actual['u'])) < 1e-6
        #assert np.all( np.abs( vel_rec['v']-actual['v'])) < 1e-6
    
    def test__SetTimeValueHandleNone(self):
        """Check TypeError exception for private method"""
        try:
            self.ossmT._SetTimeValueHandle(None)
        except TypeError:
            assert True
        