#!/usr/bin/env python

"""
Unit tests for CyOSSMTime class
"""
import os
# import basic_types and subsequently lib_gnome
import numpy as np
 
from gnome import basic_types

from gnome.cy_gnome import cy_ossm_time
import pytest

datadir = os.path.join(os.path.dirname(__file__), r"SampleData")

def test_exceptions():
    with pytest.raises(IOError):
        cy_ossm_time.CyOSSMTime(filename=os.path.join(datadir, "WindDataFromGnome.WNDX"), file_contains=basic_types.ts_format.magnitude_direction)    # bad path
        
    with pytest.raises(ValueError):
        cy_ossm_time.CyOSSMTime()  # no inputs
        cy_ossm_time.CyOSSMTime(filename=os.path.join(datadir, "WindDataFromGnome.WND"))    # insufficient input info
        cy_ossm_time.CyOSSMTime(filename=os.path.join(datadir, "WindDataFromGnome_BadUnits.WND"), file_contains=basic_types.ts_format.magnitude_direction)    # insufficient input info
    
    with pytest.raises(ValueError):
        cy_ossm_time.CyOSSMTime(filename=os.path.join(datadir, "WindDataFromGnome.WND"), file_contains=0)   # file_contains has wrong int type    

def test_init_units():
    """
    Test __init__
    - correct path 
    Updated so the user units are read from filename
    """
    ossmT2 = cy_ossm_time.CyOSSMTime(filename=os.path.join(datadir, "WindDataFromGnome.WND"), file_contains=basic_types.ts_format.magnitude_direction)
    assert ossmT2.user_units == "knot"

class TestTimeSeriesInit():
    """
    Test __init__ method and the exceptions it throws for CyOSSMTime
    """
    tval = np.array([(0,(1,2)),(1,(2,3))], dtype=basic_types.time_value_pair)

    def test_init_timeseries(self):
        """
        Sets the time series in OSSMTimeValue_c equal to the externally supplied numpy
        array containing time_value_pair data
        It then reads it back to make sure data was set correctly
        """
        ossm = cy_ossm_time.CyOSSMTime(timeseries=self.tval)
        t_val = ossm.timeseries

        assert ossm.user_units == "undefined"    #for velocity this is meters per second
        np.testing.assert_array_equal(t_val, self.tval, 
                                     "CyOSSMTime.get_time_value did not return expected numpy array", 
                                     0)

    def test_get_time_value(self):
        ossm = cy_ossm_time.CyOSSMTime(timeseries=self.tval)

        actual = np.array(self.tval['value'], dtype=basic_types.velocity_rec)
        time = np.array(self.tval['time'], dtype=basic_types.seconds)
        vel_rec = ossm.get_time_value(time)
        print vel_rec
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
    ossmT = cy_ossm_time.CyOSSMTime(filename=os.path.join(datadir, 'WindDataFromGnome.WND'),
                                    file_contains=basic_types.ts_format.magnitude_direction)
    #ossmT = cy_ossm_time.CyOSSMTime(filename=os.path.join(datadir, u'WindDataFromGnome_\xe1ccent.WND'),
    #                                file_contains=basic_types.ts_format.magnitude_direction)
    
    
    
    def test_get_time_value(self):
        """Test get_time_values method. It gets the time value pairs for the model times
        stored in the data filename. 
        For each line in the data filename, the ReadTimeValues method creates one time value pair
            This test just gets the time series that was created from the filename. It then invokes
        get_time_value for times in the time series.          
        """
        # Let's see what is stored in the Handle to expected result
        t_val = self.ossmT.timeseries 
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
            
    def test_timeseries(self):
        """
        test setting the timeseries using timeseries property
        """
        t_val = self.ossmT.timeseries
        for i in range(0,len(t_val)):
            # need to learn how to do following in 1 line of code
            print t_val['value'][i]
            t_val['value']['u'][i] = t_val['value']['u'][i] + 2
            t_val['value']['v'][i] = t_val['value']['v'][i] + 2
            print t_val['value'][i] 
        
        self.ossmT.timeseries = t_val
        new_val = self.ossmT.timeseries
        tol = 1e-10
        np.testing.assert_allclose(t_val['time'], new_val['time'], tol, tol, 
                                  "get_time_value is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(t_val['value']['u'], new_val['value']['u'], tol, tol, 
                                  "get_time_value is not within a tolerance of "+str(tol), 0)
        np.testing.assert_allclose(t_val['value']['v'], new_val['value']['v'], tol, tol, 
                                  "get_time_value is not within a tolerance of "+str(tol), 0)
        
class TestReadFileWithConstantWind():
    """
    Read contents for a filename that contains a constant wind, this will be just 1 line in the text filename.
    """
    ossmT = cy_ossm_time.CyOSSMTime(filename=os.path.join(datadir, 'WindDataFromGnomeConstantWind.WND'),
                                      file_contains=basic_types.ts_format.magnitude_direction)
    
    def test_get_time_value(self):
        """Test get_time_values method. It gets the time value pair for the constant wind
        per the data filename. 
            This test just gets the time value pair that was created from the filename. It then invokes
        get_time_value for that time in the time series and also looks at the velocity 100 sec later.
        Since wind is constant, the value should be unchanged          
        """
        # Let's see what is stored in the Handle to expected result
        t_val = self.ossmT.timeseries 
        
        actual = np.array(t_val['value'], dtype=basic_types.velocity_rec)
        time = np.array(t_val['time']+(0, 100), dtype=basic_types.seconds)
        
        vel_rec = self.ossmT.get_time_value(time)
        
        tol = 1e-6

        for vel in vel_rec:
            np.testing.assert_allclose(vel['u'], actual['u'], tol, tol, 
                                      "get_time_value is not within a tolerance of "+str(tol), 0)
            np.testing.assert_allclose(vel['v'], actual['v'], tol, tol, 
                                      "get_time_value is not within a tolerance of "+str(tol), 0)

if __name__ == "__main__":
    tt = TestTimeSeriesInit()
    tt.test_init_timeseries()
    tt.test_get_time_value()
