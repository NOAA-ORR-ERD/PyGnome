#!/usr/bin/env python

"""
unit test ossmtime. This is the python access to the cython wrapper cy_ossm_time

"""

# import basic_types and subsequently lib_gnome
import numpy as np
 
from gnome import basic_types
from gnome.cy_gnome import cy_ossm_time
 

class TestOSSMTime():
    """
    Test methods of the Cy_ossm_time class 
    """
    # sample data generated and stored via Gnome GUI
    file = r"SampleData/WindDataFromGnome.WND"
    ossmT = cy_ossm_time.Cy_ossm_time()
    err = ossmT.ReadTimeValues(file); # assume default format and units
    
    # use this to test setting / getting TimeValuePair
    tval = np.empty((2,), dtype=basic_types.time_value_pair)
    tval['time'][0] = 0
    tval['value']['u'][0]=1
    tval['value']['v'][0]=2
    
    tval['time'][1] = 1
    tval['value']['u'][1]=2
    tval['value']['v'][1]=3
    
    
    def test_ReadTimeValues(self):
        """
        Tests ReadTimeValues method. Use default format and units.
        """
        print "Read file " + self.file
        assert self.err == 0
    
    def test_ReadTimeValuesException(self):
        """Test error when ReadTimeValues does not provide units in data file or as input"""
        ossmT2 = cy_ossm_time.Cy_ossm_time()
        err2 = ossmT2.ReadTimeValues(self.file, 5, -1)
        assert err2 == -1
    
    def test_GetTimeValue(self):
        """Test GetTimeValues method at model_time = 0"""
        velrec = np.empty((1,), dtype=basic_types.velocity_rec)
        velrec['u'] = 0
        velrec['v'] = 0
        velrec = self.ossmT.GetTimeValue(0)
        print "t = 0; " 
        print "(u,v): " + str(velrec['u']) + ',' + str(velrec['v'])
        assert True
        
    
    def test_SetTimeValueHandle(self):
        """
        Sets the time series in OSSMTimeValue_c equal to the externally supplied numpy
        array containing time_value_pair data
        It then reads it back to make sure data was set correctly
        """
        self.ossmT.SetTimeValueHandle(self.tval)
        t_val = self.ossmT.GetTimeValueHandle()
        np.testing.assert_array_equal(t_val, self.tval, 
                                      "cy_ossm_time.GetTimeValue did not return expected numpy array", 
                                      0)