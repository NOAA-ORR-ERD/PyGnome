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
    
    def test_ReadTimeValues(self):
        """
        Tests ReadTimeValues method. Use default format and units.
        """
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
        self.ossmT.GetTimeValue( 0, velrec)
        assert True
        