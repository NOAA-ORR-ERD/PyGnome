#!/usr/bin/env python

"""
unit test ossmtime. This is the python access to the cython wrapper cy_ossm_time

"""

# import basic_types and subsequently lib_gnome
import numpy as np
 
from gnome import basic_types
from gnome.cy_gnome import cy_ossm_time
 
#===============================================================================
# UNIT TESTING FRAMEWORK
# import unittest
# 
# class TestOSSMTime(unittest.TestCase):
#    # sample data generated and stored via Gnome GUI
#    file = r"SampleData/WindDataFromGnome.WND"
#    ossmT = cy_ossm_time.Cy_ossm_time()
#    
#    def test_ReadTimeValues(self):
#        err = self.ossmT.ReadTimeValues(self.file); # assume default format and units
#        #err = 1
#        self.assertEqual(err, 0, "Error encountered in ReadTimeValues, error code: " + str(err) )
#    
#    #time_vals = np.empty((1,), dtype=basic_types.time_value_pair)
#    #ossm_time = cy_ossm_time.Cy_ossm_time(time_vals)
#    
# if __name__ == '__main__':
#   unittest.main()
#===============================================================================

class TestOSSMTime():
    # sample data generated and stored via Gnome GUI
    file = r"SampleData/WindDataFromGnome.WND"
    ossmT = cy_ossm_time.Cy_ossm_time()
    err = ossmT.ReadTimeValues(file); # assume default format and units
    
    def test_ReadTimeValues(self):
        #err = 1
        if (self.err != 0): 
            #print "Error encountered in ReadTimeValues, error code: " + str(err)
            assert False
    
    def test_GetTimeValue(self):
        velrec = np.empty((1,), dtype=basic_types.velocity_rec)
        velrec['u'] = 0
        velrec['v'] = 0
        self.ossmT.GetTimeValue( 0, velrec)
        assert True
        
            
