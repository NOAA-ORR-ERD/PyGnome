#!/usr/bin/evn python
"""
Test time conversions from date time to seconds and vice versa
Just a python script right now
"""

from gnome.cy_gnome import cy_date_time
import datetime
import time
import numpy as np
from gnome import basic_types

class TestCyDateTime():
    target = cy_date_time.Cy_date_time()
    now =  datetime.datetime.now()
    
    daterec = np.empty((1,), dtype=basic_types.date_rec)
    daterec['year'] = now.year
    daterec['month'] = now.month
    daterec['day'] = now.day
    daterec['hour'] = now.hour
    daterec['minute'] = now.minute
    daterec['second'] = now.second
    daterec['dayOfWeek'] = now.weekday()
    
    def test_DateToSeconds(self):
        '''
        Test DateToSeconds Gnome function. 
        Note the tm_dst = 0 before comparing against Python results
        '''
        sec = self.target.DateToSeconds(self.daterec)
        tempNow = list(datetime.datetime.timetuple(self.now))
        tempNow[-1] = 0 # last element is tm_isdst  
        pyTime = time.mktime(tempNow)
        assert pyTime == sec
