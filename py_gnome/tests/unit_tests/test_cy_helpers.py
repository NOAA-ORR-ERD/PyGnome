#!/usr/bin/evn python
"""
test time conversions from date time to seconds and vice versa
just a python script right now
"""

from gnome.cy_gnome import cy_helpers
from datetime import datetime
import time
import numpy as np
from gnome import basic_types
from gnome.utilities import time_utils

"""
Test CyDateTime
"""
#===========================
# only used in test_cy_helpers to validate time_utils.date_to_sec functionality
date_rec =   np.dtype([('year', np.short),
                       ('month', np.short),
                       ('day', np.short),
                       ('hour', np.short),
                       ('minute', np.short),
                       ('second', np.short), 
                       ('dayOfWeek', np.short),], align=True)


def _convert(x):
    """
    helper method for the next 4 tests
    """
    y = time_utils.date_to_sec(x)
    xn= time_utils.sec_to_date(y)
    return xn

def test_scalar_input():
    """
    test time_utils conversion return a scalar if that is what the user input
    
    always returns a numpy object
    """
    x = datetime.now()
    xn = _convert(x)
    assert type(xn) == datetime
    x = time_utils.round_time(x, roundTo=1)
    assert type(x) == datetime 
    assert x == xn
    
def test_datetime_array():
    """
    test time_utils conversion works for python datetime object
    """
    x = np.zeros((3,), dtype=datetime)
    xn = _convert(x)
    assert np.all( time_utils.round_time(x, roundTo=1) == xn)

def test_numpy_array():
    """
    time_utils works for numpy datetime object
    """
    x = np.zeros((3,), dtype=np.datetime64)
    xn = _convert(x)
    assert np.all( time_utils.round_time(x, roundTo=1) == xn)

class TestCyDateTime():
    target = cy_helpers.CyDateTime()
    now =  datetime.now()

    daterec = np.empty((1,), dtype=date_rec)
    daterec['year'] = now.year
    daterec['month'] = now.month
    daterec['day'] = now.day
    daterec['hour'] = now.hour
    daterec['minute'] = now.minute
    daterec['second'] = now.second
    daterec['dayOfWeek'] = now.weekday() # Gnome 
 
    # use this time for testing
    pySec = time_utils.date_to_sec(now)

    def sec_to_timestruct_from_pyGnome(self, seconds):
        '''
        py_gnome uses this to convert time back to date
        '''
        pyDate = time_utils.sec_to_timestruct(self.pySec)
        date = np.empty((1,), dtype=date_rec)
        date['year'] = pyDate.tm_year
        date['month'] = pyDate.tm_mon
        date['day'] = pyDate.tm_mday
        date['hour'] = pyDate.tm_hour
        date['minute'] = pyDate.tm_min
        date['second'] = pyDate.tm_sec
        date['dayOfWeek'] = pyDate.tm_wday
        return date

    def test_date_to_sec(self):
        '''
        Test DateToSeconds Gnome function. 
        Note the tm_dst = 0 before comparing against Python results
        '''
        sec = self.target.DateToSeconds(self.daterec)
        assert self.pySec == sec

    def test_sec_to_timestruct(self):
        '''
        Test Gnome's reverse conversion back to Date
        '''
        date = np.empty((1,), dtype=date_rec)
        date = self.target.SecondsToDate(self.pySec)

        # let's also get the date from pyGnome function
        pyDate = self.sec_to_timestruct_from_pyGnome(self.pySec)

        # check assertions for everything except dayOfWeek - this doesn't match
        # however, I don't believe this is used anywhere
        # NOTE (JS): Not sure 1 is added to day of the week in StringFunctions.CPP
        # This doesn't seem to effect the date/time value - left it as is.
        # The C++ time struct 0=Sunday and 6=Sat. For Python time struct 0=Monday and 6=Sunday
        print   # for pretty printing
        for field in list(date):
           
            # for pyDate all fields must match
            assert pyDate[field] == self.daterec[field][0]

            if field != 'dayOfWeek':
                #print field + ":" + str(date[field]) + " " + str(self.daterec[field][0])
                print "expected {0}: {1}\t actual {0}: {2}".format(field, date[field], self.daterec[field][0])
                assert date[field] == self.daterec[field][0]
                
    def test_sec_to_date(self):
        """
        Uses time_utils.secondsToDate_noDST to 
        convert the time in seconds back to a datetime object and make
        """
        tgt = time_utils.round_time( dt=self.now, roundTo=1)
        act = time_utils.sec_to_date(self.pySec)
        print
        print "expected:\t" + str(tgt)
        print "actual:  \t" + str(act)
        assert tgt == act

"""
End test CyDateTime
===================
"""

if __name__=="__main__":
    a = TestCyDateTime()
    a.test_date_to_sec()
    a.test_sec_to_timestruct()
    a.test_sec_to_date()
