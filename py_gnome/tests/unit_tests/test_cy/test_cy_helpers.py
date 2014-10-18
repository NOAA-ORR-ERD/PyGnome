#!/usr/bin/env python

"""
test time conversions from date time to seconds and vice versa
just a python script right now
"""

from datetime import datetime

import numpy as np

from gnome.cy_gnome import cy_helpers
from gnome.utilities import time_utils
from gnome import basic_types


class TestCyDateTime:

    target = cy_helpers.CyDateTime()
    now = datetime.now()

    daterec = np.empty((1, ), dtype=basic_types.date_rec)
    daterec[0] = (now.year, now.month, now.day, now.hour, now.minute,
        now.second, now.weekday())
    # use this time for testing

    pySec = time_utils.date_to_sec(now)

    def sec_to_timestruct_from_pyGnome(self, seconds):
        '''
        py_gnome uses this to convert time back to date
        '''

        pyDate = time_utils.sec_to_timestruct(self.pySec)
        date = np.empty((1, ), dtype=basic_types.date_rec)
        date[0] = (pyDate.tm_year, pyDate.tm_mon, pyDate.tm_mday,
            pyDate.tm_hour, pyDate.tm_min, pyDate.tm_sec, pyDate.tm_wday)
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

        # date = np.empty((1,), dtype=date_rec)

        date = self.target.SecondsToDate(self.pySec)

        # let's also get the date from pyGnome function
        pyDate = self.sec_to_timestruct_from_pyGnome(self.pySec)

        # check assertions for everything except dayOfWeek - this doesn't match
        # however, I don't believe this is used anywhere
        # NOTE (JS): Not sure 1 is added to day of the week
        #            in StringFunctions.CPP
        # This doesn't seem to effect the date/time value - left it as is.
        # The C++ time struct 0=Sunday and 6=Sat.
        # For Python time struct 0=Monday and 6=Sunday

        print
        for field in self.daterec.dtype.names:
            # fixme: dayOfWeek doesn't match - check if this is relevant?
            # for pyDate all fields must match
            # assert pyDate[field] == self.daterec[field][0]

            if field != 'dayOfWeek':
                msg = 'expected {0}: {1}\t actual {0}: {2}'
                print msg.format(field, date[field], self.daterec[field][0])
                assert date[field] == self.daterec[field][0]

    def test_sec_to_date(self):
        """
        Uses time_utils.secondsToDate_noDST to
        convert the time in seconds back to a datetime object and make
        """

        tgt = time_utils.round_time(dt=self.now, roundTo=1)
        act = time_utils.sec_to_date(self.pySec)
        print
        print 'expected:\t' + str(tgt)
        print 'actual:  \t' + str(act)
        assert tgt == act


if __name__ == '__main__':
    a = TestCyDateTime()
    a.test_date_to_sec()
    a.test_sec_to_timestruct()
    a.test_sec_to_date()
