#!/usr/bin/env python

"""
time_utils

assorted utilities for working with time and datetime
"""

import datetime


def round_time(dt=None, roundTo=60):
   """
   Round a datetime object to any time laps in seconds
   :param dt: datetime.datetime object, default now.
   :param roundTo: Closest number of seconds to round to, default 1 minute.

   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   
   found on : http://stackoverflow.com
   """
   if dt == None :
       dt = datetime.datetime.now()
   seconds = (dt - dt.min).seconds
   # // is a floor division, not a comment on following line:
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)



if __name__ == "__main__":
    
    dt = datetime.datetime(2012,12,31,23,44,59,1234)
    print "a datetime:"
    print dt
    print "rounded to 1 hour:"
    print round_time(dt, roundTo=60*60)

    print "rounded to 30 minutes:"
    print round_time(dt, roundTo=30*60)

    print "rounded to one day:"
    print round_time(dt, roundTo=3600*60*60)

