#!/usr/bin/env python

"""
time_utils

assorted utilities for working with time and datetime
"""

import datetime
import time

def date_to_sec(date_time):
    """
    :param date_time: python datetime object 
    Takes the tuple and forces tm_isdst=0, then calls time.mktime to convert to seconds
    Consistent with Gnome, it does not account for daylight savings time.
    The epoch is as defined in python: Jan 1, 1970
    """
    temp = list(date_time.timetuple())
    temp[-1] = 0 
    return time.mktime(temp)

def sec_to_date(seconds):
    """
    :param seconds: time in seconds
    Takes the time and converts it back to datetime object.  
    
    It invokes time_utils.sec_to_timestruct(...), which it then
    converts back to datetime object. It keeps time to seconds accuracy,
    so upto the tm_sec field. tm_isdst = 0. Does not account for DST
    
    Note: Functionality broken up into time_utils.sec_to_timestruct(...) to test
    that it works in the same way as the lib_gnome C++ cython wrapper 
    """   
    t = sec_to_timestruct(seconds)
    dt = datetime.datetime(*t[:7])
    return dt 
    
def sec_to_timestruct(seconds):
    """
    :param seconds: time in seconds
    Takes the time and converts it back using localtime()
    If tm_dst = 1 (by default), then subtract 1 hour and set this flag to 0
    Returns a time.struct_time
    """
    lt = list(time.localtime(seconds))
    if lt[-1] != 0:
        lt[-1] = 0
        lt[3] -= 1
    
    return time.struct_time(lt)
    
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

