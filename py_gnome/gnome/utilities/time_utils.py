#!/usr/bin/env python

"""
time_utils

assorted utilities for working with time and datetime
"""

import datetime
import time

import numpy
np = numpy


def date_to_sec(date_time):
    """
    :param date_time: Either a python datetime object or a numpy array
                      of dtype=datetime or dtype=numpy.datetime64
    :returns: either an array containing the time in seconds or just
              the date in seconds if only 1 input

    For each date, it makes timetuple and forces tm_isdst=0, then calls
    time.mktime to convert to seconds
    Consistent with Gnome, it does not account for daylight savings time.

    The epoch is as defined in python: Jan 1, 1970
    """

    d_array = np.asarray(date_time, dtype='datetime64[s]').reshape(-1)
    t_array = np.zeros(np.shape(d_array), dtype=np.uint32)

    for li in xrange(len(d_array)):
        date = d_array[li].astype(object)
        temp = list(date.timetuple())
        temp[-1] = 0
        t_array[li] = time.mktime(temp)

    return len(t_array) == 1 and t_array[0].astype(object) or t_array


def sec_to_date(seconds):
    """
    :param seconds: Either time in seconds or a numpy array of containing
                    time in seconds

    Takes the time and converts it back to datetime object.

    It invokes time_utils.sec_to_timestruct(...), which it then
    converts back to datetime object. It keeps time to seconds accuracy,
    so upto the tm_sec field. tm_isdst = 0. Does not account for DST

    Note: Functionality broken up into time_utils.sec_to_timestruct(...)
          to test that it works in the same way as the lib_gnome C++
          cython wrapper
    """

    t_array = np.asarray(seconds, dtype=np.uint32).reshape(-1)
    d_array = np.zeros(np.shape(t_array), dtype='datetime64[s]')

    for li in xrange(len(t_array)):
        t = sec_to_timestruct(t_array[li])
        try:
            d_array[li] = datetime.datetime(*t[:6])
        except ValueError:
            print ('Cannot convert timestruct into datetime! '
                   'idx: {0}, '
                   'array elem: {1}, '
                   'timestruct: {2}'.format(li, t_array[li], t))
            raise
    return len(d_array) == 1 and d_array[0].astype(object) or d_array


def sec_to_timestruct(seconds):
    """
    :param seconds: time in seconds

    This doesn't operate on a numpy array. This was separeted as a way to
    explicitly check that we get the same results as the C++ gnome code.
    It is unlikely to be called from pyGnome

    Takes the time and converts it back using localtime()
    If tm_dst = 1 (by default), then subtract 1 hour and set this flag to 0
    Returns a time.struct_time
    """
    secs_in_minute = 60
    minutes_in_hour = 60
    secs_in_hour = secs_in_minute * minutes_in_hour

    lt = list(time.localtime(seconds))
    if lt[-1] != 0:
        # roll clock back by an hour for daylight savings correction
        # and then unset the daylight savings flag
        lt = list(time.localtime(seconds - secs_in_hour))
        lt[-1] = 0

    return time.struct_time(lt)


def round_time(dt=None, roundTo=60):  # IGNORE:W0621
    """
    Round a datetime object or numpy array to any time laps in seconds

    :param dt: datetime.datetime object or numpy array of datetime objects,
               default now.
    :param roundTo: Closest number of seconds to round to, default 1 minute.

    :returns: either an array with rounded values or just a single value
              if only 1 value was input

    Author: Thierry Husson 2012 - Use it as you want but don't blame me.

    found on : http://stackoverflow.com
    """

    if dt is None:
        dt = datetime.datetime.now()

    dt = np.asarray(dt, dtype='datetime64[s]').reshape(-1)

    for li in range(len(dt)):
        date = dt[li].astype(object)
        seconds = (date - date.min).seconds

        # // is a floor division, not a comment on following line:
        rounding = (seconds + roundTo / 2) // roundTo * roundTo

        dt[li] = date + datetime.timedelta(0, rounding - seconds,
                -date.microsecond)

    return len(dt) == 1 and dt[0].astype(object) or dt


if __name__ == '__main__':
    dt = datetime.datetime(2012,
                           12,
                           31,
                           23,
                           44,
                           59,
                           1234)

    print 'a datetime:'
    print dt
    print 'rounded to 1 hour:'
    print round_time(dt, roundTo=60 * 60)

    print 'rounded to 30 minutes:'
    print round_time(dt, roundTo=30 * 60)

    print 'rounded to one day:'
    print round_time(dt, roundTo=3600 * 60 * 60)
