#!/usr/bin/env python
"""
time_utils

assorted utilities for working with time and datetime
"""

from datetime import datetime, timedelta, tzinfo
import cftime
from dateutil.parser import parse as parsetime
import time

import numpy as np


# tzinfo classes for use with datetime.datetime
#
# These are supplied out of the box with py3, but not py2, so here they are


class FixedOffset(tzinfo):
    """Fixed offset "timezone" in minutes east from UTC."""

    def __init__(self, offset, name):
        self.__offset = timedelta(minutes=offset)
        self.__name = name

    def __repr__(self):
        return ("FixedOffset({}, '{}')"
                .format(self.__offset.total_seconds() / 60, self.__name))

    def utcoffset(self, dt):
        return self.__offset

    def tzname(self, dt):
        return self.__name

    def dst(self, dt):
        return 0


class UTC(FixedOffset):
    """
    A simple tzinfo class for UTC (i.e. no offset, ever...)
    """
    def __init__(self):
        FixedOffset.__init__(self, 0, 'UTC')

    def __repr__(self):
        return 'UTC()'


def timezone_offset_seconds():
    '''
    Calculates the minimum acceptable date, considering timezones east of GMT.

    returns the offset value in seconds of the beginning of the unix epoch.

    This is mostly for testing purposes.
    '''
    return time.mktime(time.localtime()) - time.mktime(time.gmtime())


def zero_time():
    offset = timezone_offset_seconds()

    return offset if offset >= 0 else 0


def date_to_sec(date_times):
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

    # Can accept either a scalar datetime or a scalr datetime64 or
    # an array of datetime64 or list of datetimes -- messy!

    try:
        # can I index it?
        date_times[0]  # yup -- it's a sequence of some sort
        scalar = False
        if (type(date_times) == np.ndarray and
                date_times.dtype == np.dtype('<M8[s]')):
            date_times = date_times.astype(datetime).tolist()
    except TypeError:
        scalar = True
        date_times = [date_times]
    except IndexError:  # numpy scalars raise IndexError
        scalar = True
        date_times = [date_times.astype(datetime)]

    if not isinstance(date_times[0], (datetime, cftime.datetime)):
    # if not isinstance(date_times[0], datetime):
        raise TypeError('date_to_sec only works on datetime and datetime64 '
                        'objects. Got a: {}'.format(type(date_times[0])))

    t_list = []
    for dt in date_times:
        # last element is "is_dst" flag:
        # 0 means "not DST", 1 means "DST", -1 "unknown"
        # but IIUC, it is only used for the DST transition
        # That is to say, when there can be two times that cross the border,
        # then setting this to 0 seems to force what we want.
        timetuple = dt.timetuple()[:-1] + (0,)
        t_list.append(time.mktime(tuple(timetuple)))

    return np.array(t_list, dtype=np.uint32) if not scalar else t_list[0]


def sec_to_date(seconds):
    """
    :param seconds: Either time in seconds or a numpy array containing
                    time in seconds (integer -- ideally uint32)

    Takes the time and converts it back to datetime object.

    This does NOT use: time_utils.sec_to_timestruct(...), but rather,
    converts directly then "fixes" DST to be compatible with GNOME

    Note: time_utils.sec_to_timestruct(...)
          to test that it works in the same way as the lib_gnome C++
          cython wrapper
    FIXME: this may be broken there!!!!!
    """
    t_array = np.asarray(seconds, dtype=np.uint32).reshape(-1)
    d_list = [sec_to_datetime(sec) for sec in t_array]

    if len(d_list) == 1:
        return d_list[0]
    else:
        return np.array(d_list, dtype='datetime64[s]')


def sec_to_datetime(seconds):
    dt = datetime.fromtimestamp(seconds)
    # check for dst
    # - have to use time.localtime
    # - no idea how else to get it!
    timetuple = time.localtime(seconds)

    if timetuple[-1] == 1:  # DST flag
        dt -= timedelta(hours=1)

    return dt


def sec_to_timestruct(seconds):
    """
    FIXME: left over code from previous attempt
           mirrors Cython/C++ code, but breaks for the spring DST transition

           Only used for testing the Cython code.

    :param seconds: time in seconds

    This doesn't operate on a numpy array. This was separated as a way to
    explicitly check that we get the same results as the C++ gnome code.
    It is unlikely to be called from pyGnome

    Takes the time and converts it back using localtime()
    If tm_dst = 1 (by default), then subtract 1 hour and set this flag to 0
    Returns a time.struct_time
    """
    SECS_IN_HOUR = 3600
    timetuple = time.localtime(seconds)

    if timetuple[-1] == 1:
        # roll clock back by an hour for daylight savings correction
        # and then unset the daylight savings flag
        # FIXME: this breaks for the spring transition!
        # i.e.: datetime(2016, 3, 13, 2, 30))
        # a list so it can be mutated
        timetuple = list(time.localtime(seconds - SECS_IN_HOUR))
        timetuple[-1] = 0

    return time.struct_time(timetuple)


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
        dt = datetime.now()

    dt = np.asarray(dt, dtype='datetime64[s]').reshape(-1)

    for li in range(len(dt)):
        date = dt[li].astype(object)
        seconds = (date - date.min).seconds

        # // is a floor division, not a comment on following line:
        rounding = (seconds + roundTo / 2) // roundTo * roundTo

        dt[li] = date + timedelta(0, rounding - seconds, -date.microsecond)

    return len(dt) == 1 and dt[0].astype(object) or dt


def asdatetime(dt):
    """
    makes sure the input is a datetime.datetime object

    if it already is, it will be passed through.

    If not it will attempt to parse a string to make a datetime object.

    None will also be passed through silently
    """
    if dt is None:
        return dt
    # if not isinstance(dt, datetime):
    if not isinstance(dt, (datetime, cftime.datetime)):
        # assume it's an iso string, or something that dateutils can parse.
        return parsetime(dt, ignoretz=True)
    else:
        return dt
