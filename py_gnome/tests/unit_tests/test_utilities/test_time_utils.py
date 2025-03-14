#!/usr/bin/env python

"""
test time_utils different input formats
"""

from datetime import datetime, timedelta
try:
    from datetime import UTC as dtUTC
except ImportError:
    # datetime.UTC added as of Python 3.11
    from datetime import timezone
    dtUTC = timezone.utc

import numpy as np
import pytest

from gnome.utilities.time_utils import (date_to_sec,
                                        sec_to_date,
                                        round_time,
                                        zero_time,
                                        UTC,
                                        FixedOffset,
                                        asdatetime,
                                        TZOffset,
                                        TZOffsetSchema,
                                        )


def _convert(x):
    """
    helper method for the next 4 tests
    """
    y = date_to_sec(x)

    return sec_to_date(y)


def test_scalar_input():
    """
    test time_utils conversion return a scalar if that is what the user input

    always returns a numpy object
    """

    x = datetime.now()
    xn = _convert(x)
    assert isinstance(xn, datetime)

    x = round_time(x, roundTo=1)
    assert isinstance(x, datetime)
    assert x == xn

    # numpy scalar
    x = np.datetime64(datetime.now())
    xn = _convert(x)
    assert isinstance(xn, datetime)


def test_datetime_array():
    """
    test time_utils conversion works for python datetime object
    """

    x = np.array([datetime.fromtimestamp(zero_time())] * 3,
                 dtype=datetime)
    xn = _convert(x)

    assert np.all(round_time(x, roundTo=1) == xn)


def test_numpy_array():
    """
    time_utils works for numpy datetime object
    """

    x = np.array([datetime.fromtimestamp(zero_time())] * 3,
                 dtype='datetime64[s]')

    xn = _convert(x)
    assert np.all(x == xn)


def test_time_dst():
    """
    test it works for datetime at 23 hours with daylight savings on
    test is only valid for places that have daylight savings time
    """

    x = datetime(2013, 3, 21, 23, 10)
    xn = _convert(x)
    assert np.all(x == xn)

    x = datetime(2013, 2, 21, 23, 10)  # no daylight savings
    xn = _convert(x)
    assert np.all(x == xn)

datetimes_crossing_dst_spring = [(datetime(2016, 3, 13, 0, 30)),  # spring
                                 (datetime(2016, 3, 13, 1, 0)),
                                 (datetime(2016, 3, 13, 1, 30)),
                                 (datetime(2016, 3, 13, 2, 0)),
                                 (datetime(2016, 3, 13, 2, 30)),
                                 (datetime(2016, 3, 13, 3, 0)),
                                 ]

datetimes_crossing_dst_fall = [(datetime(2016, 11, 6, 0, 0)),  # fall
                               (datetime(2016, 11, 6, 0, 30)),
                               (datetime(2016, 11, 6, 1, 0)),
                               (datetime(2016, 11, 6, 1, 30)),
                               (datetime(2016, 11, 6, 2, 0)),
                               (datetime(2016, 11, 6, 2, 30)),
                               ]


@pytest.mark.parametrize("dt", datetimes_crossing_dst_spring + datetimes_crossing_dst_fall)
def test_round_trip_dst(dt):
    # does it round-trip?
    assert dt == sec_to_date(date_to_sec(dt))


def test_to_sec_dst_transition_spring():
    seconds = date_to_sec(datetimes_crossing_dst_spring)

    # checks that the interval is constant -- i.e. no repeated or skipped times
    assert not np.any(np.diff(np.diff(seconds)))


def test_to_sec_dst_transition_fall():
    seconds = date_to_sec(datetimes_crossing_dst_fall)

    # checks that the interval is constant -- i.e. no repeated or skipped times
    assert not np.any(np.diff(np.diff(seconds)))


def test_to_date_dst_transition_spring():
    # these are hard coded from what was generated by date_to_sec
    # they cross the spring transition, and caused a problem.
    seconds = np.array([1457857800, 1457859600, 1457861400,
                        1457863200, 1457865000, 1457866800],
                       dtype=np.uint32)

    dates = sec_to_date(seconds)
    diff = np.diff(dates).astype(np.int64)
    # checks that the interval is constant -- i.e. no repeated or skipped times
    assert not np.any(np.diff(diff))


def test_to_date_dst_transition_fall():
    seconds = [1478419200, 1478421000, 1478422800, 1478424600, 1478426400, 1478428200]

    dates = sec_to_date(seconds)
    diff = np.diff(dates).astype(np.int64)

    # checks that the interval is constant -- i.e. no repeated or skipped times
    assert not np.any(np.diff(diff))


def test_FixedOffset():
    """
    not sure what to test here, but at least make sure it gets called
    """
    tz = FixedOffset(8 * 60, "PST")

    assert tz.utcoffset(datetime(2016, 1, 1)) == timedelta(hours=8)
    assert tz.tzname(datetime(2016, 1, 1)) == "PST"
    assert tz.dst(datetime(2016, 1, 1)) == 0  # Jan, should not be DST
    assert tz.dst(datetime(2016, 7, 1)) == 0  # July, should be DST if it were there


def test_UTC():
    """
    not sure what to test here, but at least make sure it gets called
    """
    tz = UTC()

    assert tz.utcoffset(datetime(2016, 1, 1)) == timedelta(hours=0)
    assert tz.tzname(datetime(2016, 1, 1)) == "UTC"
    assert tz.dst(datetime(2016, 1, 1)) == 0  # Jan, should not be DST
    assert tz.dst(datetime(2016, 7, 1)) == 0  # July, should be DST if it were there


def test_datetime64_dst():
    """
    really a test of datetime64, but where else to put it?
    """
    dts = [(datetime(2016, 3, 13, 0, 0)),  # spring
           (datetime(2016, 3, 13, 0, 30)),
           (datetime(2016, 3, 13, 1, 0)),
           (datetime(2016, 3, 13, 1, 30)),
           (datetime(2016, 3, 13, 2, 0)),
           (datetime(2016, 3, 13, 2, 30)),
           (datetime(2016, 3, 13, 3, 0)),
           (datetime(2016, 3, 13, 3, 30)),
           (datetime(2016, 3, 13, 4, 0)),
           (datetime(2016, 11, 6, 0, 0)),  # fall
           (datetime(2016, 11, 6, 0, 30)),
           (datetime(2016, 11, 6, 1, 0)),
           (datetime(2016, 11, 6, 1, 30)),
           (datetime(2016, 11, 6, 2, 0)),
           (datetime(2016, 11, 6, 2, 30)),
           (datetime(2016, 11, 6, 3, 0)),
           ]

    # do they round-trip?
    for dt in dts:
        assert dt == np.datetime64(dt).astype(datetime)

    # do they round-trip if you process as an array?
    dt_arr = np.array(dts, dtype='datetime64[s]')
    dt_list = dt_arr.astype(datetime).tolist()

    assert dt_list == dts


def test_asdatetime_str():
    dt = asdatetime("2010-06-01T12:30")
    assert isinstance(dt, datetime)
    assert dt == datetime(2010, 6, 1, 12, 30)


def test_asdatetime_str2():
    dt = asdatetime("2010-06-01 12:30")
    assert isinstance(dt, datetime)
    assert dt == datetime(2010, 6, 1, 12, 30)


def test_asdatetime_str3():
    dt = asdatetime("2010-06-01")
    assert isinstance(dt, datetime)
    assert dt == datetime(2010, 6, 1)


def test_asdatetime_dt():
    dt = datetime(2010, 6, 1, 12, 30)
    dt2 = asdatetime(dt)
    assert dt == dt2
    # they should be the same object
    assert dt is dt2


def test_asdatetime_none():
    """ None should get passed through as well """
    dt = asdatetime(None)
    assert dt is None

def test_TZOffset():
    """
    too much in one test, but whatever ...
    """
    tzo = TZOffset(-4)

    assert tzo.offset == -4.0

    tzo = TZOffset(-4, title = "some title")

    assert tzo.offset == -4.0
    assert tzo.title == "some title"


def test_TZOffset_from_timedelta():
    """
    Too much in one test, but whatever ...
    """
    tzo = TZOffset(timedelta(hours=-4), title="some title")

    assert tzo.offset == -4.0
    assert tzo.title == "some title"

def test_TZOffset_timedelta():
    tzo = TZOffset(-3.5)

    print(tzo.as_timedelta())
    print(-timedelta(hours=3, minutes=30))
    assert tzo.as_timedelta() == -timedelta(hours=3, minutes=30)


def test_TZOffset_auto_name():
    tzo = TZOffset(-3.5)

    assert tzo.offset == -3.5
    assert tzo.title == "-03:30"


def test_TZOffset_string():
    """
    too much in one test, but whatever ...
    """
    tzo = TZOffset(-3.5)

    assert tzo.as_iso_string() == "-03:30"

    tzo = TZOffset(8)

    assert tzo.as_iso_string() == "+08:00"

def test_TZOffset_none():
    """
    default is None, with "no timezone set"
    """

    tzo = TZOffset()
    assert tzo.offset is None

    assert tzo.as_iso_string() == ""

    assert tzo.as_timedelta() == timedelta(0)

def test_TZOffset_persist():
    """
    Does is serialize and deserialize properly
    """
    tzo = TZOffset(-3.5, "TZ with half hour")

    pson = TZOffsetSchema().serialize(tzo)

    print(pson)

    assert pson['offset'] == -3.5
    assert pson['title'] == "TZ with half hour"

    tzo2 = TZOffsetSchema().deserialize(pson)

    assert tzo == tzo2

    # can it deal with None?

def test_TZOffset_persist_None():

    tzo = TZOffset()

    pson = TZOffsetSchema().serialize(tzo)

    print(f"{pson=}")

    assert pson['offset'] == None
    assert pson['title'] == "No Timezone Specified"

    tzo2 = TZOffsetSchema().deserialize(pson)

    assert tzo == tzo2

# def test_TZOffset_persist():


