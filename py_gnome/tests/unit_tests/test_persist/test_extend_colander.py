#!/usr/bin/env python

"""
tests for our extensions to colander

Not complete at all!

"""

from gnome.persist import extend_colander

from datetime import datetime, timedelta, tzinfo


# just to have a tzinfo object of some sort to play test_serialize_with_tzinfo
class FixedOffset(tzinfo):
    """Fixed offset in minutes east from UTC."""

    def __init__(self, offset, name):
        self.__offset = timedelta(minutes=offset)
        self.__name = name

    def __repr__(self):
        return "FixedOffset(%i, '%s')" % (self.__offset.total_seconds() / 60, self.__name)

    def utcoffset(self, dt):
        return self.__offset

    def tzname(self, dt):
        return self.__name

    def dst(self, dt):
        return 0


class Test_LocalDateTime(object):
    dts = extend_colander.LocalDateTime()

    def test_serialize_simple(self):
        dt = datetime(2016, 2, 12, 13, 32)
        result = self.dts.serialize(None, dt)
        assert result == '2016-02-12T13:32:00'

    def test_serialize_with_tzinfo(self):
        dt = datetime(2016, 2, 12, 13, 32, tzinfo=FixedOffset(3 * 60, '3 hr offset'))
        result = self.dts.serialize(None, dt)
        # offset stripped
        assert result == '2016-02-12T13:32:00'

    def test_deserialize(self):

        dt_str = '2016-02-12T13:32:00'

        result = self.dts.deserialize(None, dt_str)
        assert result == datetime(2016, 2, 12, 13, 32)

    def test_deserialize_with_offset(self):

        dt_str = '2016-02-12T13:32:00+03:00'

        result = self.dts.deserialize(None, dt_str)
        print repr(result)
        assert result == datetime(2016, 2, 12, 13, 32)


