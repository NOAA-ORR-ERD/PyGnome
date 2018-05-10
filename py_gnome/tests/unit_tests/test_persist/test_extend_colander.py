#!/usr/bin/env python

"""
tests for our extensions to colander

Not complete at all!

"""
import datetime as dt
import pytest
import numpy as np
from gnome.persist import extend_colander

from datetime import datetime
from gnome.utilities.time_utils import FixedOffset
from gnome.environment.gridded_objects_base import Time
from gnome.environment.timeseries_objects_base import (TimeseriesData,
                                                       TimeseriesVector)

@pytest.fixture('class')
def dates():
    return np.array([dt.datetime(2000, 1, 1, 0),
                     dt.datetime(2000, 1, 1, 2),
                     dt.datetime(2000, 1, 1, 4),
                     dt.datetime(2000, 1, 1, 6),
                     dt.datetime(2000, 1, 1, 8), ])

@pytest.fixture('class')
def series_data():
    return np.array([1,3,6,10,15])

@pytest.fixture('class')
def series_data2():
    return np.array([2,6,12,20,30])


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


class TestObjType(object):
    _t = Time(dates())
    test_class = TimeseriesVector
    test_class_instance = TimeseriesVector(
        variables=[TimeseriesData(name='u', time=_t, data=series_data()),
                   TimeseriesData(name='v', time=_t, data=series_data2())],
        units='m/s'
    )
    def test_serialization(self):
        import pdb
        import pprint as pp
        direct_schema_serial = self.test_class_instance._schema().serialize(self.test_class_instance)
        serial = self.test_class_instance.serialize()
        pp.pprint(serial)
        deser = self.test_class._schema().deserialize(serial)
        pp.pprint(deser)
        assert direct_schema_serial == serial
        assert False
        pdb.set_trace()