#!/usr/bin/env python

"""
tests for our extensions to colander

Not complete at all!
"""


import os
from datetime import datetime
import pprint as pp
import json
import tempfile

import pytest

import numpy as np

from gnome.persist import extend_colander
from gnome.persist.extend_colander import NumpyArraySchema
from gnome.utilities.time_utils import FixedOffset
from gnome.environment.gridded_objects_base import Time
from gnome.environment.timeseries_objects_base import (TimeseriesData,
                                                       TimeseriesVector)
from gnome.utilities.serializable_demo_objects import DemoObj


@pytest.fixture(scope='class')
def dates():
    return np.array([datetime(2000, 1, 1, 0),
                     datetime(2000, 1, 1, 2),
                     datetime(2000, 1, 1, 4),
                     datetime(2000, 1, 1, 6),
                     datetime(2000, 1, 1, 8), ])


@pytest.fixture(scope='class')
def series_data():
    return np.array([1, 3, 6, 10, 15])


@pytest.fixture(scope='class')
def series_data2():
    return np.array([2, 6, 12, 20, 30])


class Test_LocalDateTime(object):
    dts = extend_colander.LocalDateTime()

    def test_serialize_simple(self):
        dt = datetime(2016, 2, 12, 13, 32)
        result = self.dts.serialize(None, dt)

        assert result == '2016-02-12T13:32:00'

    def test_serialize_with_tzinfo(self):
        dt = datetime(2016, 2, 12, 13, 32,
                      tzinfo=FixedOffset(3 * 60, '3 hr offset'))
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
        print(repr(result))

        assert result == datetime(2016, 2, 12, 13, 32)


# class TestObjType(object):
#     '''
#     Tests for the colander schematype that represents gnome objects.
#     Tests for the schema are done concurrently since they are paired
#     objects
#     '''
#     def test_construction(self):
#         #__init__
#         class TestSchema(ObjTypeSchema):
#             pass
#         assert TestSchema().schema_type == ObjType
#
#     def test_cstruct_children(self):
#         #cstruct_children
#         pass
#
#     def test_impl(self):
#         #_impl
#         test_schema = ObjTypeSchema()
#
#     def test_serial_pre(self):
#         #_ser
#         pass
#
#     def test_serialize(self):
#         #serialize
#         pass
#
#     def test_deserial_post(self):
#         #_deser
#         pass
#
#     def test_deserialize(self):
#         #deserialize
#         pass
#
#     def test_save_pre(self):
#         #_prepare_save
#         pass
#
#     def test_save_post(self):
#         #_save
#         pass
#
#     def test_save(self):
#         #save
#         pass
#
#     def test_load_pre(self):
#         #_hydrate_json
#         pass
#
#     def test_load(self):
#         #load
#         pass


class TestDemoObj(object):
    def test_serialization(self, dates, series_data, series_data2):
        filename = 'foo.nc'
        times = Time(dates)
        tsv = TimeseriesVector(variables=[TimeseriesData(name='u', time=times,
                                                         data=series_data),
                                          TimeseriesData(name='v', time=times,
                                                         data=series_data2)],
                               units='m/s')

        inst = DemoObj(filename=filename, variable=tsv,
                       variables=[tsv, tsv.variables[0]])
        serial = inst.serialize()
        deser = DemoObj.deserialize(serial)

        assert deser.variable == inst.variable
        assert deser.variables == inst.variables
        assert deser.filename == 'foo.nc'

    def test_save_load(self, dates, series_data, series_data2):
        times = Time(dates)
        tsv = TimeseriesVector(variables=[TimeseriesData(name='u', time=times,
                                                         data=series_data),
                                          TimeseriesData(name='v', time=times,
                                                         data=series_data2)],
                               units='m/s')

        inst = DemoObj(filename=None, variable=tsv,
                       variables=[tsv, tsv.variables[0]])

        saveloc = tempfile.mkdtemp()
        _json_, zipfile_, _refs = inst.save(saveloc=saveloc)
        loaded = DemoObj.load(zipfile_)

        assert inst == loaded

    def test_serialization_options(self, dates, series_data, series_data2):
        times = Time(dates)
        tsv = TimeseriesVector(variables=[TimeseriesData(name='u', time=times,
                                                         data=series_data),
                                          TimeseriesData(name='v', time=times,
                                                         data=series_data2)],
                               units='m/s')

        # kludge for platform differences
        # It should work for the platform the test is running on:
        if os.name == 'posix':
            filename = 'some/random/path/foo.nc'
        else:  # if not posix, should be windows
            filename = os.path.normpath('C:\\foo.nc')

        inst = DemoObj(filename=filename, variable=tsv,
                       variables=[tsv, tsv.variables[0]])
        serial = inst.serialize(options={'raw_paths': False})

        assert serial['filename'] == 'foo.nc'


class TestObjType(object):
    test_class = TimeseriesVector

    def build_test_instance(self, dates, series_data, series_data2):
        times = Time(dates)

        return TimeseriesVector(variables=[TimeseriesData(name='u',
                                                          time=times,
                                                          data=series_data),
                                           TimeseriesData(name='v',
                                                          time=times,
                                                          data=series_data2)],
                                units='m/s')

    def test_serialization(self, dates, series_data, series_data2):
        test_instance = self.build_test_instance(dates,
                                                 series_data, series_data2)

        serial = test_instance.serialize()
        pp.pprint(serial)

        deser = self.test_class.deserialize(serial)
        pp.pprint(deser)

        assert deser == test_instance

    def test_save_load(self, dates, series_data, series_data2):
        test_instance = self.build_test_instance(dates,
                                                 series_data, series_data2)

        # without context manager
        _json_, _zipfile_, _refs = test_instance.save('Test.zip')


class Test_NumpyArraySchema:

    def test_int_tuple(self):
        sch = NumpyArraySchema()

        val = (1, 2, 3, 4)
        ser = sch.serialize(val)
        print(ser)
        deser = sch.deserialize(ser)
        print(deser)

        # Should lose no precision with integers
        assert np.alltrue(deser == val)

    def test_float_list(self):
        sch = NumpyArraySchema()

        val = (1.1, 2.2, 3.3, 4.4)
        ser = sch.serialize(val)
        print(ser)
        deser = sch.deserialize(ser)
        print(deser)

        # Should lose no precision low precision floats
        assert np.alltrue(deser == val)

    def test_float_precision_loss(self):
        sch = NumpyArraySchema(precision=4)

        val = (1.12345678, -2.2345678e20, 3.3456789e-20)
        ser = sch.serialize(val)
        print(ser)
        deser = sch.deserialize(ser)
        print(deser)

        # Should lose some precision
        assert not np.allclose(deser, val, rtol=1e-5, atol=0.0)
        assert np.allclose(deser, val, rtol=1e-3, atol=0.0)

    def test_float_precision_in_json(self):
        sch = NumpyArraySchema(precision=4)

        val = (1.12345678, -2.2345678e20, 3.3456789e-20)
        ser = sch.serialize(val)
        print(f"{ser=}")
        jsonstr = json.dumps(ser)
        print(f"{jsonstr=}")

        assert jsonstr == '[1.123, -2.235e+20, 3.346e-20]'

