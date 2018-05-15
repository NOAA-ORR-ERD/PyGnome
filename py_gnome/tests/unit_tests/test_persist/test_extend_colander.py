#!/usr/bin/env python

"""
tests for our extensions to colander

Not complete at all!

"""
import datetime as dt
import pytest
import numpy as np
from gnome.persist import extend_colander
import pdb
import pprint as pp

from datetime import datetime
from gnome.utilities.time_utils import FixedOffset
from gnome.environment.gridded_objects_base import Time
from gnome.environment.timeseries_objects_base import (TimeseriesData,
                                                       TimeseriesVector)
from gnome.utilities.serializable_demo_objects import DemoObj, DemoObjSchema
from gnome.persist.base_schema import ObjType, ObjTypeSchema, GeneralGnomeObjectSchema

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
    def test_serialization(self):
        _t = Time(dates())
        tsv = TimeseriesVector(
            variables=[TimeseriesData(name='u', time=_t, data=series_data()),
                       TimeseriesData(name='v', time=_t, data=series_data2())],
            units='m/s'
        )

        inst = DemoObj(variable=tsv, variables=[tsv, tsv.variables[0]])
        serial = inst.serialize()
        deser = DemoObj.deserialize(serial)
        assert deser.variable == inst.variable
        assert deser.variables == inst.variables
        json_, zipfile_, refs = inst.save(saveloc=None)
        loaded = DemoObj.load(zipfile_)
        pdb.set_trace()
        assert inst == loaded
        pdb.set_trace()


class TestObjType(object):
    _t = Time(dates())
    test_class = TimeseriesVector
    test_class_instance = TimeseriesVector(
        variables=[TimeseriesData(name='u', time=_t, data=series_data()),
                   TimeseriesData(name='v', time=_t, data=series_data2())],
        units='m/s'
    )
    def test_serialization(self):
        serial = self.test_class_instance.serialize()
        pp.pprint(serial)
        deser = self.test_class.deserialize(serial)
        pp.pprint(deser)
        assert deser == self.test_class_instance

    def test_save_load(self):
        #without context manager
        json_, zipfile_, refs = self.test_class_instance.save('Test.zip')
        pdb.set_trace()
