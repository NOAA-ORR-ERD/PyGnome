'''
This file is documentation and a demonstration of how to use Schema objects to
allow Gnome objects to do the following tasks:

::

    Save the object to a zip file
    Load the object from a zip file
    Get a serialization of the object (JSON)
    Deserialize a JSON structure into an instance
    Apply a JSON structure as an update
'''
import datetime as dt

import numpy as np
import colander

from gnome.persist import base_schema
from gnome.persist.extend_colander import FilenameSchema

from gnome.environment.gridded_objects_base import Time
from gnome.environment.timeseries_objects_base import (TimeseriesDataSchema,
                                                       TimeseriesVectorSchema,
                                                       TimeseriesVector,
                                                       TimeseriesData)
from gnome.gnomeobject import GnomeId


def dates():
    return np.array([dt.datetime(2000, 1, 1, 0),
                     dt.datetime(2000, 1, 1, 2),
                     dt.datetime(2000, 1, 1, 4),
                     dt.datetime(2000, 1, 1, 6),
                     dt.datetime(2000, 1, 1, 8), ])


def series_data():
    return np.array([1, 3, 6, 10, 15])


def series_data2():
    return np.array([2, 6, 12, 20, 30])


class DemoObjSchema(base_schema.ObjTypeSchema):
    filename = FilenameSchema(
        save=True, update=True, isdatafile=True, test_equal=False,
    )

    foo_float = colander.SchemaNode(
        colander.Float(), save=True, update=True
    )

    foo_float_array = colander.SequenceSchema(
        colander.SchemaNode(
            colander.Float()
        ),
        read_only=True
    )

    timeseries = colander.SequenceSchema(
        colander.TupleSchema(
            children=[colander.SchemaNode(colander.DateTime(default_tzinfo=None)),
                      colander.SchemaNode(colander.Float())]
        ),
        read_only=True
    )

    variable = base_schema.GeneralGnomeObjectSchema(
        acceptable_schemas=[TimeseriesDataSchema, TimeseriesVectorSchema],
        save=True, update=True, save_reference=True,
    )

    variables = colander.SequenceSchema(
        base_schema.GeneralGnomeObjectSchema(
            acceptable_schemas=[TimeseriesDataSchema, TimeseriesVectorSchema]
        ),
        save=True, update=True, save_reference=True
    )


class DemoObj(GnomeId):

    _schema = DemoObjSchema

    def __init__(self, filename=None, foo_float=None, foo_float_array=None,
                 variable=None, variables=None, **kwargs):
        self.filename = filename
        self.foo_float = 42.0
        self.foo_float_array = [42.0, 84.0]
        self.variable = variable
        self.variables = variables
        super(DemoObj, self).__init__(**kwargs)

    @property
    def timeseries(self):
        return [(t, self.variable.variables[0].data[i])
                for i, t in enumerate(self.variable.time)]

    @classmethod
    def demo(cls):
        _t = Time(dates())
        tsv = TimeseriesVector(
            variables=[TimeseriesData(name='u', time=_t, data=series_data()),
                       TimeseriesData(name='v', time=_t, data=series_data2())],
            units='m/s'
        )

        return DemoObj(variable=tsv, variables=[tsv, tsv.variables[0]])
    
from gnome.persist.extend_colander import DataSchemaNode
class GnomeID_Schema2(base_schema.ObjTypeSchema):
    raw_numpy = DataSchemaNode(israwdata=True, save=True)
    raw_scalar = DataSchemaNode(israwdata=True, save=True)
    pass
class GnomeID_Schema1(base_schema.ObjTypeSchema):
    raw_numpy = DataSchemaNode(israwdata=True, save=True)
    raw_masked = DataSchemaNode(israwdata=True, save=True)
    string_array = DataSchemaNode(israwdata=True, save=True)
    raw_numeric_list = DataSchemaNode(israwdata=True, save=True)
    raw_string_list = DataSchemaNode(israwdata=True, save=True)
    sub_obj = base_schema.GeneralGnomeObjectSchema(acceptable_schemas=[base_schema.ObjTypeSchema, GnomeID_Schema2], save_reference=True)

class GnomeID_OBJ1(GnomeId):
    _schema = GnomeID_Schema1
    def __init__(self,
                    raw_numpy=None,
                    raw_masked=None,
                    string_array=None,
                    raw_numeric_list=None,
                    raw_string_list=None,
                    sub_obj=None,
                    **kwargs):
        super().__init__(**kwargs)
        self.raw_numpy = raw_numpy
        self.raw_masked = raw_masked
        self.string_array = string_array
        self.raw_numeric_list = raw_numeric_list
        self.raw_string_list = raw_string_list
        self.sub_obj = sub_obj
class GnomeID_OBJ2(GnomeId):
    _schema = GnomeID_Schema2
    def __init__(self,
                    raw_numpy=None,
                    raw_scalar=None,
                    **kwargs):
        super().__init__(**kwargs)
        self.raw_numpy = raw_numpy
        self.raw_scalar = raw_scalar