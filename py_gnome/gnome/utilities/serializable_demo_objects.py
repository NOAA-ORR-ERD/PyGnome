import warnings
import copy
from numbers import Number
import collections

import numpy as np

import colander
import datetime as dt


import unit_conversion

from gnome.persist import base_schema
from gnome.persist.extend_colander import FilenameSchema

from gnome.environment.gridded_objects_base import Time, TimeSchema
from gnome.environment.timeseries_objects_base import TimeseriesDataSchema, TimeseriesVectorSchema,\
    TimeseriesVector, TimeseriesData
from gnome.gnomeobject import GnomeId

'''
This file is documentation and a demonstration of how to use Schema objects to
allow Gnome objects to do the following tasks:
    Save the object to a zip file
    Load the object from a zip file
    Get a serialization of the object (JSON)
    Deserialize a JSON structure into an instance
    Apply a JSON structure as an update
'''

def dates():
    return np.array([dt.datetime(2000, 1, 1, 0),
                     dt.datetime(2000, 1, 1, 2),
                     dt.datetime(2000, 1, 1, 4),
                     dt.datetime(2000, 1, 1, 6),
                     dt.datetime(2000, 1, 1, 8), ])

def series_data():
    return np.array([1,3,6,10,15])

def series_data2():
    return np.array([2,6,12,20,30])


class DemoObjSchema(base_schema.ObjTypeSchema):
    filename = FilenameSchema(
        save=True, update=True, isdatafile=False, test_for_eq=False,
    )

    foo_float = colander.SchemaNode(
        colander.Float(), save=True, update=True
    )

    foo_float_array = colander.SequenceSchema(
        colander.SchemaNode(
            colander.Float()
        ),
        read=True
    )

    timeseries = colander.SequenceSchema(
        colander.TupleSchema(
            children=[colander.SchemaNode(colander.DateTime(default_tzinfo=None)),
                      colander.SchemaNode(colander.Float())]
        ),
        read=True
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

    def __init__(self, filename=None, foo_float=None, foo_float_array=None, variable=None, variables=None, **kwargs):
        self.filename = filename
        self.foo_float = 42.0
        self.foo_float_array = [42.0,84.0]
        self.variable = variable
        self.variables = variables
        super(DemoObj, self).__init__(**kwargs)

    @property
    def timeseries(self):
        return [(t, self.variable.variables[0].data[i]) for i, t in enumerate(self.variable.time)]

    @classmethod
    def demo(cls):
        _t = Time(dates())
        tsv = TimeseriesVector(
            variables=[TimeseriesData(name='u', time=_t, data=series_data()),
                       TimeseriesData(name='v', time=_t, data=series_data2())],
            units='m/s'
        )

        return DemoObj(variable=tsv, variables=[tsv, tsv.variables[0]])

