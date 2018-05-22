import warnings
import copy
from numbers import Number
import collections

import numpy as np

import colander

import unit_conversion

from gnome.persist import base_schema

from gnome.environment.gridded_objects_base import Time, TimeSchema
from gnome.environment.timeseries_objects_base import TimeseriesDataSchema, TimeseriesVectorSchema
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


class DemoObjSchema(base_schema.ObjTypeSchema):
    foo_float = colander.SchemaNode(colander.String(), save=True, update=True)
    foo_float_array = colander.SequenceSchema(colander.SchemaNode(colander.String()), read=True)

    timeseries = colander.SequenceSchema(
        colander.TupleSchema(
            children=[colander.SchemaNode(colander.DateTime(default_tzinfo=None)),
                      colander.SchemaNode(colander.Float())]
        ),
        read=True
    )

    variable = base_schema.GeneralGnomeObjectSchema(
        acceptable_schemas=[TimeseriesDataSchema, TimeseriesVectorSchema],
        save=True,
        save_reference=True,
    )

    variables = colander.SequenceSchema(
        base_schema.GeneralGnomeObjectSchema(
            acceptable_schemas=[TimeseriesDataSchema, TimeseriesVectorSchema]
        ),
        save=True,
        save_reference=True
    )


class DemoObj(GnomeId):

    _schema = DemoObjSchema

    def __init__(self, foo_float=None, foo_float_array=None, variable=None, variables=None, **kwargs):
        self.foo_float = 42.0
        self.foo_float_array = [42.0,84.0]
        self.variable = variable
        self.variables = variables
        super(DemoObj, self).__init__(**kwargs)

    @property
    def timeseries(self):
        return [(t, self.variable.variables[0].data[i]) for i, t in enumerate(self.variable.time)]
