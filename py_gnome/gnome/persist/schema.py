import datetime
import time

import numpy
from colander import (
    MappingSchema,
    SchemaNode,
    #Bool,
    #Int,
    Float,
    Range,
    DateTime,
    String,
    SequenceSchema,
    #OneOf,
    #Invalid,
    Sequence,
    TupleSchema,
    deferred,
    null,
    #drop,
    Tuple
)

import gnome.basic_types
from gnome.persist import validators
from gnome.persist import types

@deferred
def now(node, kw):
    """
    Used by TimeseriesValueSchema - assume it defers the calculation of datetime.datetime.now to when it is called in Schema
    """
    return datetime.datetime.now()


class Id(MappingSchema):
    """
    any need to ensure it is valid UUID?
    """
    id = SchemaNode(String() )

    