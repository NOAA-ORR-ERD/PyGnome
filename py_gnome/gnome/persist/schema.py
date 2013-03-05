import datetime
import time

import numpy
from colander import (
    MappingSchema,
    SchemaNode,
    #Bool,
    #Int,
    Float,
    #Range,
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
from gnome.persist.validators import no_duplicates, convertable_to_seconds, zero_or_greater, degrees_true

@deferred
def now(node, kw):
    """
    Used by TimeseriesValueSchema - assume it defers the calculation of datetime.datetime.now to when it is called in Schema
    """
    return datetime.datetime.now()


class DefaultTupleSchema(TupleSchema):
    schema_type = DefaultTuple


class TimeseriesValueSchema(DefaultTupleSchema):
    datetime = SchemaNode(LocalDateTime(default_tzinfo=None), default=now,
                          validator=convertable_to_seconds)
    speed = SchemaNode(Float(), default=0, validator=zero_or_greater)
    # TODO: Validate string and float or just float?
    direction = SchemaNode(Float(), default=0, validator=degrees_true)


class DatetimeValue2dArraySchema(SequenceSchema):
    schema_type = DatetimeValue2dArray
    
    def validator(self, node, cstruct):
        # place validator for this type in the class
        # just testing it here - incase there is more validation for this type,
        # it can be added here
        no_duplicates(node, cstruct)
        

class Id(MappingSchema):
    # validation only happens during deserialize
    # need to make sure it is valid UUID?
    id = SchemaNode(String() )

    