'''
Created on Feb 26, 2013
'''
import datetime 

from colander import (
    MappingSchema,
    SchemaNode,
    Bool,
    String,
    OneOf,
    #===========================================================================
    # Bool,
    # Int,
    # Float,
    # Range,
    # DateTime,
    # String,
    # SequenceSchema,
    # OneOf,
    # Invalid,
    # Sequence,
    # TupleSchema,
    # deferred,
    # null,
    # #drop,
    # Tuple
    #===========================================================================
)

import gnome
from gnome.persist.validators import convertable_to_seconds,no_duplicates
from gnome.persist.schema import DatetimeValue2dArraySchema, LocalDateTime, TimeseriesValueSchema, Id


class WindTimeSeriesSchema(DatetimeValue2dArraySchema):
    value = TimeseriesValueSchema(default=(datetime.datetime.now(), 0, 0))

class Mover(MappingSchema):
    on = SchemaNode(Bool(), default=True, missing=True)
    active_start = SchemaNode(LocalDateTime(), default=None, missing=None,
                              validator=convertable_to_seconds)
    active_stop = SchemaNode(LocalDateTime(), default=None, missing=None,
                             validator=convertable_to_seconds)

class WindReadWrite(MappingSchema):
    """
    validate data before it is given back to pyGnome's from_dict to set state of object
    """
    description = SchemaNode(String() )
    latitude = SchemaNode(String() )
    longitude = SchemaNode(String() )
    name = SchemaNode(String() )
    source_id = SchemaNode(String() )
    source_type = SchemaNode(String(), validator=OneOf(gnome.basic_types.wind_datasource._attr))
    
    timeseries = WindTimeSeriesSchema(default=[], validator=no_duplicates)
    updated_at = SchemaNode(LocalDateTime(), default=None, missing=None, )

class WindState( Id, WindReadWrite):
    """
    Likely to be used when validating the state of the object read in from a save file.
    The resulting dict is used in new_from_dict(dict) method to construct a new object with same
    state as originally saved object
    """
    units = SchemaNode(String() )