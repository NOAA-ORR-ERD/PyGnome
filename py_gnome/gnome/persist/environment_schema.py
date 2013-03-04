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
    Float,
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
    """
    not sure why this is requied?
    """
    value = TimeseriesValueSchema(default=(datetime.datetime.now(), 0, 0))


#===============================================================================
# class ReadWind(MappingSchema):
#    """
#    creates valid json schema for readonly fields of Wind object
#    """
#===============================================================================
    
class UpdateWind(MappingSchema):
    """
    validate data after deserialize, before it is given back to pyGnome's 
    from_dict to set state of object
    """
    description = SchemaNode(String() )
    latitude = SchemaNode(Float(), default=None, missing=None)
    longitude = SchemaNode(Float(), default=None, missing=None)
    name = SchemaNode(String() )
    source_id = SchemaNode(String() )
    source_type = SchemaNode(String(), validator=OneOf(gnome.basic_types.wind_datasource._attr))
    
    timeseries = WindTimeSeriesSchema(default=[], validator=no_duplicates)
    updated_at = SchemaNode(LocalDateTime(), default=None, missing=None, )
    units = SchemaNode(String() )

class CreateWind( Id, UpdateWind):
    """
    Likely to be used when validating the state of the object read in from a save file.
    The resulting dict is used in new_from_dict(dict) method to construct a new object with same
    state as originally saved object
    
    This is a union of the properties in UpdateWind and Id
    """
    pass