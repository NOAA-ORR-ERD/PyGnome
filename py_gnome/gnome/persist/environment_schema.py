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
    Range,
)

import gnome
from gnome.persist import validators

from gnome.persist.base_schema import  Id,now
from gnome.persist.extend_colander import LocalDateTime,DatetimeValue2dArray,DatetimeValue2dArraySchema,DefaultTupleSchema


class WindTupleSchema(DefaultTupleSchema):
    """
    Schema for each tuple in WindTimeSeries list
    """
    datetime = SchemaNode(LocalDateTime(default_tzinfo=None), default=now,
                          validator=validators.convertable_to_seconds)
    speed = SchemaNode(Float(), 
                       default=0, 
                       validator=Range(min=0,min_err="wind speed must be greater than or equal to 0"))
    direction = SchemaNode(Float(), default=0, 
                           validator=Range(0,360,
                                           min_err="wind direction must be greater than or equal to 0",
                                           max_err="wind direction must be less than or equal to 360deg"))


class WindTimeSeriesSchema(DatetimeValue2dArraySchema):
    """
    Schema for list of Wind tuples, to make the wind timeseries
    """
    value = WindTupleSchema(default=(datetime.datetime.now(), 0, 0))

    def validator(self, node, cstruct):
        """
        validate wind timeseries numpy array
        """
        validators.no_duplicate_datetime(node, cstruct)
        validators.ascending_datetime(node, cstruct)

    
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
    
    updated_at = SchemaNode(LocalDateTime(), default=None, missing=None, )
    units = SchemaNode(String() )
 
    timeseries = WindTimeSeriesSchema()

class CreateWind( Id, UpdateWind):
    """
    Likely to be used when validating the state of the object read in from a save file.
    The resulting dict is used in new_from_dict(dict) method to construct a new object with same
    state as originally saved object
    
    This is a union of the properties in UpdateWind and Id
    """
    pass

class UpdateTide(MappingSchema):
    filename = SchemaNode( String(), missing=None)
    yeardata = SchemaNode( String() )
    
class CreateTide(Id, UpdateTide):
    pass
