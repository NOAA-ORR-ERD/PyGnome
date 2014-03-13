'''
Created on Feb 26, 2013
'''
import datetime

from colander import (MappingSchema, SchemaNode,
                      Bool, String, Float, Range,
                      OneOf, drop)

import gnome
from gnome.persist import validators

from gnome.persist.base_schema import Id, now
from gnome.persist.extend_colander import (LocalDateTime,
                                           DatetimeValue2dArray,
                                           DatetimeValue2dArraySchema,
                                           DefaultTupleSchema)


class WindTupleSchema(DefaultTupleSchema):

    """
    Schema for each tuple in WindTimeSeries list
    """

    datetime = SchemaNode(LocalDateTime(default_tzinfo=None),
                          default=now,
                          validator=validators.convertible_to_seconds)
    speed = SchemaNode(Float(),
                       default=0,
                       validator=Range(min=0, min_err="wind speed must be greater than or equal to 0"))
    direction = SchemaNode(Float(), default=0,
                           validator=Range(0, 360,
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


class Wind(Id, MappingSchema):

    """
    validate data after deserialize, before it is given back to pyGnome's
    from_dict to set _state of object
    """

    description = SchemaNode(String(), missing=drop)
    latitude = SchemaNode(Float(), missing=drop)
    longitude = SchemaNode(Float(), missing=drop)
    name = SchemaNode(String(), missing=drop)
    source_id = SchemaNode(String(), missing=drop)
    source_type = SchemaNode(String(),
                             validator=OneOf(gnome.basic_types.wind_datasource._attr),
                             default='undefined', missing='undefined')
    updated_at = SchemaNode(LocalDateTime(), missing=drop)
    units = SchemaNode(String(), default='m/s')

    timeseries = WindTimeSeriesSchema(missing=drop)
    filename = SchemaNode(String(), missing=drop)
    name = 'wind'


class Tide(Id, MappingSchema):
    filename = SchemaNode(String(), missing=drop)
    yeardata = SchemaNode(String())
    name = 'tide'
