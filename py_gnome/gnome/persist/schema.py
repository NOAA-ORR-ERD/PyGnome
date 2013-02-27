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

from gnome.persist import util

@deferred
def now(node, kw):
    """
    Used by TimeseriesValueSchema - assume it defers the calculation of datetime.datetime.now to when it is called in Schema
    """
    return datetime.datetime.now()

class LocalDateTime(DateTime):
    def __init__(self, *args, **kwargs):
        kwargs['default_tzinfo'] = kwargs.get('default_tzinfo', None)
        super(LocalDateTime, self).__init__(*args, **kwargs)

    def strip_timezone(self, _datetime):
        if _datetime and isinstance(_datetime, datetime.datetime)\
                or isinstance(_datetime, datetime.date):
            _datetime = _datetime.replace(tzinfo=None)
        return _datetime

    def serialize(self, node, appstruct):
        appstruct = self.strip_timezone(appstruct)
        return super(LocalDateTime, self).serialize(node, appstruct)

    def deserialize(self, node, cstruct):
        dt = super(LocalDateTime, self).deserialize(node, cstruct)
        return self.strip_timezone(dt)


class DefaultTuple(Tuple):
    """
    A Tuple subclass that provides defaults from child nodes.

    Required because Tuple returns `colander.null` by default when ``appstruct``
    is not provided, instead of creating a Tuple of default values.
    """
    def serialize(self, node, appstruct):
        items = super(DefaultTuple, self).serialize(node, appstruct)

        if items is null and node.children:
            items = tuple([field.default for field in node.children])

        return items


class DefaultTupleSchema(TupleSchema):
    schema_type = DefaultTuple


class TimeseriesValueSchema(DefaultTupleSchema):
    datetime = SchemaNode(LocalDateTime(default_tzinfo=None), default=now,
                          validator=convertable_to_seconds)
    speed = SchemaNode(Float(), default=0, validator=zero_or_greater)
    # TODO: Validate string and float or just float?
    direction = SchemaNode(Float(), default=0, validator=degrees_true)


class DatetimeValue2dArray(Sequence):
    """
    A subclass of :class:`colander.Sequence` that converts itself to a numpy
    array using :class:`gnome.basic_types.datetime_value_2d` as the data type.
    """
    def __init__(self, *args, **kwargs):
        super(DatetimeValue2dArray, self).__init__(*args, **kwargs)
    
    def serialize(self, node, appstruct):
        """
        This is not used for validation - would be used to converting to JSON
        if monkey-patch is applied and this outputs valid JSON data types
        """
        series = []
        
        for wind_value in appstruct:
            dt = wind_value[0].astype(object)
            series.append((dt, wind_value[1][0], wind_value[1][1]))
        appstruct = series
        
        return super(DatetimeValue2dArray,self).serialize(node, appstruct)
    
    def deserialize(self, *args, **kwargs):
        items = super(DatetimeValue2dArray, self).deserialize(*args, **kwargs)
#===============================================================================
#        num_timeseries = len(items)
#        timeseries = numpy.zeros((num_timeseries,),
#                                 dtype=gnome.basic_types.datetime_value_2d)
# 
#        for idx, value in enumerate(items):
#            timeseries['time'][idx] = value[0]
#            timeseries['value'][idx] = (value[1], value[2])
# 
#        return timeseries
#===============================================================================
        return util.list_to_datetime_value_2d(items)    # validator requires numpy array


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
