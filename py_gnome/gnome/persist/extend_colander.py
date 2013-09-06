'''
Extend standard colander types for gnome specific types
'''

import datetime
import time

import numpy
from colander import Float, DateTime, Sequence, null, Tuple, \
    TupleSchema, SequenceSchema

import gnome.basic_types
from gnome.utilities import inf_datetime


class LocalDateTime(DateTime):

    def __init__(self, *args, **kwargs):
        kwargs['default_tzinfo'] = kwargs.get('default_tzinfo', None)
        super(LocalDateTime, self).__init__(*args, **kwargs)

    def strip_timezone(self, _datetime):
        if _datetime and isinstance(_datetime, datetime.datetime) \
            or isinstance(_datetime, datetime.date):
            _datetime = _datetime.replace(tzinfo=None)
        return _datetime

    def serialize(self, node, appstruct):
        if isinstance(appstruct, datetime.datetime):
            appstruct = self.strip_timezone(appstruct)
            return super(LocalDateTime, self).serialize(node, appstruct)
        elif isinstance(appstruct, inf_datetime.MinusInfTime) \
            or isinstance(appstruct, inf_datetime.InfTime):

            return appstruct.isoformat()

    def deserialize(self, node, cstruct):
        if cstruct == '-inf' or cstruct == 'inf':
            return inf_datetime.InfDateTime(cstruct)
        else:
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


class DatetimeValue2dArray(Sequence):

    """
    A subclass of :class:`colander.Sequence` that converts itself to a numpy
    array using :class:`gnome.basic_types.datetime_value_2d` as the data type.
    """

    def serialize(self, node, appstruct):

        if appstruct is null:  # colander.null
            return null

        series = []

        for wind_value in appstruct:
            dt = wind_value[0].astype(object)
            series.append((dt, wind_value[1][0], wind_value[1][1]))
        appstruct = series

        return super(DatetimeValue2dArray, self).serialize(node,
                appstruct)

    def deserialize(self, node, cstruct):
        if cstruct is null:
            return null

        items = super(DatetimeValue2dArray, self).deserialize(node,
                cstruct, accept_scalar=False)
        num_timeseries = len(items)
        timeseries = numpy.zeros((num_timeseries, ),
                                 dtype=gnome.basic_types.datetime_value_2d)

        for (idx, value) in enumerate(items):
            timeseries['time'][idx] = value[0]
            timeseries['value'][idx] = (value[1], value[2])

        return timeseries  # validator requires numpy array


class TimeDelta(Float):

    """
    Add a type to serialize/deserialize timedelta objects
    """

    def serialize(self, node, appstruct):
        return super(TimeDelta, self).serialize(node,
                appstruct.total_seconds())

    def deserialize(self, *args, **kwargs):
        sec = super(TimeDelta, self).deserialize(*args, **kwargs)
        return datetime.timedelta(seconds=sec)


class DefaultTupleSchema(TupleSchema):

    schema_type = DefaultTuple


class DatetimeValue2dArraySchema(SequenceSchema):

    schema_type = DatetimeValue2dArray


