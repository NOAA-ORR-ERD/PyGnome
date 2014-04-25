'''
Extend colander's basic types for serialization/deserialization
of gnome specific types
'''
import datetime

import numpy
np = numpy

from colander import Float, DateTime, Sequence, null, Tuple, \
    TupleSchema, SequenceSchema, null

import gnome.basic_types
from gnome.utilities import inf_datetime


class LocalDateTime(DateTime):
    def __init__(self, *args, **kwargs):
        kwargs['default_tzinfo'] = kwargs.get('default_tzinfo', None)
        super(LocalDateTime, self).__init__(*args, **kwargs)

    def strip_timezone(self, _datetime):
        if (_datetime and
            (isinstance(_datetime, datetime.datetime) or
             isinstance(_datetime, datetime.date))
            ):
            _datetime = _datetime.replace(tzinfo=None)
        return _datetime

    def serialize(self, node, appstruct):
        if isinstance(appstruct, datetime.datetime):
            appstruct = self.strip_timezone(appstruct)
            return super(LocalDateTime, self).serialize(node, appstruct)
        elif (isinstance(appstruct, inf_datetime.MinusInfTime) or
              isinstance(appstruct, inf_datetime.InfTime)):
            return appstruct.isoformat()

    def deserialize(self, node, cstruct):
        if cstruct in ('inf', '-inf'):
            return inf_datetime.InfDateTime(cstruct)
        else:
            dt = super(LocalDateTime, self).deserialize(node, cstruct)
            return self.strip_timezone(dt)


class DefaultTuple(Tuple):
    """
    A Tuple subclass that provides defaults from child nodes.

    Required because Tuple returns `colander.null` by default
    when ``appstruct`` is not provided, instead of creating a Tuple of
    default values.
    """
    def serialize(self, node, appstruct):
        items = super(DefaultTuple, self).serialize(node, appstruct)

        if items is null and node.children:
            items = tuple([field.default for field in node.children])

        return items


class VelocityArray(Tuple):
    """
    A subclass of :class:`colander.Sequence` that converts itself to a numpy
    array representing a [velX, velY, velZ] value.
    """
    def serialize(self, node, appstruct):
        if appstruct is null:  # colander.null
            return null

        return super(VelocityArray, self).serialize(node, list(appstruct))

    def deserialize(self, node, cstruct):
        if cstruct is null:
            return null

        return np.array(cstruct, dtype=np.float64)


class VelocityArraySchema(TupleSchema):
    schema_type = VelocityArray


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
            series.append((dt, (wind_value[1][0], wind_value[1][1])))
        appstruct = series

        return super(DatetimeValue2dArray, self).serialize(node, appstruct)

    def deserialize(self, node, cstruct):
        if cstruct is null:
            return null

        items = super(DatetimeValue2dArray, self).deserialize(node,
                cstruct, accept_scalar=False)

        timeseries = np.array(items, dtype=gnome.basic_types.datetime_value_2d)

        return timeseries  # validator requires numpy array


class TimeDelta(Float):
    """
    Add a type to serialize/deserialize timedelta objects
    """
    def serialize(self, node, appstruct):
        if appstruct is not null:
            return super(TimeDelta, self).serialize(node,
                                                    appstruct.total_seconds())
        else:
            return super(TimeDelta, self).serialize(node, null)

    def deserialize(self, *args, **kwargs):
        sec = super(TimeDelta, self).deserialize(*args, **kwargs)
        if sec is not null:
            return datetime.timedelta(seconds=sec)
        else:
            return sec

"""
Following define new schemas for above custom types. This is so
serialize/deserialize is called correctly.

Specifically a new DefaultTypeSchema and a DatetimeValue2dArraySchema
"""


class DefaultTupleSchema(TupleSchema):
    schema_type = DefaultTuple


class DatetimeValue2dArraySchema(SequenceSchema):
    schema_type = DatetimeValue2dArray
