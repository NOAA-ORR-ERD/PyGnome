'''
Extend colander's basic types for serialization/deserialization
of gnome specific types
'''

import datetime
import os

import numpy as np

from colander import (Float, DateTime, Sequence, Tuple, List,
                      TupleSchema, SequenceSchema, null, SchemaNode, String, Invalid)

import gnome.basic_types
from gnome.utilities import inf_datetime, round_sf_array

class LocalDateTime(DateTime):
    def __init__(self, *args, **kwargs):
        kwargs['default_tzinfo'] = kwargs.get('default_tzinfo', None)
        super(LocalDateTime, self).__init__(*args, **kwargs)

    def strip_timezone(self, _datetime):
        if (_datetime and isinstance(_datetime, (datetime.datetime,
                                                 datetime.date))):
            _datetime = _datetime.replace(tzinfo=None)

        return _datetime

    def serialize(self, node, appstruct):
        """
        Serialize a DateTime object

        returns an iso formatted string
        """
        if isinstance(appstruct, datetime.datetime):
            appstruct = self.strip_timezone(appstruct)

            return super(LocalDateTime, self).serialize(node, appstruct)
        elif isinstance(appstruct, (inf_datetime.InfTime,
                                    inf_datetime.MinusInfTime)):
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


class NumpyFixedLen(Tuple):
    """
    A subclass of :class:`colander.Tuple` that converts itself to a Tuple and
    back to a numpy array. This is used to define schemas for Numpy arrays that
    have a fixed size like WorldPoint, 3D velocity of SimpleMover.
    """
    def serialize(self, node, appstruct):
        """
        Serialize a fixed length numpy array
        """
        if appstruct is null:  # colander.null
            return null

        return super(NumpyFixedLen, self).serialize(node, appstruct.tolist())

    def deserialize(self, node, cstruct):
        if cstruct is null:
            return null

        return np.array(cstruct, dtype=np.float64)


class NumpyArray(List):
    """
    A subclass of :class:`colander.List` that converts itself to a more general
    numpy array of greater than length 1.
    """
    def serialize(self, node, appstruct):
        """
        Serialize a numpy array
        """
        if appstruct is null:  # colander.null
            return null

        return super(NumpyArray, self).serialize(node, np.array(appstruct).tolist())

    def deserialize(self, node, cstruct):
        if cstruct is null:
            return null

        return np.array(cstruct, dtype=np.float64)


class NumpyFixedLenSchema(TupleSchema):
    schema_type = NumpyFixedLen


class DatetimeValue2dArray(Sequence):
    """
    A subclass of :class:`colander.Sequence` that converts itself to a numpy
    array using :class:`gnome.basic_types.datetime_value_2d` as the data type.

    todo: serialize/deserialize must happen for each element - not very
        efficient.
    """
    def serialize(self, node, appstruct):
        """
        Serialize a 2D Datetime value array
        """
        if appstruct is null:  # colander.null
            return null

        # getting serialized by PyGnome so data should be correct
        # is the list() call required? Can we pass a iterable
        # into serialize?
        series = list(zip(appstruct['time'].astype(object),
                     appstruct['value'].tolist()))

        return super(DatetimeValue2dArray, self).serialize(node, series)

    def deserialize(self, node, cstruct):
        if cstruct is null:
            return null

        items = (super(DatetimeValue2dArray, self)
                 .deserialize(node, cstruct, accept_scalar=False))
        timeseries = np.array(items, dtype=gnome.basic_types.datetime_value_2d)

        return timeseries  # validator requires numpy array


class DatetimeValue1dArray(Sequence):
    """
    A subclass of :class:`colander.Sequence` that converts itself to a numpy
    array using :class:`gnome.basic_types.datetime_value_2d` as the data type.
    """
    def serialize(self, node, appstruct):
        if appstruct is null:  # colander.null
            return null

        appstruct = list(zip(appstruct['time'].astype(object), appstruct['value']))

        return super(DatetimeValue1dArray, self).serialize(node, appstruct)

    def deserialize(self, node, cstruct):
        if cstruct is null:
            return null

        items = (super(DatetimeValue1dArray, self)
                 .deserialize(node, cstruct, accept_scalar=False))

        timeseries = np.array(items, dtype=gnome.basic_types.datetime_value_1d)

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


class OrderedCollectionType(Sequence):
    # identical to SequenceSchema except it can tolerate a 'get'
    def _validate(self, node, value, accept_scalar):
        if (hasattr(value, '__iter__') and
            not isinstance(value, str)):
            return list(value)
        if accept_scalar:
            return [value]
        else:
            raise Invalid(node, '{0} is not iterable'.format(value))


"""
Following define new schemas for above custom types. This is so
serialize/deserialize is called correctly.

Specifically a new DefaultTypeSchema and a DatetimeValue2dArraySchema
"""

class FilenameSchema(SequenceSchema):
    def __init__(self, *args, **kwargs):
        kwargs['typ'] = Sequence(accept_scalar=True)
        super(FilenameSchema, self).__init__(SchemaNode(String()), *args, **kwargs)

    def serialize(self, appstruct, options=None):
        rv = super(FilenameSchema, self).serialize(appstruct)
        if rv and options is not None:
            if not options.get('raw_paths', True):
                for i, filename in enumerate(rv):
                    rv[i] = os.path.split(filename)[1]
        if rv and len(rv) == 1:
            return rv[0]
        return rv

    def deserialize(self, cstruct):
        """
        Deserialize a file name
        """
        if cstruct is None or cstruct is null:
            return None
        rv = super(FilenameSchema, self).deserialize(cstruct)
        if len(rv) == 1:
            return rv[0]
        else:
            return rv

'''
np_array = NumpyArraySchema(
    Float(), save=True
)
'''

class NumpyArraySchema(SchemaNode):
    '''
    This schema cannot nest any further schemas inside since it does not follow
    Colander convention for serializing and deserializing.

    It will serialize a numpy array to nested lists of lists of numbers, using
    array.tolist(). It will attempt to convert the array to the type specified
    with the precision specified before doing so.

    It deserializes lists of lists of numbers into a numpy array of dtype
    specified with dtype specified, if at all.

    :param dtype: numpy data type to (de-)serialize to/from
    '''

    def __init__(self, *args, **kwargs):
        # fixme: where/how is this used?? -- why not a class attribute?
        #        and shouldn't it be ndarray, or ???
        self.typ = np.float64
        self.dtype = kwargs.pop('dtype', np.float64)
        self.precision = kwargs.pop('precision', 8)

    def serialize(self, appstruct):
        """
        Serialize a numpy array

        returns data as a list
        """
        if not isinstance(appstruct, (np.ndarray, list, tuple)):
            raise ValueError('Cannot serialize: {0} is not a numpy array, list, or tuple'.format(appstruct))

        return round_sf_array(appstruct, self.precision).astype(self.dtype, copy=False).tolist()

    def deserialize(self, cstruct):
        """
        Deserialize a numpy array

        returns a numpy array from a list
        """
        return np.array(cstruct, dtype=self.dtype)


class OrderedCollectionSchema(SequenceSchema):
    schema_type = OrderedCollectionType


class DefaultTupleSchema(TupleSchema):
    schema_type = DefaultTuple


class DatetimeValue2dArraySchema(SequenceSchema):
    schema_type = DatetimeValue2dArray


class DatetimeValue1dArraySchema(SequenceSchema):
    schema_type = DatetimeValue1dArray
