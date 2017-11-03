import copy
from numbers import Number
import collections
import warnings

import numpy as np

from colander import (SchemaNode, SequenceSchema, TupleSchema,
                      Float, String, DateTime,
                      drop)

import unit_conversion

from gnome.utilities import serializable
from gnome.utilities.orderedcollection import OrderedCollection
from gnome.environment.property import (EnvProp, VectorProp,
                                        PropertySchema, VectorPropSchema)
from gnome.environment.gridded_objects_base import Time, TimeSchema


class TimeSeriesPropSchema(PropertySchema):
    time = TimeSchema(missing=drop)
    data = SequenceSchema(SchemaNode(Float()), missing=drop)
    timeseries = SequenceSchema(TupleSchema(children=[SchemaNode(DateTime(default_tzinfo=None),
                                                                 missing=drop),
                                                      SchemaNode(Float(),
                                                                 missing=0)
                                                      ],
                                            missing=drop),
                                missing=drop)


class TimeSeriesProp(EnvProp, serializable.Serializable):

    _state = copy.deepcopy(EnvProp._state)
    _schema = TimeSeriesPropSchema

    _state.add_field([serializable.Field('timeseries', save=False,
                                         update=True),
                      serializable.Field('data', save=True, update=True)])

#     _state.update('time', update=False)

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 data=None,
                 **kwargs):
        '''
            A class that represents a scalar natural phenomenon using a
            time series

            :param name: Name
            :type name: string

            :param units: Units
            :type units: string

            :param time: Time axis of the data
            :type time: [] of datetime.datetime, netCDF4.Variable,
                        or Time object

            :param data: Underlying data source
            :type data: numpy.array, list, or other iterable
        '''
        if len(time) != len(data):
            raise ValueError('Time and data sequences are of '
                             'different length.\n'
                             'len(time) == {0}, len(data) == {1}'
                             .format(len(time), len(data)))

        super(TimeSeriesProp, self).__init__(name, units, time, data)

        self.time = time

        if isinstance(self.data, list):
            self.data = np.asarray(self.data)

    @classmethod
    def constant(cls,
                 name=None,
                 units=None,
                 data=None,):
        if any(var is None for var in (name, data, units)):
            raise ValueError("name, data, or units may not be None")

        if not isinstance(data, Number):
            raise TypeError('{0} data must be a number'.format(name))

        t = Time.constant_time()

        return cls(name=name, units=units, time=t, data=[data])

    @property
    def timeseries(self):
        '''
        Creates a representation of the time series

        :rtype: list of (datetime, double) tuples
        '''
        return map(lambda x, y: (x, y), self.time.data, self.data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        if self.time is not None and len(d) != len(self.time):
            raise ValueError("Data/time interval mismatch")
        else:
            self._data = d

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        if self.data is not None and len(t) != len(self.data):
            warnings.warn("Data/time interval mismatch, doing nothing")
            return

        if isinstance(t, Time):
            self._time = t
        elif isinstance(t, collections.Iterable):
            self._time = Time(t)
        else:
            raise ValueError('Object being assigned must be an iterable '
                             'or a Time object')

    def set_attr(self,
                 name=None,
                 units=None,
                 time=None,
                 data=None):
        self.name = name if name is not None else self.name
        self.units = units if units is not None else self.units

        if data is not None and time is not None:
            if len(time) != len(data):
                raise ValueError("Data/time interval mismatch")

            self._data = data
            self.time = time
        else:
            self.data = data if data is not None else self.data
            self.time = time if time is not None else self.time

    def at(self, points, time, units=None, extrapolate=False, **kwargs):
        '''
            Interpolates this property to the given points at the given time
            with the units specified.

            :param points: A Nx2 array of lon,lat points

            :param time: A datetime object. May be None; if this is so,
                         the variable is assumed to be gridded but
                         time-invariant

            :param units: The units that the result would be converted to
        '''
        value = None

        if len(self.time) == 1:
            # single time time series (constant)
            value = np.full((points.shape[0], 1), self.data, dtype=np.float64)

            if units is not None and units != self.units:
                value = unit_conversion.convert(self.units, units, value)

            return value

        if not extrapolate:
            self.time.valid_time(time)

        t_index = self.time.index_of(time, extrapolate)

        if time > self.time.max_time:
            value = self.data[-1]

        if time <= self.time.min_time:
            value = self.data[0]

        if value is None:
            t_alphas = self.time.interp_alpha(time, extrapolate)

            d0 = self.data[t_index - 1]
            d1 = self.data[t_index]

            value = d0 + (d1 - d0) * t_alphas

        if units is not None and units != self.units:
            value = unit_conversion.convert(self.units, units, value)

        return np.full((points.shape[0], 1), value, dtype=np.float64)

    def is_constant(self):
        return len(self.data) == 1

    def __eq__(self, o):
        t1 = (self.name == o.name and
              self.units == o.units and
              self.time == o.time)
        t2 = all(np.isclose(self.data, o.data))

        return t1 and t2

    def __ne__(self, o):
        return not self.__eq__(o)


class TSVectorPropSchema(VectorPropSchema):
    timeseries = SequenceSchema(TupleSchema(children=[SchemaNode(DateTime(default_tzinfo=None),
                                                                 missing=drop),
                                                      TupleSchema(children=[SchemaNode(Float(), missing=0),
                                                                            SchemaNode(Float(), missing=0)
                                                                            ]
                                                                  )
                                                      ],
                                            missing=drop),
                                missing=drop)
    varnames = SequenceSchema(SchemaNode(String(), missing=drop), missing=drop)


class TSVectorProp(VectorProp):

    _schema = TSVectorPropSchema
    _state = copy.deepcopy(VectorProp._state)

    _state.add_field([serializable.Field('timeseries', save=False,
                                         update=True),
                      serializable.Field('variables', save=True,
                                         update=True, iscollection=True),
                      serializable.Field('varnames', save=True,
                                         update=False)])

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 varnames=None,
                 **kwargs):
        '''
        This class represents a vector phenomenon using a time series
        '''
        if (any([units is None, time is None]) and
                not all([isinstance(v, TimeSeriesProp) for v in variables])):
            raise ValueError('All attributes except name, varnames '
                             'MUST be defined if variables is not a '
                             'list of TimeSeriesProp objects')

        if variables is None or len(variables) < 2:
            raise TypeError('Variables must be an array-like of 2 or more '
                            'TimeSeriesProp or array-like')

        VectorProp.__init__(self, name, units, time, variables)

    @classmethod
    def constant(cls,
                 name=None,
                 units=None,
                 variables=None):
        if any(var is None for var in (name, variables, units)):
            raise ValueError("name, variables, or units may not be None")

        if not isinstance(variables, collections.Iterable):
            raise TypeError('{0} variables must be an iterable'.format(name))

        t = Time.constant_time()

        return cls(name=name, units=units, time=t,
                   variables=[v for v in variables])

    @property
    def timeseries(self):
        '''
        Creates a representation of the time series

        :rtype: list of (datetime, (double, double)) tuples
        '''
        return map(lambda x, y, z: (x, (y, z)),
                   self.time.time,
                   self.variables[0],
                   self.variables[1])

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        if self.variables is not None:
            for v in self.variables:
                v.time = t

        if isinstance(t, Time):
            self._time = t
        elif isinstance(t, collections.Iterable):
            self._time = Time(t)
        else:
            raise ValueError('Object being assigned must be an iterable '
                             'or a Time object')

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, v):
        if v is None:
            self._variables = v

        if isinstance(v, collections.Iterable):
            self._variables = OrderedCollection(v)

    def is_constant(self):
        return len(self.variables[0]) == 1

    def in_units(self, units):
        '''
        Returns a full copy of this property in the units specified.
        WARNING: This will copy the data of the original property!
        '''
        cpy = copy.deepcopy(self)

        for i, var in enumerate(cpy._variables):
            cpy._variables[i] = var.in_units(units)

        cpy._units = units

        return cpy
