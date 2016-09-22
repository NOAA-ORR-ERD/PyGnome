import warnings
import copy

import netCDF4 as nc4
import numpy as np

from gnome.environment.property import EnvProp, VectorProp, Time
from datetime import datetime, timedelta
from dateutil import parser
from colander import SchemaNode, Float, Boolean, Sequence, MappingSchema, drop, String, OneOf, SequenceSchema, TupleSchema, DateTime

import unit_conversion
import collections


class TimeSeriesProp(EnvProp):

    def __init__(self,

                 name=None,
                 units=None,
                 time=None,
                 data=None):
        '''
        A class that represents a scalar natural phenomenon using a time series

        :param name: Name
        :param units: Units
        :param time: Time axis of the data
        :param data: Underlying data source
        :type name: string
        :type units: string
        :type time: [] of datetime.datetime, netCDF4.Variable, or Time object
        :type data: numpy.array, list, or other iterable
        '''
        if len(time) != len(data):
            raise ValueError("Time and data sequences are of different length.\n\
            len(time) == {0}, len(data) == {1}".format(len(time), len(data)))
        super(TimeSeriesProp, self).__init__(name, units, time, data)
        self.time = time

    @property
    def timeseries(self):
        '''
        Creates a representation of the time series

        :rtype: list of (datetime, double) tuples
        '''
        return map(lambda x, y: (x, y), self.time.time, self.data)

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
            raise ValueError("Data/time interval mismatch")
        if isinstance(t, Time):
            self._time = t
        elif isinstance(t, collections.Iterable):
            self._time = Time(t)
        else:
            raise ValueError("Object being assigned must be an iterable or a Time object")

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

    def at(self, points, time, units=None, extrapolate=False):
        '''
        Interpolates this property to the given points at the given time with the units specified
        :param points: A Nx2 array of lon,lat points
        :param time: A datetime object. May be None; if this is so, the variable is assumed to be gridded
        but time-invariant
        :param units: The units that the result would be converted to
        '''
        value = None
        if len(self.time) == 1:
            # single time time series (constant)
            value = np.full((points.shape[0], 1), self.data)
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

        return np.full((points.shape[0], 1), value)

    def __eq__(self, o):
        t1 = (self.name == o.name and
              self.units == o.units and
              self.time == o.time)
        t2 = all(np.isclose(self.data, o.data))
        return t1 and t2

    def __ne__(self, o):
        return not self.__eq__(o)


class TSVectorProp(VectorProp):

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 varnames=None):
        '''
        This class represents a vector phenomenon using a time series
        '''

        if any([units is None, time is None]) and not all([isinstance(v, TimeSeriesProp) for v in variables]):
            raise ValueError("All attributes except name, varnames MUST be defined if variables is not a list of TimeSeriesProp objects")

        if variables is None or len(variables) < 2:
            raise TypeError('Variables must be an array-like of 2 or more TimeSeriesProp or array-like')
        VectorProp.__init__(self, name, units, time, variables)
        self._check_consistency()

    def _check_consistency(self):
        '''
        Checks that the attributes of each GriddedProp in varlist are the same as the GridVectorProp
        '''
        for v in self.variables:
            if (v.units != self.units or
                    v.time != self.time):
                raise ValueError("Variable {0} did not have parameters consistent with what was specified".format(v.name))

    @property
    def timeseries(self):
        '''
        Creates a representation of the time series

        :rtype: list of (datetime, (double, double)) tuples
        '''
        return map(lambda x, y, z: (x, (y, z)), self.time.time, self.variables[0], self.variables[1])

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, vs):
        new_vars = []
        for i, var in enumerate(vs):
            if not isinstance(var, TimeSeriesProp):
                if isinstance(var, collections.Iterable) and len(var) == len(self.time):
                    new_vars.append(TimeSeriesProp(name='var{0}'.format(i),
                                                   units=self.units, time=self.time,
                                                   data=vs[i]))
                else:
                    raise ValueError('Variables must contain iterables or TimeSeriesProp objects')
            else:
                new_vars.append(var)
        self._variables = new_vars
        self._check_consistency()

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
            raise ValueError("Object being assigned must be an iterable or a Time object")

    def set_attr(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None):
        self.name = name if name is not None else self.name
        self.units = units if units is not None else self.units
        if variables is not None and time is not None:
            new_vars = []
            for i, var in enumerate(variables):
                if not isinstance(var, TimeSeriesProp):
                    if isinstance(var, collections.Iterable) and len(var) == len(time):
                        new_vars.append(TimeSeriesProp(name='var{0}'.format(i),
                                                       units=self.units, time=time,
                                                       data=variables[i]))
                    else:
                        raise ValueError('Variables must contain iterables or TimeSeriesProp objects')
            self._variables = new_vars
            self.time = time
        else:
            if variables is not None:
                self.variables = variables
            self.time = time if time is not None else self.time

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
