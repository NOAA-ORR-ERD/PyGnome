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
    '''
    This class represents a phenomenon using a time series
    '''

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 data=None,
                 extrapolate=False):
        if len(time) != len(data):
            raise ValueError("Time and data sequences are of different length.\n\
            len(time) == {0}, len(data) == {1}".format(len(time), len(data)))
        super(TimeSeriesProp, self).__init__(name, units, time, data, extrapolate)
        self.time = time

    @property
    def timeseries(self):
        return map(lambda x,y:(x,y), self.time.time, self.data)

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
        elif isinstance(t,collections.Iterable):
            self._time = Time(t)
        else:
            raise ValueError("Object being assigned must be an iterable or a Time object")

    def set_attr(self,
               name=None,
               units=None,
               time=None,
               data=None,
               extrapolate=None):
        self.name = name if name is not None else self.name
        self.units = units if units is not None else self.units
        self.extrapolate = extrapolate if extrapolate is not None else self.extrapolate
        if data is not None and time is not None:
            if len(time) != len(data):
                raise ValueError("Data/time interval mismatch")
            self._data = data
            self.time = time
        else:
            self.data = data if data is not None else self.data
            self.time = time if time is not None else self.time

    def at(self, points, time, units=None):
        '''
        Interpolates this property to the given points at the given time with the units specified
        :param points: A Nx2 array of lon,lat points
        :param time: A datetime object. May be None; if this is so, the variable is assumed to be gridded
        but time-invariant
        :param units: The units that the result would be converted to
        '''
        value = None
        if len(self.time) == 1:
            #single time time series (constant)
            value = np.full((points.shape[0], 1), self.data)
            value = unit_conversion.convert(None, self.units, units, value)
            return value

        if not self.extrapolate:
            self.time.valid_time(time)
        t_index = self.time.index_of(time)
        if time > self.time.max_time:
            value = self.data[-1]
        if time <= self.time.min_time:
            value = self.data[0]
        if value is None:
            t_alphas = self.time.interp_alpha(time)

            d0 = self.data[t_index - 1]
            d1 = self.data[t_index]
            value = d0 + (d1 - d0) * t_alphas
        if units is not None and units != self.units:
            value = unit_conversion.convert(None, self.units, units, value)

        return np.full((points.shape[0], 1), value)

    def __eq__(self, o):
        t1 = (self.name == o.name and
              self.units == o.units and
              self.extrapolate == o.extrapolate and
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
                 varnames=None,
                 extrapolate=False):

        if any([units is None, time is None]) and not all([isinstance(v, TimeSeriesProp) for v in variables]):
            raise ValueError("All attributes except name, varnames and extrapolate MUST be defined if variables is not a list of TimeSeriesProp objects")

        if variables is None or len(variables) < 2:
            raise TypeError('Variables must be an array-like of 2 or more TimeSeriesProp or array-like')
        VectorProp.__init__(self, name, units, time, variables, extrapolate)
        self._check_consistency()

    def _check_consistency(self):
        '''
        Checks that the attributes of each GriddedProp in varlist are the same as the GridVectorProp
        '''
        for v in self.variables:
            if (v.units != self.units or
                v.time != self.time or
                v.extrapolate != self.extrapolate):
                raise ValueError("Variable {0} did not have parameters consistent with what was specified".format(v.name))

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, vars):
        new_vars = []
        for i, var in enumerate(vars):
            if not isinstance(var, TimeSeriesProp):
                if isinstance(var, collections.Iterable) and len(var) == len(self.time):
                    new_vars.append(TimeSeriesProp(name='var{0}'.format(i),
                                                          units=self.units, time=self.time,
                                                          data = vars[i],
                                                          extrapolate=self.extrapolate))
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
        elif isinstance(t,collections.Iterable):
            self._time = Time(t)
        else:
            raise ValueError("Object being assigned must be an iterable or a Time object")

    def set_attr(self,
               name=None,
               units=None,
               time=None,
               variables=None,
               extrapolate=None):
        self.name = name if name is not None else self.name
        self.units = units if units is not None else self.units
        if variables is not None and time is not None:
            new_vars = []
            for i, var in enumerate(variables):
                if not isinstance(var, TimeSeriesProp):
                    if isinstance(var, collections.Iterable) and len(var) == len(time):
                        new_vars.append(TimeSeriesProp(name = 'var{0}'.format(i),
                                                              units = self.units, time=time,
                                                              data = variables[i],
                                                              extrapolate = extrapolate))
                    else:
                        raise ValueError('Variables must contain iterables or TimeSeriesProp objects')
            self._variables = new_vars
            self.time = time
        else:
            if variables is not None:
                self.variables = variables
            self.time = time if time is not None else self.time
        self.extrapolate = extrapolate if extrapolate is not None else self.extrapolate

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
