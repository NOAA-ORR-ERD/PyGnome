import warnings
import copy

import netCDF4 as nc4
import numpy as np

from datetime import datetime, timedelta
from colander import SchemaNode, Float, Boolean, Sequence, MappingSchema, drop, String, OneOf, SequenceSchema, TupleSchema, DateTime
from gnome.persist.base_schema import ObjType
from gnome.utilities import serializable
from gnome.persist import base_schema

import pyugrid
import pysgrid
import unit_conversion
import collections


class PropertySchema(base_schema.ObjType):
    name = SchemaNode(String(), missing='default')
    units = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String(), missing=drop), SchemaNode(String(), missing=drop)])
    time = SequenceSchema(SchemaNode(DateTime(default_tzinfo=None), missing=drop), missing=drop)


class EnvProp(object):

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 data=None,
                 **kwargs):
        '''
        A class that represents a natural phenomenon and provides an interface to get
        the value of the phenomenon at a position in space and time. EnvProp is the base
        class, and returns only a single value regardless of the time.

        :param name: Name
        :param units: Units
        :param time: Time axis of the data
        :param data: Value of the property
        :type name: string
        :type units: string
        :type time: [] of datetime.datetime, netCDF4.Variable, or Time object
        :type data: netCDF4.Variable or numpy.array
        '''

        self.name = self._units = self._time = self._data = None

        self.name = name
        self.units = units
        self.data = data
        self.time = time
        for k in kwargs:
            setattr(self, k, kwargs[k])

    '''
    Subclasses should override\add any attribute property function getter/setters as needed
    '''

    @property
    def data(self):
        '''
        Underlying data

        :rtype: netCDF4.Variable or numpy.array
        '''
        return self._data

    @property
    def units(self):
        '''
        Units of underlying data

        :rtype: string
        '''
        return self._units

    @units.setter
    def units(self, unit):
        if unit is not None:
            if not unit_conversion.is_supported(unit):
                raise ValueError('Units of {0} are not supported'.format(unit))
        self._units = unit

    @property
    def time(self):
        '''
        Time axis of data

        :rtype: gnome.environment.property.Time
        '''
        return self._time

    @time.setter
    def time(self, t):
        if isinstance(t, Time):
            self._time = t
        elif isinstance(t, collections.Iterable):
            self._time = Time(t)
        else:
            raise ValueError("Object being assigned must be an iterable or a Time object")

    def at(self, *args, **kwargs):
        '''
        Find the value of the property at positions P at time T

        :param points: Coordinates to be queried (P)
        :param time: The time at which to query these points (T)
        :param time: Specifies the time level of the variable
        :param units: units the values will be returned in (or converted to)
        :param extrapolate: if True, extrapolation will be supported
        :type points: Nx2 array of double
        :type time: datetime.datetime object
        :type time: integer
        :type units: string such as ('m/s', 'knots', etc)
        :type extrapolate: boolean (True or False)
        :return: returns a Nx1 array of interpolated values
        :rtype: double
        '''

        raise NotImplementedError()

    def in_units(self, unit):
        '''
        Returns a full cpy of this property in the units specified.
        WARNING: This will cpy the data of the original property!

        :param units: Units to convert to
        :type units: string
        :return: Copy of self converted to new units
        :rtype: Same as self
        '''
        cpy = copy.copy(self)
        if hasattr(cpy.data, '__mul__'):
            cpy.data = unit_conversion.convert(cpy.units, unit, cpy.data)
        else:
            warnings.warn('Data was not converted to new units and was not copied because it does not support multiplication')
        cpy._units = unit
        return cpy


class VectorProp(object):

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 **kwargs):
        '''
        A class that represents a vector natural phenomenon and provides an interface to get the value of
        the phenomenon at a position in space and time. VectorProp is the base class

        :param name: Name of the Property
        :param units: Unit of the underlying data
        :param time: Time axis of the data
        :param variables: component data arrays
        :type name: string
        :type units: string
        :type time: [] of datetime.datetime, netCDF4.Variable, or Time object
        :type variables: [] of EnvProp or numpy.array (Max len=2)
        '''

        self.name = self._units = self._time = self._variables = None

        self.name = name

        if all([isinstance(v, EnvProp) for v in variables]):
            if time is not None and not isinstance(time, Time):
                time = Time(time)
            units = variables[0].units if units is None else units
            time = variables[0].time if time is None else time
            for v in variables:
                if (v.units != units or
                        v.time != time):
                    raise ValueError("Variable {0} did not have parameters consistent with what was specified".format(v.name))

        if units is None:
            units = variables[0].units
        self._units = units
        if variables is None or len(variables) < 2:
            raise ValueError('Variables must be an array-like of 2 or more Property objects')
        self.time = time
        for k in kwargs:
            setattr(self, k, kwargs[k])
        self.variables = variables

    @property
    def time(self):
        '''
        Time axis of data

        :rtype: gnome.environment.property.Time
        '''
        return self._time

    @property
    def units(self):
        '''
        Units of underlying data

        :rtype: string
        '''
        if hasattr(self._units, '__iter__'):
            if len(set(self._units) > 1):
                return self._units
            else:
                return self._units[0]
        else:
            return self._units

    @units.setter
    def units(self, unit):
        if unit is not None:
            if not unit_conversion.is_supported(unit):
                raise ValueError('Units of {0} are not supported'.format(unit))
        self._units = unit
        if self.variables is not None:
            for v in self.variables:
                v.units = unit

    @property
    def varnames(self):
        '''
        Names of underlying variables

        :rtype: [] of strings
        '''
        return [v.name for v in self.variables]

    def _check_consistency(self):
        '''
        Checks that the attributes of each GriddedProp in varlist are the same as the GridVectorProp
        '''
        raise NotImplementedError()

    def at(self, *args, **kwargs):
        '''
        Find the value of the property at positions P at time T

        :param points: Coordinates to be queried (P)
        :param time: The time at which to query these points (T)
        :param time: Specifies the time level of the variable
        :param units: units the values will be returned in (or converted to)
        :param extrapolate: if True, extrapolation will be supported
        :type points: Nx2 array of double
        :type time: datetime.datetime object
        :type time: integer
        :type units: string such as ('m/s', 'knots', etc)
        :type extrapolate: boolean (True or False)
        :return: returns a Nx2 array of interpolated values
        :rtype: double
        '''
        return np.column_stack([var.at(*args, **kwargs) for var in self._variables])


class Time(object):

    def __init__(self, time_seq, tz_offset=None, offset=None):
        '''
        Representation of a time axis. Provides interpolation alphas and indexing.

        :param time_seq: Ascending list of times to use
        :param tz_offset: offset to compensate for time zone shifts
        :type time_seq: netCDF4.Variable or [] of datetime.datetime
        :type tz_offset: datetime.timedelta

        '''
        if isinstance(time_seq, (nc4.Variable, nc4._netCDF4._Variable)):
            self.time = nc4.num2date(time_seq[:], units=time_seq.units)
        else:
            self.time = time_seq

        if tz_offset is not None:
            self.time += tz_offset

        if not self._timeseries_is_ascending(self.time):
            raise ValueError("Time sequence is not ascending")
        if self._has_duplicates(self.time):
            raise ValueError("Time sequence has duplicate entries")

    @classmethod
    def time_from_nc_var(cls, var):
        return cls(nc4.num2date(var[:], units=var.units))

    def __len__(self):
        return len(self.time)

    def __iter__(self):
        return self.time.__iter__()

    def __eq__(self, other):
        r = self.time == other.time
        return all(r) if hasattr(r, '__len__') else r

    def __ne__(self, other):
        return not self.__eq__(other)

    def _timeseries_is_ascending(self, ts):
        return all(np.sort(ts) == ts)

    def _has_duplicates(self, time):
        return len(np.unique(time)) != len(time) and len(time) != 1

    @property
    def min_time(self):
        '''
        First time in series

        :rtype: datetime.datetime
        '''
        return self.time[0]

    @property
    def max_time(self):
        '''
        Last time in series

        :rtype: datetime.datetime
        '''
        return self.time[-1]

    def get_time_array(self):
        return self.time[:]

    def time_in_bounds(self, time):
        '''
        Checks if time provided is within the bounds represented by this object.

        :param time: time to be queried
        :type time: datetime.datetime
        :rtype: boolean
        '''
        return not time < self.min_time or time > self.max_time

    def valid_time(self, time):
        if time < self.min_time or time > self.max_time:
            raise ValueError('time specified ({0}) is not within the bounds of the time ({1} to {2})'.format(
                time.strftime('%c'), self.min_time.strftime('%c'), self.max_time.strftime('%c')))

    def index_of(self, time, extrapolate):
        '''
        Returns the index of the provided time with respect to the time intervals in the file.

        :param time: Time to be queried
        :param extrapolate:
        :type time: datetime.datetime
        :type extrapolate: boolean
        :return: index of first time before specified time
        :rtype: integer
        '''
        if not (extrapolate or len(self.time) == 1):
            self.valid_time(time)
        index = np.searchsorted(self.time, time)
        return index

    def interp_alpha(self, time, extrapolate=False):
        '''
        Returns interpolation alpha for the specified time

        :param time: Time to be queried
        :param extrapolate:
        :type time: datetime.datetime
        :type extrapolate: boolean
        :return: interpolation alpha
        :rtype: double (0 <= r <= 1)
        '''
        if not len(self.time) == 1 or not extrapolate:
            self.valid_time(time)
        i0 = self.index_of(time, extrapolate)
        if i0 > len(self.time) - 1:
            return 1
        if i0 == 0:
            return 0
        t0 = self.time[i0 - 1]
        t1 = self.time[i0]
        return (time - t0).total_seconds() / (t1 - t0).total_seconds()
