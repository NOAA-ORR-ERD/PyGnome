import warnings
import copy

import netCDF4 as nc4
import numpy as np

from datetime import datetime, timedelta
from colander import SchemaNode, Float, Boolean, Sequence, MappingSchema, drop, String, OneOf, SequenceSchema, TupleSchema, DateTime
from gnome.persist.base_schema import ObjType
from gnome.utilities import serializable
from gnome.movers import ProcessSchema
from gnome.persist import base_schema

import pyugrid
import pysgrid
import unit_conversion
import collections


class PropertySchema(base_schema.ObjType):
    name = SchemaNode(String(), missing='default')
    units = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String(), missing=drop),SchemaNode(String(), missing=drop)])
    time = SequenceSchema(SchemaNode(DateTime(default_tzinfo=None), missing=drop), missing=drop)


class EnvProp(object):
    '''
    A class that represents a natural phenomenon and provides an interface to get
    the value of the phenomenon at a position in space and time. EnvProp is the base
    class, and returns only a single value regardless of the time
    '''

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 data=None,
                 **kwargs):
        '''
        EnvProp base class constructor
        :param name: Name
        :param units: Units
        :param time: Array of datetime objects, netCDF4 Variable, or gnome.environment.property.Time object
        :param data: Value of the property
        '''

        self.name = self._units = self._time = self._data = None

        self.name = name
        if units in unit_conversion.unit_data.supported_units:
            self._units = units
        else:
            raise ValueError('Units of {0} are not supported'.format(units))
        self.data = data
        self.time = time
        for k in kwargs:
            setattr(self, k, kwargs[k])

    '''
    Subclasses should override\add any attribute property function getter/setters as needed
    '''

    @property
    def data(self):
        return self._data

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, unit):
        if unit in unit_conversion.unit_data.supported_units:
            self._units = unit
        else:
            raise ValueError('Units of {0} are not supported'.format(unit))

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        if isinstance(t, Time):
            self._time = t
        elif isinstance(t,collections.Iterable):
            self._time = Time(t)
        else:
            raise ValueError("Object being assigned must be an iterable or a Time object")

    def at(self, points, time, extrapolate):
        raise NotImplementedError()

    def in_units(self, unit):
        '''
        Returns a full cpy of this property in the units specified. 
        WARNING: This will cpy the data of the original property!
        '''
        cpy = copy.copy(self)
        if hasattr(cpy.data, '__mul__'):
            cpy.data = unit_conversion.Convert(None, cpy.units, unit, cpy.data)
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
        self.time=time
        for k in kwargs:
            setattr(self, k, kwargs[k])
        self.variables = variables

    @property
    def time(self):
        return self._time

    @property
    def units(self):
        if hasattr(self._units, '__iter__'):
            if len(set(self._units) > 1):
                return self._units
            else:
                return self._units[0]
        else:
            return self._units

    @units.setter
    def units(self, unit):
        if unit in unit_conversion.unit_data.supported_units:
            self._units = unit
        else:
            raise ValueError(5,'Units of {0} are not supported'.format(unit))

    @property
    def varnames(self):
        return [v.name for v in self.variables]

    def _check_consistency(self):
        '''
        Checks that the attributes of each GriddedProp in varlist are the same as the GridVectorProp
        '''
        raise NotImplementedError()

    def at(self, *args, **kwargs):
        return np.column_stack([var.at(*args, **kwargs) for var in self._variables])


class Time(object):

    def __init__(self, time_seq, tz_offset=None, offset=None):
        '''
        Functions for a time array
        :param time_seq: An ascending array of datetime objects of length N
        '''
        if isinstance(time_seq, nc4.Variable):
            self.time = nc4.num2date(time_seq[:], units=time_seq.units)
        else:
            self.time = time_seq

        if tz_offset is not None:
            self.time += tz_offset

        if offset is not None:
            delta = timedelta(**offset)
            self.time += delta
#         if not self._timeseries_is_ascending(self.time):
#             raise ValueError("Time sequence is not ascending")
#         if self._has_duplicates(self.time):
#             raise ValueError("Time sequence has duplicate entries")

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

    def _has_duplicates(self, ts):
        return len(np.unique(ts)) != len(ts)

    @property
    def min_time(self):
        return self.time[0]

    @property
    def max_time(self):
        return self.time[-1]

    def get_time_array(self):
        return self.time[:]

    def time_in_bounds(self, time):
        return not time < self.min_time or time > self.max_time

    def valid_time(self, time):
        if time < self.min_time or time > self.max_time:
            raise ValueError('time specified ({0}) is not within the bounds of the time ({1} to {2})'.format(
                time.strftime('%c'), self.min_time.strftime('%c'), self.max_time.strftime('%c')))

    def index_of(self, time, extrapolate):
        '''
        Returns the index of the provided time with respect to the time intervals in the file.
        :param time:
        :return:
        '''
        if not (extrapolate or len(self.time) == 1):
            self.valid_time(time)
        index = np.searchsorted(self.time, time)
        return index

    def interp_alpha(self, time, extrapolate=False):
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
