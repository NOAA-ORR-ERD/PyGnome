import warnings
import copy

import netCDF4 as nc4
import numpy as np

from gnome.utilities.geometry.cy_point_in_polygon import points_in_polys
from datetime import datetime, timedelta
from dateutil import parser
from colander import SchemaNode, Float, Boolean, Sequence, MappingSchema, drop, String, OneOf, SequenceSchema, TupleSchema, DateTime
from gnome.persist.base_schema import ObjType
from gnome.utilities import serializable
from gnome.movers import ProcessSchema
from gnome.persist import base_schema

from gnome.utilities.timeseries_generic import DataTimeSeries

import pyugrid
import pysgrid
import unit_conversion
import collections


class EnvProp(object):
    '''
    A class that represents a natural phenomenon and provides an interface to get
    the value of the phenomenon with respect to space and time. EnvProp is the base
    class, and returns only a single value regardless of the time
    '''

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 data=None,
                 extrapolate=False):


        self.name = name
        if units in unit_conversion.unit_data.supported_units:
            self._units = units
        else:
            raise ValueError('Units of {0} are not supported'.format(units))
        self._time = time
        self._data = data
        self._extrapolate = extrapolate

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, unit):
        if unit in unit_conversion.unit_data.supported_units:
            self._units = unit
        else:
            raise ValueError('Units of {0} are not supported'.format(unit))

    def at(self, points, time):
        return np.full((points.shape[0], 1), data)

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
        if len(d) != len(self.time):
            raise ValueError("Data/time interval mismatch")
        self._data = d

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        if len(t) != len(self.data):
            raise ValueError("Data/time interval mismatch")
        if isinstance(t, Time):
            self._time = t
        elif isinstance(t,collections.Iterable):
            self._time = Time(t)
        else:
            raise ValueError("Object being assigned must be an iterable or a Time object")

    @property
    def extrapolate(self):
        return self._extrapolate

    @extrapolate.setter
    def extrapolate(self, e):
        self._extrapolate = e
        self.time.extrapolate = e

    def set_ts(self,
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
            return np.full((points.shape[0], 1), data)
        elif self.extrapolate and time >= self.time.max_time:
            value = self.data[-1]
        elif self.extrapolate and time <= self.time.min_time:
            value = self.data[0]
        else:
            t_index = self.time.indexof(time)
            t_alphas = self.time.interp_alpha(time)
            d0 = self.data[t_index]
            d1 = self.data[t_index + 1]
            value = d0 + (d1 - d0) * t_alphas
        if units is not None and units != self.units:
            value = unit_conversion.convert(None, self.units, units, value)

        return np.full((points.shape[0], 1), value)


class GriddedProp(EnvProp):
    '''
    This class represents a phenomenon using gridded data
    '''

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 data=None,
                 grid=None,
                 extrapolate=False,
                 data_file=None,
                 grid_file=None):
        if grid is None or data is None:
            raise ValueError('Must provide a grid and data that can fit to the grid')
        if grid.infer_grid(data) is None:
            raise ValueError('Data must be able to fit to the grid')
        super(GriddedProp, self).__init__(name=name, units=units, data=data, extrapolate=extrapolate)
        self.grid = grid
        self.time = Time(time)
        self.data_file = data_file
        self.grid_file = grid_file

    @property
    def time(self):
        return self._time


    @time.setter
    def time(self, t):
        if len(t) != len(self.data):
            raise ValueError("Data and time array must be the same length")
        if isinstance(t,collections.Iterable) or isinstance(t, nc4.Variable):
            t = Time(t)
        else:
            raise ValueError("Time must be set with an iterable container or netCDF variable")
        if len(t) != self.data.shape[0]:
            raise ValueError("Time provided is incompatible with data source time dimension")
        self._time = t


    @property
    def data(self):
        return self._data

    def set_data(self, d, datafile, grid=None, gridfile=None):
        if grid is not None:
            if gridfile is None:
                raise ValueError('Gridfile needs to be provided')
            if d.shape[-2:] != grid.shape or grid.infer_grid(d) is None:
                raise ValueError("Data shape is incompatible with grid shape.")
            self.grid = grid
            self.gridfile = gridfile
        if d.shape[-2:] != self.grid.shape or self.grid.infer_grid(d) is None:
            raise ValueError("Data shape is incompatible with grid shape.")
        if d.shape[0] != len(self.time):
            raise ValueError("Data time dimension is incompatible with time series")
        self.data = d
        self.datafile = datafile


    def at(self, points, time, units=None):
        '''
        Interpolates this property to the given points at the given time.
        :param points: A Nx2 array of lon,lat points
        :param time: A datetime object. May be None; if this is so, the variable is assumed to be gridded
        but time-invariant
        '''
        t_alphas = t_index = s0 = s1 = value = None
        if self.time is not None:
            t_index = self.time.indexof(time)
            if self.extrapolate and t_index == len(self.time.time):
                s0 = [t_index]
                value = self.grid.interpolate_var_to_points(points, self.data, slices=s0, memo=True)
            else:
                t_alphas = self.time.interp_alpha(time)
                s0 = [t_index]
                s1 = [t_index + 1]
                if len(self.data.shape) == 4:
                    s1.append(depth)
                    s2.append(depth)
                v0 = self.grid.interpolate_var_to_points(points, self.data, slices=s0, memo=True)
                v1 = self.grid.interpolate_var_to_points(points, self.data, slices=s1, memo=True)
                value = v0 + (v1 - v0) * t_alphas
        else:
            s0 = None
            value = self.grid.interpolate_var_to_points(points, self.data, slices=s0, memo=True)

        if units is not None and units != self.units:
            value = unit_conversion.convert(None, self.units, units, value)
        return value

class VectorProp(object):

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 extrapolate=False):
        self.name = name
        if variables is None or len(variables) < 2:
            raise ValueError('Variables must be an array-like of 2 or more Property objects')
        self.variables = variables
        if units is None:
            units = variables[0].units
        self._units = units
        self.time=time

        self.extrapolate = extrapolate

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
            raise ValueError('Units of {0} are not supported'.format(unit))

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        print self._variables
        if len(t) != len(self._variables[0].data):
            raise ValueError("Data/time interval mismatch")
        if isinstance(t, Time):
            self._time = t
        elif isinstance(t,collections.Iterable):
            self._time = Time(t)
        else:
            raise ValueError("Object being assigned must be an iterable or a Time object")

    @property
    def variables(self):
        return np.column_stack([v.data for v in self._variables])

    @variables.setter
    def variables(self, vars):
        self.variables = vars


    @property
    def varnames(self):
        return [v.name for v in self.variables]

    def at(self, points, time, units=None):
        return np.column_stack((var.at(points, time, units) for var in self._variables))


class TSVectorProp(VectorProp):

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 extrapolate=False):

        self.name = name
        if variables is None or len(variables) < 2:
            raise ValueError('Variables must be an array-like of 2 or more Property objects')
        if units is None:
            if isinstance(variables[0], EnvProp):
                units = variables[0].units
            else:
                raise ValueError('Need to specify units (cannot infer from variables provided)')
        self.units = units
        self._extrapolate = extrapolate
        self.set_ts(time = time, variables=variables)

    @property
    def variables(self):
        return np.stack([v.data for v in self._variables])

    @variables.setter
    def variables(self, vars):
        print self.units
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
        self._variables = new_vars

    @property
    def extrapolate(self):
        return self._extrapolate

    @extrapolate.setter
    def extrapolate(self, e):
        self._extrapolate = e
        for v in self._variables:
            v.extrapolate = e
        self.time.extrapolate = e

    def set_ts(self,
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

class GridVectorProp(VectorProp):

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 extrapolate=False):

        self.data_file = variables[0].data_file
        self.grid_file = variables[0].grid_file

        for var in variables:
            if not isinstance(var, GriddedProp):
                raise ValueError('All variables must be GriddedProp objects')
            if var.data_file != self.data_file:
                raise ValueError("""Data filename for component property {0} is different 
                                 than reference datafile {1}""".format(var.data_file, self.data_file))
            if var.grid_file != self.grid_file:
                raise ValueError("""Grid filename for component property {0} is different 
                                     than reference gridfile {1}""".format(var.grid_file, self.grid_file))
        VectorProp.__init__(name, units, time, variables, extrapolate)


class Time(object):

    def __init__(self, time_seq, extrapolate=False):
        '''
        Functions for a time array
        :param time_seq: An ascending array of datetime objects of length N
        '''
        if isinstance(time_seq, nc4.Variable):
            self.time = nc4.num2date(time_seq[:], units=time_seq.units)
        else:
            self.time = time_seq

#         if not self._timeseries_is_ascending(self.time):
#             raise ValueError("Time sequence is not ascending")
#         if self._has_duplicates(self.time):
#             raise ValueError("Time sequence has duplicate entries")
        self.extrapolate = False

    @classmethod
    def time_from_nc_var(cls, var):
        return cls(nc4.num2date(var[:], units=var.units))

    def __len__(self):
        return len(self.time)

    def __iter__(self):
        return self.time.__iter__()

    def __eq__(self, other):
        return (self.time == other.time).all()

    def __ne__(self, other):
        return (self.time != other.time).all()

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

    def indexof(self, time):
        '''
        Returns the index of the provided time with respect to the time intervals in the file.
        :param time:
        :return:
        '''
        if not self.extrapolate:
            self.valid_time(time)
        index = np.searchsorted(self.time, time) - 1
        return index

    def interp_alpha(self, time):
        if not self.extrapolate:
            self.valid_time(time)
        i0 = self.indexof(time)
        if i0 == len(self.time):
            return 1
        t0 = self.time[i0]
        t1 = self.time[i0 + 1]
        return (time - t0).total_seconds() / (t1 - t0).total_seconds()

if __name__ == "__main__":
    pass
