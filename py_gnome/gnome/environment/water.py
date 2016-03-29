import warnings

import netCDF4 as nc4
import numpy as np

from collections import namedtuple

from gnome.utilities.geometry.cy_point_in_polygon import points_in_polys
from datetime import datetime, timedelta
from dateutil import parser
from colander import SchemaNode, Float, MappingSchema, drop, String, OneOf
from gnome.persist.base_schema import ObjType
from gnome.utilities import serializable
from gnome.movers import ProcessSchema

from gnome.utilities.timeseries_generic import DataTimeSeries

import pyugrid
import pysgrid


class EnvProperty(object):

    def __init__(self,
                 name=None,
                 units=None,
                 ** kwargs):
        self.name = name
        self.units = units

    @classmethod
    def constant_var(cls, name=None, value=None):
        if name is None or value is None:
            raise ValueError("Name and value must be provided")
        return cls(name=name, is_constant=True, constant_val=value)

    @classmethod
    def gridded_var(cls, name=None, grid=None, data=None, time=None):
        if grid is None or data_source is None:
            raise ValueError('Must provide a grid and data source that can fit to the grid')
        if grid.infer_grid(data_source) is None:
            raise ValueError('Data source must be able to fit to the grid')
        return cls(name=name, is_constant=False, grid=grid, data=data, time=time)


class TimeSeriesProp(EnvProperty):

    def __init__(self,
                 name=None,
                 units=None,
                 data=None,
                 times=None):
        super(TimeSeriesProp, self).__init__(name, units)
        if len(time) != len(data):
            raise ValueError("Time and data sequences are of different length.\n\
            len(time_seq) == {0}, len(data_seq) == {1}".format(len(time_seq), len(data_seq)))

        self.time = Time(times)
        self.data = data_seq

    def at(self, points, time):
        t_alphas = self.time.interp_alpha(time)
        t_index = self.time.indexof(time)
        d0 = self.data[t_index]
        d1 = self.data[t_index + 1]
        value = d0 + (d1 - d0) * t_alphas
        return np.full((points.shape[0], 1), value)


class GriddedProp(EnvProperty):

    def __init__(self,
                 name=None,
                 units=None,
                 grid=None,
                 data=None,
                 times=None):
        super(TimeSeriesProp, self).__init__(name, units)
        self.grid = grid
        self.data = data
        self.time = time

    def at(self, points, time):
        '''
        Interpolates this property to the given points at the given time.
        :param points: A Nx2 array of lon,lat points
        :param time: A datetime object. May be None; if this is so, the variable is assumed to be gridded
        but time-invariant
        '''
        if self.is_constant:
            return np.full((points.shape[0], 1), self.constant_val)
        if self.is_timeseries:
            # TBD
            return None
        if self.is_gridded:
            t_alphas = t_index = s0 = s1 = None
            if time is not None:
                t_alphas = self.time.interp_alpha(time)
                t_index = self.time.indexof(time)
                s0 = [t_index]
                s1 = [t_index + 1]
                if len(variable.shape) == 4:
                    s1.append(depth)
                    s2.append(depth)
                v0 = self.grid.interpolate_var_to_poitns(points, self.data, slices=s0, memo=True)
                v1 = self.grid.interpolate_var_to_points(points, self.data, slices=s2, memo=True)
            else:
                s0 = None
                v0 = self.grid.interpolate_var_to_points(points, self.data, slices=s0, memo=True)

            vt = v0 + (v1 - v0) * t_alphas
            return vt
        warnings.warn("Type of property is unspecified. No data is indicated to be active \
        (is_constant, is_gridded, is_timeseries == False)")
        return None

    # Should vector quantities even be supported?
#     @classmethod
#     def constant_vec(cls, name=None, vector=None, magnitude=None, direction=None):
#         if name is None:
#             raise ValueError('Name must be provided')
#         if vector is None:
#             if magnitude is None or direction is None:
#                 raise ValueError('If vector is not provided, magnitude and direction must be')
#             else:
#                 return cls(name=name, is_constant=True, constant_val=magnitude, constant_direction=direction)
#         else:
#             vector = np.asarray(vector, dtype=np.double)
#             if magnitude is not None or direction is not None:
#                 warnings.warn("vector is defined, ignoring magnitude and direction")
#             if vector.shape != (2,) or vector.shape != (3,):
#                 raise ValueError("Must provide [u,v] or [u,v,w] for vector")
#             return cls(name=name, is_constant=True, constant_vec=vector)
#


class WaterConditions(object):

    validprops = ['temperature',
                  'salinity',
                  'velocity',
                  'angles']

    def __init__(self, **kwargs):
        for k in kwargs.keys():
            if k not in WaterConditions.validprops:
                raise ValueError('Property {0} is not part of this environment object\n \
                Valid properties are: {1}'.format(k, WaterConditions.validprops))
            prop = kwargs.get(k, None)
            if prop is not None:
                if isinstance(k, EnvProperty):
                    self.__setattr__(k,)
        temp = kwargs.get('temperature', None)
        salinity = kwargs.get('salinity', None)
        vel_u, vel_v = kwargs.get('velocity', (None, None))
        angles = kwargs.get('angles', None)

    @classmethod
    def from_roms(cls, filename):
        '''
        Takes a file that follows ROMS conventions and creates a populated WaterConditions object
        from it.
        '''
        return None

    def set_property(self, property, name=None, value=None, grid=None, time=None):
        if property not in WaterConditions.validprops:
            raise ValueError('Unsupported property property for WaterConditions. \
            Must be one of: {0}'.format(WaterConditions.validprops))
        if isinstance(value, collections.Sequence):
            if grid is None:
                raise ValueError('Cannot create a variable from a non-scalar without \
                specifying grid or time')

        else:
            if grid is not None:
                warnings.warn("Constant value specified, ignoring grid and time")

    @property
    def temperature(self):
        return self.temperature_var

    @temperature.setter
    def temperature(self, value):
        # TODO
        if isinstance(value, collections.Sequence):
            raise ValueError("Temperature value may not be set to a collections.Sequence.\
             Use the appropriate setter function")
        self.temperature_var = EnvProperty.constant_var('Water Temperature', value)

    def temperature_at(self, points, time):
        # if single coordinate pair/triplet
        if points.shape == (2,) or points.shape == (3,):
            points = points.reshape((-1, points.shape[0]))
            # should we only ever use the x/y coordinates? ingore z until later?
        if temperature_var is None:
            # do not fail catastrophically here?
            warnings.warn("Temperature source is not defined")
            return None
        return self.temperature_var.at(points, time)

        # if we get here, temperature variable is invalid
        raise RuntimeError("Temperature var is not constant, timeseries, or gridded!")

    def salinity_at(self, points):
        pass

    def velocity_at(self, points):
        # if single coordinate pair/triplet
        if points.shape == (2,) or points.shape == (3,):
            points = points.reshape((-1, points.shape[0]))
            # should we only ever use the x/y coordinates? ingore z until later?
        if self.velocity_u is None:
            warnings.warn("Velocity u-componenet is not defined")
            return (None, None)
        if self.velocity_v is None:
            warnings.warn("Velocity v-componenet is not defined")
            return (None, None)
        u = self.velocity_u.at(points, time)
        v = self.velocity_v.at(points, time)
        if angles is not None:
            vels = np.column_stack((u, v))
            angs = self.angles.at(points, None)
            rotations = np.array(
                ([np.cos(angs), -np.sin(angs)], [np.sin(angs), np.cos(angs)]))

            return np.matmul(rotations.T, vels[:, :, np.newaxis]).reshape(-1, 2)
        return np.column_stack((u, v))


class Time(object):

    def __init__(self, time_seq):
        '''
        Functions for a time array
        :param time_seq: An ascending array of datetime objects of length N
        '''
        if not self._timeseries_is_ascending(time_seq):
            raise ValueError("Time sequence is not ascending")
        if self._has_duplicates(time_seq):
            raise ValueError("Time sequence has duplicate entries")
        self.time = time_seq

    @classmethod
    def time_from_nc_var(cls, var):
        return cls(nc4.num2date(var[:], units=var.units))

    def _timeseries_is_ascending(self, ts):
        return all(np.sort(ts) == time_seq)

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

#     def valid_time(self, time):
#         if time < self.min_time or time > self.max_time:
#             raise ValueError('time specified ({0}) is not within the bounds of the time ({1} to {2})'.format(
#                 time.strftime('%c'), self.min_time.strftime('%c'), self.max_time.strftime('%c')))

    def indexof(self, time):
        '''
        Returns the index of the provided time with respect to the time intervals in the file.
        :param time:
        :return:
        '''
        self.valid_time(time)
        index = np.searchsorted(self.time, time) - 1
        return index

    def interp_alpha(self, time):
        i0 = self.indexof(time)
        t0 = self.time[i0]
        t1 = self.time[i0 + 1]
        return (time - t0).total_seconds() / (t1 - t0).total_seconds()
