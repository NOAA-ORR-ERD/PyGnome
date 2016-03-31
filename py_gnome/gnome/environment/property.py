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
import unit_conversion


class TimeSeriesProp(object):

    def __init__(self,
                 name,
                 units,
                 data,
                 times):
        self.name = name
        self.units = units
        if len(times) != len(data):
            raise ValueError("Time and data sequences are of different length.\n\
            len(time_seq) == {0}, len(data_seq) == {1}".format(len(time_seq), len(data_seq)))

        self.time = Time(times)
        self.data = data

    def at(self, points, time, units=None):
        t_alphas = self.time.interp_alpha(time)
        t_index = self.time.indexof(time)
        d0 = self.data[t_index]
        d1 = self.data[t_index + 1]
        value = d0 + (d1 - d0) * t_alphas
        if units is not None and units != self.units:
            value = unit_conversion.convert(None, self.units, units, value)

        return np.full((points.shape[0], 1), value)


class GriddedProp(object):

    def __init__(self,
                 name,
                 units,
                 grid,
                 data,
                 time=None):
        if grid is None or data_source is None:
            raise ValueError('Must provide a grid and data source that can fit to the grid')
        if grid.infer_grid(data_source) is None:
            raise ValueError('Data source must be able to fit to the grid')
        self.grid = grid
        self.data = data
        self.time = time

    def at(self, points, time, units=None):
        '''
        Interpolates this property to the given points at the given time.
        :param points: A Nx2 array of lon,lat points
        :param time: A datetime object. May be None; if this is so, the variable is assumed to be gridded
        but time-invariant
        '''
        t_alphas = t_index = s0 = s1 = None
        if self.time is not None:
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
        if units is not None and units != self.units:
            vt = unit_conversion.convert(None, self.units, units, vt)
        return vt


class VectorProp(object):

    def __init__(self,
                 name,
                 units,
                 variables):
        self.name = name
        self.units = units
        for var in variables:
            if var.units != self.units:
                raise ValueError('Units of {0} for component property {1} are not the same as \
                units specified for compound proprety {2}'.format(var.name, var.units, self.units))
        self.variables = variables

    def at(self, points, time, units=None):
        return np.column_stack((var.at(points, time, units) for var in self.variables))


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
        index = np.searchsorted(self.time, time) - 1
        return index

    def interp_alpha(self, time):
        i0 = self.indexof(time)
        t0 = self.time[i0]
        t1 = self.time[i0 + 1]
        return (time - t0).total_seconds() / (t1 - t0).total_seconds()

if __name__ == "__main__":
    import datetime as dt
    dates = np.array([dt.datetime(1, 1, 1, 0), dt.datetime(1, 1, 1, 2), dt.datetime(1, 1, 1, 4)])
    u_data = np.array([2, 4, 6])
    v_data = np.array([5, 7, 9])
    u = TimeSeriesProp('u', 'm/s', u_data, dates)
    v = TimeSeriesProp('v', 'm/s', v_data, dates)

    print u.at(np.array([(1, 1), (1, 2)]), dt.datetime(1, 1, 1, 1))

    vprop = VectorProp('velocity', 'm/s', [u, v])
    print vprop.at(np.array([(1, 1), (1, 2)]), dt.datetime(1, 1, 1, 3), 'knots')
    pass
