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

import pyugrid
import pysgrid


class EnvProperty(object):

    def __init__(self,
                 name=None,
                 is_constant=False,
                 constant_vec=None,
                 constant_val=None,
                 constant_direction=None,
                 is_gridded=False,
                 grid=None,
                 data_source=None,
                 time=None,
                 is_timeseries=False,
                 time_series=None
                 ** kwargs):
        self.name = name
        self.is_constant = is_constant
        self.constant_val = constant_val
        self.constant_vec = constant_vec
        self.constant_direction = constant_direction
        self.is_gridded = is_gridded
        # should all this be rolled into vector field's responsibility?
        self.grid = grid
        self.data = data_source
        self.time = time
        ###
        self.is_timeseries = is_timeseries
        self.time_series = time_series

    @classmethod
    def constant_var(cls, name=None, value=None):
        if name is None or value is None:
            raise ValueError("Name and value must be provided")
        return cls(name=name, is_constant=True, constant_val=value)

    @classmethod
    def gridded_var(cls, name=None, grid=None, data_source=None, time=None):
        if grid is None or data_source is None:
            raise ValueError('Must provide a grid and data source that can fit to the grid')
        if grid.infer_grid(data_source) is None:
            raise ValueError('Data source must be able to fit to the grid')
        return cls(name=name, is_constant=False, grid=grid, data_source=data_source, time=time)

    # Should vector quantities even be supported?
    @classmethod
    def constant_vec(cls, name=None, vector=None, magnitude=None, direction=None):
        if name is None:
            raise ValueError('Name must be provided')
        if vector is None:
            if magnitude is None or direction is None:
                raise ValueError('If vector is not provided, magnitude and direction must be')
            else:
                return cls(name=name, is_constant=True, constant_val=magnitude, constant_direction=direction)
        else:
            vector = np.asarray(vector, dtype=np.double)
            if magnitude is not None or direction is not None:
                warnings.warn("vector is defined, ignoring magnitude and direction")
            if vector.shape != (2,) or vector.shape != (3,):
                raise ValueError("Must provide [u,v] or [u,v,w] for vector")
            return cls(name=name, is_constant=True, constant_vec=vector)

    def at(self, points, time):
        if self.is_constant:
            return np.full((points.shape[0], 1), self.temperature_var.constant_val)
        if self.is_timeseries:
            # TBD
            return None

        t_alphas = t_index = s0 = s1 = None
        if time is not None:
            t_alphas = self.time.interp_alpha(time)
            t_index = self.time.indexof(time)
            s0 = [t_index]
            s1 = [t_index + 1]
            if len(variable.shape) == 4:
                s1.append(depth)
                s2.append(depth)
            v0 = self.grid.interpolate_var_to_poitns(points, self.data, slices=s1, memo=True)
            v1 = self.grid.interpolate_var_to_points(points, self.data, slices=s2, memo=True)
        else:
            s0 = None
            v0 = self.grid.interpolate_var_to_points(points, self.data, slices=s0, memo=True)

        vt = v0 + (v1 - v0) * t_alphas
        return vt


class WaterConditions(object):

    validprops = ['temperature',
                  'salinity',
                  'velocity',
                  'angles']

    def __init__(self, **kwargs):
        self.temperature_var = kwargs.get('temperature', None)
        self.salinity_var = kwargs.get('salinity', None)
        self.velocity_u, self.velocity_v = kwargs.get('velocity', (None, None))
        self.angles = kwargs.get('angles', None)

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
        if velocity_u is None:
            warnings.warn("Velocity u-componenet is not defined")
            return (None, None)
        if velocity_v is None:
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
