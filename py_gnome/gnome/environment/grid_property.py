import warnings
import copy

import netCDF4 as nc4
import numpy as np

from datetime import datetime, timedelta
from colander import SchemaNode, Float, Boolean, Sequence, MappingSchema, drop, String, OneOf, SequenceSchema, TupleSchema, DateTime
from gnome.environment.property import *

import pyugrid
import pysgrid
import unit_conversion
import collections



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
        if any([units == None, time == None, grid == None, data == None, grid_file == None, data_file == None]):
            raise ValueError("All attributes except name and extrapolate MUST be defined if variables is not a list of GriddedProp objects")
        if not hasattr(data, 'shape') or grid.infer_grid(data) is None:
            raise ValueError('Data must be able to fit to the grid')
        super(GriddedProp, self).__init__(name=name, units=units, data=data, extrapolate=extrapolate)
        self._grid = grid
        self.time = time
        self.data_file = data_file
        self.grid_file = grid_file

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        if len(t) != self.data.shape[0]:
            raise ValueError("Data/time interval mismatch")
        if isinstance(t,collections.Iterable) or isinstance(t, nc4.Variable):
            t = Time(t)
        else:
            raise ValueError("Time must be set with an iterable container or netCDF variable")
        self._time = t

    @property
    def units(self):
        return self._units

    @property
    def data(self):
        return self._data

    @property
    def grid(self):
        return self._grid

    def set_attr(self,
                 name=None,
#                  units=None,
                 time=None,
                 data=None,
                 data_file=None,
                 grid=None,
                 grid_file=None,
                 extrapolate=None):
        self.name = name if name is not None else self.name
#         self.units = units if units is not None else self.units
        self.extrapolate = extrapolate if extrapolate is not None else self.extrapolate
        if time is not None:
            if data and len(time) != data.shape[0]:
                raise ValueError("Time provided is incompatible with data source time dimension")
            self.time = time
        if data is not None:
            if (grid and grid.infer_grid(data) is None) or self.grid.infer_grid(data) is None:
                raise ValueError("Data shape is incompatible with grid shape.")
            self._data = data
        if grid is not None:
            if data is None:
                if grid.infer_grid(self.data) is None:
                    raise ValueError("Grid shape is incompatible with current data shape.")
            self._grid = grid
        self.grid_file = grid_file if grid_file is not None else self.grid_file
        self.data_file = data_file if data_file is not None else self.data_file

    def center_values(self, time, units=None):
        if not self.extrapolate:
            self.time.valid_time(time)
        if len(self.time) == 1:
            if self.grid.infer_grid(self.data) == 'center':
                if len(self.data.shape) == 2:
                    #curv grid
                    value = self.data[0:1:-2,1:-2]
                else:
                    value = self.data
            if units is not None and units != self.units:
                value = unit_conversion.convert(None, self.units, units, value)
        else:
            t_index = self.time.index_of(time)
            centers = self.grid.get_center_points()
            value = self.at(centers, time, units)
        return value

    def at(self, points, time, units=None):
        '''
        Interpolates this property to the given points at the given time.
        :param points: A Nx2 array of lon,lat points
        :param time: A datetime object. May be None; if this is so, the variable is assumed to be gridded
        but time-invariant
        '''
        t_alphas = t_index = s0 = s1 = value = None
        if not self.extrapolate:
            self.time.valid_time(time)
        t_index = self.time.index_of(time)
        if len(self.time) == 1:
            value = self.grid.interpolate_var_to_points(points, self.data, slices=s0, memo=True)
        else:
            if time > self.time.max_time:
                value = self.data[-1]
            if time <= self.time.min_time:
                value = self.data[0]
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

        if units is not None and units != self.units:
            value = unit_conversion.convert(None, self.units, units, value)
        return value


class GridVectorProp(VectorProp):

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 extrapolate=False,
                 grid = None,
                 grid_file=None,
                 data_file=None,
                 varnames=None):

        self.name = name

        if variables is None or len(variables) < 2:
            raise TypeError("Variables needs to be a list of at least 2 GriddedProp objects or ndarray-like arrays")

        gp = True
        if all([isinstance(v, GriddedProp) for v in variables]):
            if time is not None and not isinstance(time, Time):
                time = Time(time)
            units = variables[0].units if units is None else units
            time = variables[0].time if time is None else time
            grid = variables[0].grid if grid is None else extrapolate
            grid_file = variables[0].grid_file if grid_file is None else grid_file
            data_file = variables[0].data_file if data_file is None else data_file
            for v in variables:
                if (v.units != units or
                    v.time != time or
                    v.extrapolate != extrapolate or
                    v.grid != grid or
                    v.grid_file != grid_file or
                    v.data_file != data_file):
                    raise ValueError("Variable {0} did not have parameters consistent with what was specified".format(v.name))
        else:
            if any([isinstance(v, GriddedProp) for v in variables]):
                raise TypeError("Cannot mix GriddedProp objects with other data sources.")
            if any([units == None, time == None, grid == None, grid_file == None, data_file == None]):
                raise ValueError("All attributes except name, varnames and extrapolate MUST be defined if variables is not a list of GriddedProp objects")
            for i, var in enumerate(variables):
                name = 'var{0}'.format(i) if varnames is None else varnames[i]
                variables[i] = GriddedProp(name = 'var{0}'.format(i),
                                           units = units,
                                           time = time,
                                           data = variables[i],
                                           grid = grid,
                                           grid_file = grid_file,
                                           data_file = data_file,
                                           extrapolate = extrapolate)

        self._variables = variables
        self.data_file = data_file
        self.grid_file = grid_file
        self.time = time
        self._grid = grid
        self._units = units
        self._extrapolate = extrapolate
        self._check_consistency(self.variables)

    def _check_consistency(self, varlist):
        '''
        Checks that the attributes of each GriddedProp in varlist are the same as the GridVectorProp
        '''
        for v in varlist:
            if (v.units != self.units or
                v.time != self.time or
                v.extrapolate != self.extrapolate or
                v.grid != self.grid or
                v.grid_file != self.grid_file or
                v.data_file != self.data_file):
                raise ValueError("Variable {0} did not have parameters consistent with what was specified".format(v.name))

    @property
    def grid(self):
        return self._grid

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        if not isinstance(t, Time):
            if isinstance(t,collections.Iterable) or isinstance(t, nc4.Variable):
                t = Time(t)
            else:
                raise ValueError("Time must be set with an iterable container of datetime.datatime, gnome.environment.property.Time, or netCDF4.Variable")
        self._time = t

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, variables):
        if all([isinstance(v, GriddedProp) for v in variables]):
            self._check_consistency(variables)
            self._variables = variables
            return
        if any([isinstance(v, GriddedProp) for v in variables]):
            raise ValueError("Cannot mix GriddedProp objects with other data sources.")

    def set_attr(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 extrapolate=None,
                 grid = None,
                 grid_file=None,
                 data_file=None,):

        self.name = name if name is not None else self.name
        if variables is not None:
            if self.variables is not None and len(variables) != len(self.variables):
                raise ValueError('Cannot change the number of variables using set_attr. {0} provided, {1} required'.format(len(variables), len(self.variables)))

            for i, v in enumerate(variables):
                if isinstance(v, GriddedProp):
                    variables[i] = variables.data

        units = self.units if units is None else units
        time = self.time if time is None else time
        extrapolate = self._extrapolate if extrapolate is None else extrapolate
        grid = self.grid if grid is None else grid
        grid_file = self.grid_file if grid_file is None else grid_file
        data_file = self.data_file if data_file is None else data_file

        for i, var in enumerate(self.variables):
            if variables is None:
                nv = None
            else:
                nv = variables[i]
            var.set_attr(units=units,
                         time = time,
                         data = nv,
                         extrapolate = extrapolate,
                         grid = grid,
                         grid_file = grid_file,
                         data_file = data_file,)
        else:
            for i, var in enumerate(self.variables):
                var.set_attr(units=units,
                             time=time,
                             extrapolate=extrapolate,
                             grid = grid,
                             grid_file = grid_file,
                             data_file = data_file,)
        self._units = units
        self._time = time
        self._extrapolate = extrapolate
        self._grid = grid
        self.grid_file = grid_file
        self.grid_file = grid_file
