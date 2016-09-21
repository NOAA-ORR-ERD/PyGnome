import netCDF4 as nc4
import numpy as np

from datetime import datetime, timedelta
from collections import OrderedDict
from colander import SchemaNode, Float, Boolean, Sequence, MappingSchema, drop, String, OneOf, SequenceSchema, TupleSchema, DateTime
from gnome.utilities.file_tools.data_helpers import _init_grid, _gen_topology, _get_dataset
from gnome.environment.property import *

import pyugrid
import pysgrid
import unit_conversion
import collections
import hashlib


class GridPropSchema(PropertySchema):
    varname = SchemaNode(String(), missing=drop)
    data_file = SchemaNode(String(), missing=drop)
    grid_file = SchemaNode(String(), missing=drop)


class GriddedProp(EnvProp):

    default_names = []

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 data=None,
                 grid=None,
                 data_file=None,
                 grid_file=None,
                 dataset=None,
                 varname=None):
        '''
        This class represents a phenomenon using gridded data

        :param name: Name
        :param units: Units
        :param time: Time axis of the data
        :param data: Underlying data source
        :param grid: Grid that the data corresponds with
        :param data_file: Name of data source file
        :param grid_file: Name of grid source file
        :param varname: Name of the variable in the data source file
        :type name: string
        :type units: string
        :type time: [] of datetime.datetime, netCDF4 Variable, or Time object
        :type data: netCDF4.Variable or numpy.array
        :type grid: pysgrid or pyugrid
        :type data_file: string
        :type grid_file: string
        :type varname: string
        '''

        self._grid = self._data_file = self._grid_file = None

        if any([grid is None, data is None]):
            raise ValueError("Grid and Data must be defined")
        if not hasattr(data, 'shape'):
            if grid.infer_location is None:
                raise ValueError('Data must be able to fit to the grid')
        self._grid = grid
        super(GriddedProp, self).__init__(name=name, units=units, time=time, data=data)
        self.data_file = data_file
        self.grid_file = grid_file
        self._result_memo = OrderedDict()

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    varname=None,
                    grid_topology=None,
                    name=None,
                    units=None,
                    time=None,
                    grid=None,
                    dataset=None,
                    data_file=None,
                    grid_file=None,
                    load_all=False,
                    **kwargs
                    ):
        '''
        Allows one-function creation of a GriddedProp from a file.

        :param filename: Default data source. Parameters below take precedence
        :param varname: Name of the variable in the data source file
        :param grid_topology: Description of the relationship between grid attributes and variable names.
        :param name: Name of property
        :param units: Units
        :param time: Time axis of the data
        :param data: Underlying data source
        :param grid: Grid that the data corresponds with
        :param dataset: Instance of open Dataset
        :param data_file: Name of data source file
        :param grid_file: Name of grid source file
        :type filename: string
        :type varname: string
        :type grid_topology: {string : string, ...}
        :type name: string
        :type units: string
        :type time: [] of datetime.datetime, netCDF4 Variable, or Time object
        :type data: netCDF4.Variable or numpy.array
        :type grid: pysgrid or pyugrid
        :type dataset: netCDF4.Dataset
        :type data_file: string
        :type grid_file: string
        '''
        if filename is not None:
            data_file = filename
            grid_file = filename

        ds = None
        dg = None
        if dataset is None:
            if grid_file == data_file:
                ds = dg = _get_dataset(grid_file)
            else:
                ds = _get_dataset(data_file)
                dg = _get_dataset(grid_file)
        else:
            ds = dg = dataset

        if grid is None:
            grid = _init_grid(grid_file,
                              grid_topology=grid_topology,
                              dataset=dg)
        if varname is None:
            varname = cls._gen_varname(data_file,
                                       dataset=ds)
            if varname is None:
                raise NameError('Default current names are not in the data file, must supply variable name')
        data = ds[varname]
        name = varname if name is None else name
        if units is None:
            try:
                units = data.units
            except AttributeError:
                units = None
        timevar = None
        if time is None:
            try:
                timevar = data.time if data.time == data.dimensions[0] else data.dimensions[0]
            except AttributeError:
                timevar = data.dimensions[0]
            time = Time(ds[timevar])
        if load_all:
            data = data[:]
        return cls(name=name,
                   units=units,
                   time=time,
                   data=data,
                   grid=grid,
                   grid_file=grid_file,
                   data_file=data_file)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        if t is None:
            self._time = None
            return
        if self.data is not None and len(t) != self.data.shape[0] and len(t) > 1:
            raise ValueError("Data/time interval mismatch")
        if isinstance(t, Time):
            self._time = t
        elif isinstance(t, collections.Iterable) or isinstance(t, nc4.Variable):
            self._time = Time(t)
        else:
            raise ValueError("Time must be set with an iterable container or netCDF variable")

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        if self.time is not None and len(d) != len(self.time):
            raise ValueError("Data/time interval mismatch")
        if self.grid is not None and self.grid.infer_location(d) is None:
            raise ValueError("Data/grid shape mismatch. Data shape is {0}, Grid shape is {1}".format(d.shape, self.grid.shape))
        self._data = d

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, g):
        if not (isinstance(g, (pyugrid.UGrid, pysgrid.SGrid))):
            raise ValueError('Grid must be set with a pyugrid.UGrid or pysgrid.SGrid object')
        if self.data is not None and g.infer_location(self.data) is None:
            raise ValueError("Data/grid shape mismatch. Data shape is {0}, Grid shape is {1}".format(self.data.shape, self.grid.shape))
        self._grid = g

    @property
    def grid_shape(self):
        if hasattr(self.grid, 'shape'):
            return self.grid.shape
        else:
            return self.grid.node_lon.shape

    @property
    def data_shape(self):
        return self.data.shape

    @property
    def is_data_on_nodes(self):
        return self.grid.infer_location(self._data) == 'node'

    def _get_hash(self, points, time):
        """
        Returns a SHA1 hash of the array of points passed in
        """
        return (hashlib.sha1(points.tobytes()).hexdigest(), hashlib.sha1(str(time)).hexdigest())

    def _memoize_result(self, points, time, result, D, _copy=False, _hash=None):
        if _copy:
            result = result.copy()
        result.setflags(write=False)
        if _hash is None:
            _hash = self._get_hash(points, time)
        if D is not None and len(D) > 4:
            D.popitem(last=False)
        D[_hash] = result
        D[_hash].setflags(write=False)

    def _get_memoed(self, points, time, D, _copy=False, _hash=None):
        if _hash is None:
            _hash = self._get_hash(points, time)
        if (D is not None and _hash in D):
            return D[_hash].copy() if _copy else D[_hash]
        else:
            return None

    def set_attr(self,
                 name=None,
                 time=None,
                 data=None,
                 data_file=None,
                 grid=None,
                 grid_file=None):
        self.name = name if name is not None else self.name
#         self.units = units if units is not None else self.units
        if time is not None:
            if data and len(time) != data.shape[0]:
                raise ValueError("Time provided is incompatible with data source time dimension")
            self.time = time
        if data is not None:
            if (grid and grid.infer_location(data) is None) or self.grid.infer_location(data) is None:
                raise ValueError("Data shape is incompatible with grid shape.")
            self._data = data
        if grid is not None:
            if data is None:
                if grid.infer_location(self.data) is None:
                    raise ValueError("Grid shape is incompatible with current data shape.")
            self._grid = grid
        self.grid_file = grid_file if grid_file is not None else self.grid_file
        self.data_file = data_file if data_file is not None else self.data_file

    def center_values(self, time, units=None, extrapolate=False):
        # NOT COMPLETE
        if not extrapolate:
            self.time.valid_time(time)
        if len(self.time) == 1:
            if len(self.data.shape) == 2:
                if isinstance(self.grid, pysgrid.sgrid):
                    # curv grid
                    value = self.data[0:1:-2, 1:-2]
                else:
                    value = self.data
            if units is not None and units != self.units:
                value = unit_conversion.convert(self.units, units, value)
        else:
            centers = self.grid.get_center_points()
            value = self.at(centers, time, units)
        return value

    def at(self, points, time, units=None, depth=-1, extrapolate=False, memoize=True, _hash=None, mask=False, **kwargs):
        '''
        Find the value of the property at positions P at time T

        :param points: Coordinates to be queried (P)
        :param time: The time at which to query these points (T)
        :param depth: Specifies the depth level of the variable
        :param units: units the values will be returned in (or converted to)
        :param extrapolate: if True, extrapolation will be supported
        :type points: Nx2 array of double
        :type time: datetime.datetime object
        :type depth: integer
        :type units: string such as ('mem/s', 'knots', etc)
        :type extrapolate: boolean (True or False)
        :return: returns a Nx1 array of interpolated values
        :rtype: double
        '''

        sg = False
        mem = memoize

        if _hash is None:
            _hash = self._get_hash(points, time)

        if mem:
            res = self._get_memoed(points, time, self._result_memo, _hash=_hash)
            if res is not None:
                return np.ma.filled(res) if not _copy else np.ma.filled(res).copy()

        if self.time is None:
            # special case! prop has no time variance
            v0 = self.grid.interpolate_var_to_points(points, self.data, slices=None, slice_grid=sg, _memo=mem, _hash=_hash,)
            return np.ma.filled(v0) if not _copy else np.ma.filled(v0).copy()

        t_alphas = s0 = s1 = value = None
        if not extrapolate:
            self.time.valid_time(time)
        t_index = self.time.index_of(time, extrapolate)
        if len(self.time) == 1:
            value = self.grid.interpolate_var_to_points(points, self.data, slices=[0], _memo=mem, _hash=_hash,)
        else:
            if time > self.time.max_time:
                value = self.data[-1]
            if time <= self.time.min_time:
                value = self.data[0]
            if extrapolate and t_index == len(self.time.time):
                s0 = [t_index - 1]
                value = self.grid.interpolate_var_to_points(points, self.data, slices=s0, _memo=mem, _hash=_hash,)
            else:
                t_alphas = self.time.interp_alpha(time, extrapolate)
                s1 = [t_index]
                s0 = [t_index - 1]
                if len(self.data.shape) == 4:
                    s0.append(depth)
                    s1.append(depth)
                v0 = self.grid.interpolate_var_to_points(points, self.data, slices=s0, slice_grid=sg, _memo=mem, _hash=_hash[0],)
                v1 = self.grid.interpolate_var_to_points(points, self.data, slices=s1, slice_grid=sg, _memo=mem, _hash=_hash[0],)
                value = v0 + (v1 - v0) * t_alphas

        if units is not None and units != self.units:
            value = unit_conversion.convert(self.units, units, value)

        if mem:
            self._memoize_result(points, time, value, self._result_memo, _hash=_hash)
            
        return np.ma.filled(value) if not _copy else np.ma.filled(value).copy()

    @classmethod
    def _gen_varname(cls,
                     filename=None,
                     dataset=None):
        """
        Function to find the default variable names if they are not provided.

        :param filename: Name of file that will be searched for variables
        :param dataset: Existing instance of a netCDF4.Dataset
        :type filename: string
        :type dataset: netCDF.Dataset
        :return: List of default variable names, or None if none are found
        """
        df = None
        if dataset is not None:
            df = dataset
        else:
            df = _get_dataset(filename)
        for n in cls.default_names:
            if n in df.variables.keys():
                return n
        raise ValueError("Default names not found.")


class GridVectorProp(VectorProp):

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 grid=None,
                 grid_file=None,
                 data_file=None,
                 dataset=None,
                 varnames=None):

        self._grid = self._grid_file = self._data_file = None

        if any([units is None, time is None, grid is None]) and not all([isinstance(v, GriddedProp) for v in variables]):
            raise ValueError("All attributes except name, varnames and MUST be defined if variables is not a list of TimeSeriesProp objects")

        if variables is None or len(variables) < 2:
            raise TypeError("Variables needs to be a list of at least 2 GriddedProp objects or ndarray-like arrays")

        if all([isinstance(v, GriddedProp) for v in variables]):
            grid = variables[0].grid if grid is None else grid
            grid_file = variables[0].grid_file if grid_file is None else grid_file
            data_file = variables[0].data_file if data_file is None else data_file
        VectorProp.__init__(self,
                            name,
                            units,
                            time,
                            variables,
                            grid=grid,
                            dataset=dataset,
                            data_file=data_file,
                            grid_file=grid_file)

        self._check_consistency()
        self._result_memo = OrderedDict()

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    varnames=None,
                    grid_topology=None,
                    name=None,
                    units=None,
                    time=None,
                    grid=None,
                    data_file=None,
                    grid_file=None,
                    dataset=None,
                    load_all=False,
                    **kwargs
                    ):
        '''
        Allows one-function creation of a GridVectorProp from a file.

        :param filename: Default data source. Parameters below take precedence
        :param varnames: Names of the variables in the data source file
        :param grid_topology: Description of the relationship between grid attributes and variable names.
        :param name: Name of property
        :param units: Units
        :param time: Time axis of the data
        :param data: Underlying data source
        :param grid: Grid that the data corresponds with
        :param dataset: Instance of open Dataset
        :param data_file: Name of data source file
        :param grid_file: Name of grid source file
        :type filename: string
        :type varnames: [] of string
        :type grid_topology: {string : string, ...}
        :type name: string
        :type units: string
        :type time: [] of datetime.datetime, netCDF4 Variable, or Time object
        :type data: netCDF4.Variable or numpy.array
        :type grid: pysgrid or pyugrid
        :type dataset: netCDF4.Dataset
        :type data_file: string
        :type grid_file: string
        '''
        if filename is not None:
            data_file = filename
            grid_file = filename

        ds = None
        dg = None
        if dataset is None:
            if grid_file == data_file:
                ds = dg = _get_dataset(grid_file)
            else:
                ds = _get_dataset(data_file)
                dg = _get_dataset(grid_file)
        else:
            ds = dg = dataset

        if grid is None:
            grid = _init_grid(grid_file,
                              grid_topology=grid_topology,
                              dataset=dg)
        if varnames is None:
            varnames = cls._gen_varnames(data_file,
                                         dataset=ds)
        if name is None:
            name = 'GridVectorProp'
        timevar = None
        data = ds[varnames[0]]
        if time is None:
            try:
                timevar = data.time if data.time == data.dimensions[0] else data.dimensions[0]
            except AttributeError:
                timevar = data.dimensions[0]
            time = Time(ds[timevar])
        variables = []
        for vn in varnames:
            variables.append(GriddedProp.from_netCDF(filename=filename,
                                                     varname=vn,
                                                     grid_topology=grid_topology,
                                                     units=units,
                                                     time=time,
                                                     grid=grid,
                                                     data_file=data_file,
                                                     grid_file=grid_file,
                                                     dataset=ds,
                                                     load_all=load_all))
        return cls(name,
                   units,
                   time,
                   variables,
                   grid=grid,
                   grid_file=grid_file,
                   data_file=data_file,
                   dataset=ds)

    def _check_consistency(self):
        '''
        Checks that the attributes of each GriddedProp in varlist are the same as the GridVectorProp
        '''
        if self.units is None or self.time is None or self.grid is None:
            return
        for v in self.variables:
            if v.units != self.units:
                raise ValueError("Variable {0} did not have units consistent with what was specified. Got: {1} Expected {2}".format(v.name, v.units, self.units))
            if v.time != self.time:
                raise ValueError("Variable {0} did not have time consistent with what was specified Got: {1} Expected {2}".format(v.name, v.time, self.time))
            if v.grid != self.grid:
                raise ValueError("Variable {0} did not have grid consistent with what was specified Got: {1} Expected {2}".format(v.name, v.grid, self.grid))
            if v.grid_file != self.grid_file:
                raise ValueError("Variable {0} did not have grid_file consistent with what was specified Got: {1} Expected {2}".format(v.name, v.grid_file, self.grid_file))
            if v.data_file != self.data_file:
                raise ValueError("Variable {0} did not have data_file consistent with what was specified Got: {1} Expected {2}".format(v.name, v.data_file, self.data_file))

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, g):
        if not (isinstance(g, (pyugrid.UGrid, pysgrid.SGrid))):
            raise ValueError('Grid must be set with a pyugrid.UGrid or pysgrid.SGrid object')
        if self._variables is not None:
            if g.infer_location(self.variables[0]) is None:
                raise ValueError("Grid with shape {0} not compatible with data of shape {1}".format(g.shape, self.data_shape))
            for v in self.variables:
                v.grid = g
        else:
            self._grid = g

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, vs):
        if vs is None:
            self._variables = None
            return
        new_vars = []
        for i, var in enumerate(vs):
            if not isinstance(var, GriddedProp):
                if (isinstance(var, (collections.Iterable, nc4.Variable)) and
                        len(var) == len(self.time) and
                        self.grid.infer_location(var) is not None):
                    new_vars.append(GriddedProp(name='var{0}'.format(i),
                                    units=self.units,
                                    time=self.time,
                                    grid=self.grid,
                                    data=vs[i],
                                    grid_file=self.grid_file,
                                    data_file=self.data_file))
                else:
                    raise ValueError('Variables must contain an iterable, netCDF4.Variable or GriddedProp objects')
            else:
                new_vars.append(var)
        self._variables = new_vars
        self._check_consistency()

    @property
    def is_data_on_nodes(self):
        return self.grid.infer_location(self.variables[0].data) == 'node'

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        if self.variables is not None:
            for v in self.variables:
                try:
                    v.time = t
                except ValueError as e:
                    raise ValueError('''Time was not compatible with variables.
                    Set variables attribute to None to allow changing other attributes
                    Original error: {0}'''.format(str(e)))
        if isinstance(t, Time):
            self._time = t
        elif isinstance(t, collections.Iterable) or isinstance(t, nc4.Variable):
            self._time = Time(t)
        else:
            raise ValueError("Time must be set with an iterable container or netCDF variable")

    @property
    def data_shape(self):
        if self._variables is not None:
            return self.variables.data.shape
        else:
            return None

    def _get_hash(self, points, time):
        """
        Returns a SHA1 hash of the array of points passed in
        """
        return (hashlib.sha1(points.tobytes()).hexdigest(), hashlib.sha1(str(time)).hexdigest())

    def _memoize_result(self, points, time, result, D, _copy=True, _hash=None):
        if _copy:
            result = result.copy()
        result.setflags(write=False)
        if _hash is None:
            _hash = self._get_hash(points, time)
        if D is not None and len(D) > 8:
            D.popitem(last=False)
        D[_hash] = result

    def _get_memoed(self, points, time, D, _copy=True, _hash=None):
        if _hash is None:
            _hash = self._get_hash(points, time)
        if (D is not None and _hash in D):
            return D[_hash].copy() if _copy else D[_hash]
        else:
            return None

    def set_attr(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 grid=None,
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
        grid = self.grid if grid is None else grid
        grid_file = self.grid_file if grid_file is None else grid_file
        data_file = self.data_file if data_file is None else data_file

        for i, var in enumerate(self.variables):
            if variables is None:
                nv = None
            else:
                nv = variables[i]
            var.set_attr(units=units,
                         time=time,
                         data=nv,
                         grid=grid,
                         grid_file=grid_file,
                         data_file=data_file,)
        else:
            for i, var in enumerate(self.variables):
                var.set_attr(units=units,
                             time=time,
                             grid=grid,
                             grid_file=grid_file,
                             data_file=data_file,)
        self._units = units
        self._time = time
        self._grid = grid
        self.grid_file = grid_file
        self.grid_file = grid_file

    def at(self, points, time, units=None, depth=-1, extrapolate=False, memoize=True, _hash=None, **kwargs):
        mem = memoize
        if hash is None:
            _hash = self._get_hash(points, time)

        if mem:
            res = self._get_memoed(points, time, self._result_memo, _hash=_hash)
            if res is not None:
                return res

        value = super(GridVectorProp, self).at(points=points,
                                               time=time,
                                               units=units,
                                               depth=depth,
                                               extrapolate=extrapolate,
                                               memoize=memoize,
                                               _hash=_hash,
                                               **kwargs)

        if mem:
            self._memoize_result(points, time, value, self._result_memo, _hash=_hash)
        return value

    @classmethod
    def _gen_varnames(cls,
                      filename=None,
                      dataset=None):
        """
        Function to find the default variable names if they are not provided.

        :param filename: Name of file that will be searched for variables
        :param dataset: Existing instance of a netCDF4.Dataset
        :type filename: string
        :type dataset: netCDF.Dataset
        :return: List of default variable names, or None if none are found
        """
        df = None
        if dataset is not None:
            df = dataset
        else:
            df = _get_dataset(filename)
        for n in cls.default_names:
            if n[0] in df.variables.keys() and n[1] in df.variables.keys():
                return n
        raise ValueError("Default names not found.")
