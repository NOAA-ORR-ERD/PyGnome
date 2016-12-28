import netCDF4 as nc4
import numpy as np

from collections import OrderedDict
from colander import SchemaNode, SchemaType, Float, Boolean, Sequence, MappingSchema, drop, String, OneOf, SequenceSchema, TupleSchema, DateTime, List
from gnome.utilities.file_tools.data_helpers import _get_dataset
from gnome.environment.property import *
from gnome.environment.grid import PyGrid, PyGrid_U, PyGrid_S, PyGridSchema

import hashlib
from gnome.utilities.orderedcollection import OrderedCollection
from gnome.environment.ts_property import TimeSeriesProp
from functools import wraps
import pytest

class GridPropSchema(PropertySchema):
    varname = SchemaNode(String())
    grid = PyGridSchema(missing=drop)
    data_file = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String())])
    grid_file = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String())])


class GriddedProp(EnvProp):

    _state = copy.deepcopy(EnvProp._state)

    _schema = GridPropSchema

    _state.add_field([serializable.Field('grid', save=True, update=True, save_reference=True),
                      serializable.Field('varname', save=True, update=True),
                      serializable.Field('data_file', save=True, update=True, isdatafile=True),
                      serializable.Field('grid_file', save=True, update=True, isdatafile=True)])

    default_names = []
    _def_count = 0

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 data=None,
                 grid=None,
                 depth=None,
                 data_file=None,
                 grid_file=None,
                 dataset=None,
                 varname=None,
                 fill_value=0,
                 **kwargs):
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

        if any([grid is None, data is None]):
            raise ValueError("Grid and Data must be defined")
        if not hasattr(data, 'shape'):
            if grid.infer_location is None:
                raise ValueError('Data must be able to fit to the grid')
        self.grid = grid
        self.depth = depth
        super(GriddedProp, self).__init__(name=name, units=units, time=time, data=data)
        self.data_file = data_file
        self.grid_file = grid_file
        self.varname = varname
        self._result_memo = OrderedDict()
        self.fill_value = fill_value

#     def __repr__(self):
#         return str(self.serialize())

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    varname=None,
                    grid_topology=None,
                    name=None,
                    units=None,
                    time=None,
                    grid=None,
                    depth=None,
                    dataset=None,
                    data_file=None,
                    grid_file=None,
                    load_all=False,
                    fill_value=0,
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
        :param depth: Depth axis object
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
        :type depth: Depth, S_Depth or L_Depth
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
            if grid_file is not None:
                dg = _get_dataset(grid_file)
            else:
                dg = dataset
            ds = dataset

        if grid is None:
            grid = PyGrid.from_netCDF(grid_file,
                                      dataset=dg,
                                      grid_topology=grid_topology)
        if varname is None:
            varname = cls._gen_varname(data_file,
                                       dataset=ds)
            if varname is None:
                raise NameError('Default current names are not in the data file, must supply variable name')
        data = ds[varname]
        if name is None:
            name = cls.__name__ + str(cls._def_count)
            cls._def_count += 1
        if units is None:
            try:
                units = data.units
            except AttributeError:
                units = None
        if time is None:
            time = Time.from_netCDF(filename=data_file,
                                    dataset=ds,
                                    datavar=data)
        if depth is None:
            if (isinstance(grid, PyGrid_S) and len(data.shape) == 4 or
                    isinstance(grid, PyGrid_U) and len(data.shape) == 3):
                from gnome.environment.environment_objects import Depth
                depth = Depth(surface_index=-1)
#             if len(data.shape) == 4 or (len(data.shape) == 3 and time is None):
#                 from gnome.environment.environment_objects import S_Depth
#                 depth = S_Depth.from_netCDF(grid=grid,
#                                             depth=1,
#                                             data_file=data_file,
#                                             grid_file=grid_file,
#                                             **kwargs)
        if load_all:
            data = data[:]
        return cls(name=name,
                   units=units,
                   time=time,
                   data=data,
                   grid=grid,
                   depth=depth,
                   grid_file=grid_file,
                   data_file=data_file,
                   fill_value=fill_value,
                   varname=varname,
                   **kwargs)

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
            raise ValueError("Data/grid shape mismatch. Data shape is {0}, Grid shape is {1}".format(d.shape, self.grid.node_lon.shape))
        self._data = d

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

    @property
    def dimension_ordering(self):
        '''
        Returns a list that describes the dimensions of the property's data. If a dimension_ordering is assigned,
        it will continue to use that. If no dimension_ordering is set, then a default ordering will be generated
        based on the object properties and data shape.

        For example, if the data has 4 dimensions and is represented by a PyGrid_S (structured grid), and the
        GriddedProp has a depth and time assigned, then the assumed ordering is ['time','depth','lon','lat']

        If the data has 3 dimensions, self.grid is a PyGrid_S, and self.time is None, then the ordering is
        ['depth','lon','lat']
        If the data has 3 dimensions, self.grid is a PyGrid_U, the ordering is ['time','depth','ele']
        '''
        if not hasattr(self, '_order'):
            self._order = None
        if self._order is not None:
            return self._order
        else:
            if isinstance(self.grid, PyGrid_S):
                order = ['time', 'depth', 'lon', 'lat']
            else:
                order = ['time', 'depth', 'ele']
            ndim = len(self.data.shape)
            diff = len(order) - ndim
            if diff == 0:
                return order
            elif diff == 1:
                if self.time is not None:
                    del order[1]
                elif self.depth is not None:
                    del order[0]
                else:
                    raise ValueError('Generated ordering too short to fit data. Time or depth must not be None')
            elif diff == 2:
                order = order[2:]
            else:
                raise ValueError('Too many/too few dimensions ndim={0}'.format(ndim))
            return order

    @dimension_ordering.setter
    def dimension_ordering(self, order):
        self._order = order

#     @profile
    def at(self, points, time, units=None, extrapolate=False, _hash=None, _mem=True, **kwargs):
        '''
        Find the value of the property at positions P at time T

        :param points: Coordinates to be queried (P)
        :param time: The time at which to query these points (T)
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
        if _hash is None:
            _hash = self._get_hash(points, time)

        if _mem:
            res = self._get_memoed(points, time, self._result_memo, _hash=_hash)
            if res is not None:
                return np.ma.filled(res)

        order = self.dimension_ordering
        if order[0] == 'time':
            value = self._time_interp(points, time, extrapolate, _mem=_mem, _hash=_hash, **kwargs)
        elif order[0] == 'depth':
            value = self._depth_interp(points, time, extrapolate, _mem=_mem, _hash=_hash, **kwargs)
        else:
            value = self._xy_interp(points, time, extrapolate, _mem=_mem, _hash=_hash, **kwargs)

        if _mem:
            self._memoize_result(points, time, value, self._result_memo, _hash=_hash)
        return np.ma.filled(value)

    def _xy_interp(self, points, time, extrapolate, slices=(), **kwargs):
        '''
        Uses the py(s/u)grid interpolation to determine the values at the points, and returns it
        :param points: Coordinates to be queried (3D)
        :param time: Time of the query
        :param extrapolate: Turns extrapolation on or off
        :param slices: describes how the data needs to be sliced to reach the appropriate dimension
        :type points: Nx3 array of double
        :type time: datetime.datetime object
        :type extrapolate: boolean
        :type slices: tuple of integers or slice objects
        '''
        _hash = kwargs['_hash'] if '_hash' in kwargs else None
        units = kwargs['units'] if 'units' in kwargs else None
        value = self.grid.interpolate_var_to_points(points[:, 0:2], self.data, _hash=_hash[0], slices=slices, _memo=True)
        if units is not None and units != self.units:
            value = unit_conversion.convert(self.units, units, value)
        return value

    def _time_interp(self, points, time, extrapolate, slices=(), **kwargs):
        '''
        Uses the Time object to interpolate the result of the next level of interpolation, as specified
        by the dimension_ordering attribute.
        :param points: Coordinates to be queried (3D)
        :param time: Time of the query
        :param extrapolate: Turns extrapolation on or off
        :param slices: describes how the data needs to be sliced to reach the appropriate dimension
        :type points: Nx3 array of double
        :type time: datetime.datetime object
        :type extrapolate: boolean
        :type slices: tuple of integers or slice objects
        '''
        order = self.dimension_ordering
        idx = order.index('time')
        if order[idx + 1] != 'depth':
            val_func = self._xy_interp
        else:
            val_func = self._depth_interp

        if time == self.time.min_time or (extrapolate and time < self.time.min_time):
            # min or before
            return val_func(points, time, extrapolate, slices=(0,), ** kwargs)
        elif time == self.time.max_time or (extrapolate and time > self.time.max_time):
            return val_func(points, time, extrapolate, slices=(-1,), **kwargs)
        else:
            ind = self.time.index_of(time)
            s1 = slices + (ind,)
            s0 = slices + (ind - 1,)
            v0 = val_func(points, time, extrapolate, slices=s0, **kwargs)
            v1 = val_func(points, time, extrapolate, slices=s1, **kwargs)
            alphas = self.time.interp_alpha(time)
            value = v0 + (v1 - v0) * alphas
            return value

    def _depth_interp(self, points, time, extrapolate, slices=(), **kwargs):
        '''
        Uses the Depth object to interpolate the result of the next level of interpolation, as specified
        by the dimension_ordering attribute.
        :param points: Coordinates to be queried (3D)
        :param time: Time of the query
        :param extrapolate: Turns extrapolation on or off
        :param slices: describes how the data needs to be sliced to reach the appropriate dimension
        :type points: Nx3 array of double
        :type time: datetime.datetime object
        :type extrapolate: boolean
        :type slices: tuple of integers or slice objects
        '''
        order = self.dimension_ordering
        idx = order.index('depth')
        if order[idx + 1] != 'time':
            val_func = self._xy_interp
        else:
            val_func = self._time_interp
        indices, alphas = self.depth.interpolation_alphas(points, self.data.shape[1:], kwargs.get('_hash', None))
        if indices is None and alphas is None:
            # all particles are on surface
            return val_func(points, time, extrapolate, slices=slices + (self.depth.surface_index,), **kwargs)
        else:
            min_idx = indices[indices != -1].min() - 1
            max_idx = indices.max()
            values = np.zeros(len(points), dtype=np.float64)
            v0 = val_func(points, time, extrapolate, slices=slices + (min_idx - 1,), **kwargs)
            for idx in range(min_idx + 1, max_idx + 1):
                v1 = val_func(points, time, extrapolate, slices=slices + (idx,), **kwargs)
                pos_idxs = np.where(indices == idx)[0]
                sub_vals = v0 + (v1 - v0) * alphas
                if len(pos_idxs) > 0:
                    values.put(pos_idxs, sub_vals.take(pos_idxs))
                v0 = v1
            if extrapolate:
                underground = (indices == self.depth.bottom_index)
                values[underground] = val_func(points, time, extrapolate, slices=slices + (self.depth.bottom_index,), **kwargs)
            else:
                underground = (indices == self.depth.bottom_index)
                values[underground] = self.fill_value
            return values

#     def serialize(self, json_='webapi'):
#         _dict = serializable.Serializable.serialize(self, json_=json_)
#         if self.data_file is not None:
#             # put file in save zip
#             pass
#         else:
#             # write data to file and put in zip
#             pass
#         if self.grid_file is not None:
#             # put grid in save zip. make sure it's not in there twice.
#             pass
#         else:
#             # write grid to file and put in zip
#             pass

    @classmethod
    def new_from_dict(cls, dict_):
        if 'data' not in dict_:
            return cls.from_netCDF(**dict_)
        return super(GriddedProp, cls).new_from_dict(dict_)

    @classmethod
    def deserialize(cls, json_):
        return super(GriddedProp, cls).deserialize(json_)

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


class GridVectorPropSchema(VectorPropSchema):
    varnames = SequenceSchema(SchemaNode(String()))
    grid = PyGridSchema(missing=drop)
    data_file = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String())])
    grid_file = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String())])
     
    def __init__(self, json_='webapi', *args, **kwargs):
        if json_ == 'save':
            self.add(SchemaNode(typ=Sequence(), children=[SchemaNode(EnvProp())], name='variables'))
        super(GridVectorPropSchema, self).__init__(*args, **kwargs)

class GridVectorProp(VectorProp):
    _state = copy.deepcopy(VectorProp._state)

    _schema = GridVectorPropSchema

    _state.add_field([serializable.Field('grid', save=True, update=True, save_reference=True),
                      serializable.Field('variables', save=True, update=True, iscollection=True),
                      serializable.Field('varnames', save=True, update=True),
                      serializable.Field('data_file', save=True, update=True, isdatafile=True),
                      serializable.Field('grid_file', save=True, update=True, isdatafile=True)])

    default_names = []

    _def_count = 0

    def __init__(self,
                 grid=None,
                 depth=None,
                 grid_file=None,
                 data_file=None,
                 dataset=None,
                 varnames=None,
                 **kwargs):

        super(GridVectorProp, self).__init__(**kwargs)
        if isinstance(self.variables, list):
            self.variables = OrderedCollection(elems=self.variables, dtype=EnvProp)
        if isinstance(self.variables[0], GriddedProp):
            self.grid = self.variables[0].grid if grid is None else grid
            self.depth = self.variables[0].depth if depth is None else depth
            self.grid_file = self.variables[0].grid_file if grid_file is None else grid_file
            self.data_file = self.variables[0].data_file if data_file is None else data_file

#         self._check_consistency()
        self._result_memo = OrderedDict()

    def __repr__(self):
        return str(self.serialize())

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    varnames=None,
                    grid_topology=None,
                    name=None,
                    units=None,
                    time=None,
                    grid=None,
                    depth=None,
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
            if grid_file is not None:
                dg = _get_dataset(grid_file)
            else:
                dg = dataset
            ds = dataset

        if grid is None:
            grid = PyGrid.from_netCDF(grid_file,
                                      dataset=dg,
                                      grid_topology=grid_topology)
        if varnames is None:
            varnames = cls._gen_varnames(data_file,
                                         dataset=ds)
        if name is None:
            name = cls.__name__ + str(cls._def_count)
            cls._def_count += 1
        data = ds[varnames[0]]
        if time is None:
            time = Time.from_netCDF(filename=data_file,
                                    dataset=ds,
                                    datavar=data)
        if depth is None:
            if (isinstance(grid, PyGrid_S) and len(data.shape) == 4 or 
                        (len(data.shape) == 3 and time is None) or 
                    (isinstance(grid, PyGrid_U) and len(data.shape) == 3 or 
                        (len(data.shape) == 2 and time is None))):
                from gnome.environment.environment_objects import Depth
                depth = Depth(surface_index=-1)
#             if len(data.shape) == 4 or (len(data.shape) == 3 and time is None):
#                 from gnome.environment.environment_objects import S_Depth
#                 depth = S_Depth.from_netCDF(grid=grid,
#                                             depth=1,
#                                             data_file=data_file,
#                                             grid_file=grid_file,
#                                             **kwargs)
        variables = OrderedCollection(dtype=EnvProp)
        for vn in varnames:
            variables.append(GriddedProp.from_netCDF(filename=filename,
                                                     varname=vn,
                                                     grid_topology=grid_topology,
                                                     units=units,
                                                     time=time,
                                                     grid=grid,
                                                     depth=depth,
                                                     data_file=data_file,
                                                     grid_file=grid_file,
                                                     dataset=ds,
                                                     load_all=load_all,
                                                     **kwargs))
        if units is None:
            units = [v.units for v in variables]
            if all(u == units[0] for u in units):
                units = units[0]
        return cls(filename=filename,
                   varnames=varnames,
                   grid_topology=grid_topology,
                   units=units,
                   time=time,
                   grid=grid,
                   depth=depth,
                   variables=variables,
                   data_file=data_file,
                   grid_file=grid_file,
                   dataset=ds,
                   load_all=load_all,
                   **kwargs)

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
            if all([sn in df.variables.keys() for sn in n]):
                return n
        raise ValueError("Default names not found.")

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
        if self.variables is not None:
            return self.variables[0].data.shape
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

    def at(self, points, time, units=None, extrapolate=False, memoize=True, _hash=None, **kwargs):
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
                                               extrapolate=extrapolate,
                                               memoize=memoize,
                                               _hash=_hash,
                                               **kwargs)

        if mem:
            self._memoize_result(points, time, value, self._result_memo, _hash=_hash)
        return value


    @classmethod
    def _get_shared_vars(cls, *sh_args):
        default_shared = ['dataset', 'data_file', 'grid_file', 'grid']
        if len(sh_args) != 0:
            shared = sh_args
        else:
            shared = default_shared

        def getvars(func):
            @wraps(func)
            def wrapper(*args, **kws):
                def _mod(n):
                    k = kws
                    s = shared
                    return (n in s) and ((n not in k) or (n in k and k[n] is None))
                if 'filename' in kws and kws['filename'] is not None:
                    kws['data_file'] = kws['grid_file'] = kws['filename']
                if _mod('dataset'):
                    if 'grid_file' in kws and 'data_file' in kws:
                        if kws['grid_file'] == kws['data_file']:
                            ds = dg = _get_dataset(kws['grid_file'])
                        else:
                            ds = _get_dataset(kws['data_file'])
                            dg = _get_dataset(kws['grid_file'])
                    kws['dataset'] = ds
                else:
                    if 'grid_file' in kws and kws['grid_file'] is not None:
                        dg = _get_dataset(kws['grid_file'])
                    else:
                        dg = kws['dataset']
                    ds = kws['dataset']
                if _mod('grid'):
                    gt = kws.get('grid_topology', None)
                    kws['grid'] = PyGrid.from_netCDF(kws['grid_file'], dataset=dg, grid_topology=gt)
                if kws.get('varnames', None) is None:
                    varnames = cls._gen_varnames(kws['data_file'],
                                                 dataset=ds)
#                 if _mod('time'):
#                     time = Time.from_netCDF(filename=kws['data_file'],
#                                             dataset=ds,
#                                             varname=data)
#                     kws['time'] = time
                return func(*args, **kws)
            return wrapper
        return getvars
