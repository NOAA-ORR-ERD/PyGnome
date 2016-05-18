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
                 data_file=None,
                 grid_file=None):

        self._grid = self._data_file = self._grid_file = None

        if any([units is None, time is None, grid is None, data is None]):
            raise ValueError("All attributes except name MUST be defined if variables is not a list of GriddedProp objects")
        if not hasattr(data, 'shape'):
            if isinstance(grid, pyugrid.UGrid) and not grid.fits_data(data):
                raise ValueError('Data must be able to fit to the grid')
            if isinstance(grid,pysgrid.SGrid) and grid.infer_grid(data) is None:
                raise ValueError('Data must be able to fit to the grid')
        super(GriddedProp, self).__init__(name=name, units=units, time=time, data=data)
        self._grid = grid
        self.data_file = data_file
        self.grid_file = grid_file

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    varname=None,
                    grid_topology=None,
                    name=None,
                    units=None,
                    time=None,
                    grid=None,
                    data_file=None,
                    grid_file=None):
        if filename is not None:
            data_file = filename
            grid_file = filename
        gt = grid_topology
        gf = nc4.Dataset(grid_file)
        df = gf
        if data_file != grid_file:
            df = nc4.Dataset(data_file)
        if grid is None:
            if gt is None:
                try:
                    grid = pyugrid.UGrid.from_nc_dataset(gf)
                except ValueError:
                    grid = pysgrid.SGrid.load_grid(gf)
            else:
                nodes = node_lon = node_lat = None
                if 'nodes' not in gt:
                    if 'node_lon' not in gt and 'node_lat' not in gt:
                        raise ValueError('Nodes must be specified with either the "nodes" or "node_lon" and "node_lat" keys')
                    node_lon = gf[gt['node_lon']]
                    node_lat = gf[gt['node_lat']]
                else:
                    nodes = gf[gt['nodes']]
                if 'faces' in gt and gf[gt['faces']]:
                    #UGrid
                    faces = gf[gt['faces']]
                    if faces.shape[0] == 3:
                        faces=np.ascontiguousarray(np.array(faces).T - 1)
                    if nodes is None:
                        nodes = np.column_stack((node_lon, node_lat))
                    grid = pyugrid.UGrid(nodes = nodes, faces=faces)
                else:
                    #SGrid
                    center_lon = center_lat = edge1_lon = edge1_lat = edge2_lon = edge2_lat = None
                    if node_lon is None:
                        node_lon = nodes[:,0]
                    if node_lat is None:
                        node_lat = nodes[:,1]
                    if 'center_lon' in gt:
                        center_lon = gf[gt['center_lon']]
                    if 'center_lat' in gt:
                        center_lat = gf[gt['center_lat']]
                    if 'edge1_lon' in gt:
                        edge1_lon = gf[gt['edge1_lon']]
                    if 'edge1_lat' in gt:
                        edge1_lat = gf[gt['edge1_lat']]
                    if 'edge2_lon' in gt:
                        edge2_lon = gf[gt['edge2_lon']]
                    if 'edge2_lat' in gt:
                        edge2_lat = gf[gt['edge2_lat']]
                    grid = pysgrid.SGrid(node_lon = node_lon,
                                         node_lat = node_lat,
                                         center_lon = center_lon,
                                         center_lat = center_lat,
                                         edge1_lon = edge1_lon,
                                         edge1_lat = edge1_lat,
                                         edge2_lon = edge2_lon,
                                         edge2_lat = edge2_lat)
        if varname is None:
            raise ValueError("Must supply variable name")
        if varname not in df.variables.keys():
            raise ValueError("Data file does not contain variable name {0}".format(varname))
        data = df[varname]
        name = varname if name is None else name
        units = data.units if units is None else units
        timevar=None
        if time is None:
            try:
                timevar = data.time
            except AttributeError:
                timevar = data.dimensions[0]
            time = Time(df[timevar])

        prop = cls(name=name,
                   units=units,
                   time=time,
                   data=data,
                   grid=grid,
                   data_file=data_file,
                   grid_file=grid_file)
        return prop


    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        if self.data is not None and len(t) != len(self.data) and len(t) > 1:
            raise ValueError("Data/time interval mismatch")
        if isinstance(t, Time):
            self._time = t
        elif isinstance(t,collections.Iterable) or isinstance(t, nc4.Variable):
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
        if self.grid is not None and self.grid.infer_grid(d) is None:
            raise ValueError("Data/grid shape mismatch. Data shape is {0}, Grid shape is {1}".format(d.shape, grid.shape))
        self._data = d

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, g):
        if not (isinstance(g, (pyugrid.UGrid, pysgrid.SGrid))):
            raise ValueError('Grid must be set with a pyugrid.UGrid or pysgrid.SGrid object')
        if self.data is not None and g.infer_grid(self.data) is None:
            raise ValueError("Data/grid shape mismatch. Data shape is {0}, Grid shape is {1}".format(d.shape, grid.shape))
        self._grid = g

    @property
    def data_shape(self):
        return self.data.shape

    @property
    def is_data_on_nodes(self):
        return self.grid.infer_grid(self._data) == 'node'

    def set_attr(self,
                 name=None,
#                  units=None,
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

    def center_values(self, time, units=None, extrapolate=False):
        #NOT COMPLETE
        if not extrapolate:
            self.time.valid_time(time)
        if len(self.time) == 1:
            if len(self.data.shape) == 2:
                if isinstance(self.grid, pysgrid.sgrid):
                    #curv grid
                    value = self.data[0:1:-2,1:-2]
                else:
                    value = self.data
            if units is not None and units != self.units:
                value = unit_conversion.convert(None, self.units, units, value)
        else:
            t_index = self.time.index_of(time, extrapolate)
            centers = self.grid.get_center_points()
            value = self.at(centers, time, units)
        return value

    def at(self, points, time, units=None, depth = -1, extrapolate=False):
        '''
        Interpolates this property to the given points at the given time.
        :param points: A Nx2 array of lon,lat points
        :param time: A datetime object. May be None; if this is so, the variable is assumed to be gridded
        but time-invariant
        '''
        t_alphas = t_index = s0 = s1 = value = None
        if not extrapolate:
            self.time.valid_time(time)
        t_index = self.time.index_of(time, extrapolate)
        if len(self.time) == 1:
            value = self.grid.interpolate_var_to_points(points, self.data, slices=s0, memo=True)
        else:
            if time > self.time.max_time:
                value = self.data[-1]
            if time <= self.time.min_time:
                value = self.data[0]
            if extrapolate and t_index == len(self.time.time):
                s0 = [t_index]
                value = self.grid.interpolate_var_to_points(points, self.data, slices=s0, memo=True)
            else:
                t_alphas = self.time.interp_alpha(time, extrapolate)
                s1 = [t_index]
                s0 = [t_index - 1]
                if len(self.data.shape) == 4:
                    s0.append(depth)
                    s1.append(depth)

                #THIS WILL BE REMOVED WHEN PYUGRID AND SGRID API IS CLOSER
                if isinstance(self.grid, pysgrid.SGrid):
                    v0 = self.grid.interpolate_var_to_points(points, self.data, slices=s0, slice_grid=True, memo=True)
                    v1 = self.grid.interpolate_var_to_points(points, self.data, slices=s1, slice_grid=True, memo=True)
                    value = v0 + (v1 - v0) * t_alphas
                else:
                    v0 = self.grid.interpolate_var_to_points(points, self.data[t_index])
                    v1 = self.grid.interpolate_var_to_points(points, self.data[t_index])
                    value = v0 + (v1 - v0) * t_alphas
                #THIS WILL BE REMOVED WHEN PYUGRID AND SGRID API IS CLOSER

        if units is not None and units != self.units:
            value = unit_conversion.convert(None, self.units, units, value)
        return value


class GridVectorProp(VectorProp):

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 grid = None,
                 grid_file=None,
                 data_file=None,
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
                            grid = grid,
                            data_file=data_file,
                            grid_file=grid_file)

        self._check_consistency()

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
                    grid_file=None
                    ):
        variables = []
        if filename is not None:
            data_file = filename
            grid_file = filename
        grid = init_grid(grid_file,
                                       grid_topology=grid_topology)
        for vn in varnames:
            variables.append(GriddedProp.from_netCDF(filename=filename,
                                                     varname=vn,
                                                     grid_topology=grid_topology,
                                                     units=units,
                                                     time=time,
                                                     grid=grid,
                                                     data_file=data_file,
                                                     grid_file=grid_file))
        return cls(name,
                   units,
                   time,
                   variables,
                   grid=grid,
                   grid_file=grid_file,
                   data_file=data_file)


    def _check_consistency(self):
        '''
        Checks that the attributes of each GriddedProp in varlist are the same as the GridVectorProp
        '''
        if self.units is None or self.time is None or self.grid is None:
            return
        for v in self.variables:
            if v.units != self.units:
                raise ValueError("Variable {0} did not have units consistent with what was specified. Got: {1} Expected {2}".format(v.name,v.units, self.units))
            if v.time != self.time:
                raise ValueError("Variable {0} did not have time consistent with what was specified Got: {1} Expected {2}".format(v.name,v.time, self.time))
            if v.grid != self.grid:
                raise ValueError("Variable {0} did not have grid consistent with what was specified Got: {1} Expected {2}".format(v.name,v.grid, self.grid))
            if v.grid_file != self.grid_file:
                raise ValueError("Variable {0} did not have grid_file consistent with what was specified Got: {1} Expected {2}".format(v.name,v.grid_file, self.grid_file))
            if v.data_file != self.data_file:
                raise ValueError("Variable {0} did not have data_file consistent with what was specified Got: {1} Expected {2}".format(v.name,v.data_file, self.data_file))

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, g):
        if not (isinstance(g, (pyugrid.UGrid, pysgrid.SGrid))):
            raise ValueError('Grid must be set with a pyugrid.UGrid or pysgrid.SGrid object')
        if self._variables is not None:
            if g.infer_grid(self.variables[0]) is None:
                raise ValueError("Grid with shape {0} not compatible with data of shape {1}".format(g.shape, self.data_shape))
            for v in self.variables:
                v.grid = g
        else:
            self._grid = g


    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, vars):
        if vars is None:
            self._variables = None
            return
        new_vars = []
        for i, var in enumerate(vars):
            if not isinstance(var, GriddedProp):
                if (isinstance(var, (collections.Iterable, nc4.Variable)) and
                    len(var) == len(self.time) and
                    self.grid.infer_grid(var) is not None):
                    new_vars.append(GriddedProp(name='var{0}'.format(i),
                                    units=self.units,
                                    time=self.time,
                                    grid = self.grid,
                                    data = vars[i],
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
        return self.grid.infer_grid(self.variables[0].data) == 'node'

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
        elif isinstance(t,collections.Iterable) or isinstance(t, nc4.Variable):
            self._time = Time(t)
        else:
            raise ValueError("Time must be set with an iterable container or netCDF variable")

    @property
    def data_shape(self):
        if self._variables is not None:
            return self.variables.data.shape
        else:
            return None

    def set_attr(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
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
                         grid = grid,
                         grid_file = grid_file,
                         data_file = data_file,)
        else:
            for i, var in enumerate(self.variables):
                var.set_attr(units=units,
                             time=time,
                             grid = grid,
                             grid_file = grid_file,
                             data_file = data_file,)
        self._units = units
        self._time = time
        self._grid = grid
        self.grid_file = grid_file
        self.grid_file = grid_file

def init_grid(filename,
              grid_topology=None):
    gt = grid_topology
    gf = nc4.Dataset(filename)
    grid = None
    if gt is None:
        try:
            grid = pyugrid.UGrid.from_nc_dataset(gf)
        except ValueError:
            pass
        try:
            grid = pysgrid.SGrid.load_grid(gf)
        except ValueError:
            gt = _gen_topology(filename)
    if grid is None:
        nodes = node_lon = node_lat = None
        if 'nodes' not in gt:
            if 'node_lon' not in gt and 'node_lat' not in gt:
                raise ValueError('Nodes must be specified with either the "nodes" or "node_lon" and "node_lat" keys')
            node_lon = gf[gt['node_lon']]
            node_lat = gf[gt['node_lat']]
        else:
            nodes = gf[gt['nodes']]
        if 'faces' in gt and gf[gt['faces']]:
            #UGrid
            faces = gf[gt['faces']]
            if faces.shape[0] == 3:
                faces=np.ascontiguousarray(np.array(faces).T - 1)
            if nodes is None:
                nodes = np.column_stack((node_lon, node_lat))
            grid = pyugrid.UGrid(nodes = nodes, faces=faces)
        else:
            #SGrid
            center_lon = center_lat = edge1_lon = edge1_lat = edge2_lon = edge2_lat = None
            if node_lon is None:
                node_lon = nodes[:,0]
            if node_lat is None:
                node_lat = nodes[:,1]
            if 'center_lon' in gt:
                center_lon = gf[gt['center_lon']]
            if 'center_lat' in gt:
                center_lat = gf[gt['center_lat']]
            if 'edge1_lon' in gt:
                edge1_lon = gf[gt['edge1_lon']]
            if 'edge1_lat' in gt:
                edge1_lat = gf[gt['edge1_lat']]
            if 'edge2_lon' in gt:
                edge2_lon = gf[gt['edge2_lon']]
            if 'edge2_lat' in gt:
                edge2_lat = gf[gt['edge2_lat']]
            grid = pysgrid.SGrid(node_lon = node_lon,
                                 node_lat = node_lat,
                                 center_lon = center_lon,
                                 center_lat = center_lat,
                                 edge1_lon = edge1_lon,
                                 edge1_lat = edge1_lat,
                                 edge2_lon = edge2_lon,
                                 edge2_lat = edge2_lat)
    return grid

def _gen_topology(filename):
    '''
    Function to determine create the correct default topology if it is not provided

    :param filename: Name of file that will be searched for variables
    :return: List of default variable names, or None if none are found
    '''
    gf = nc4.Dataset(filename)
    gt = {}
    node_coord_names = [['node_lon','node_lat'], ['lon', 'lat'], ['lon_psi', 'lat_psi']]
    face_var_names = ['nv']
    center_coord_names = [['center_lon', 'center_lat'], ['lon_rho', 'lat_rho']]
    edge1_coord_names = [['edge1_lon', 'edge1_lat'], ['lon_u', 'lat_u']]
    edge2_coord_names = [['edge2_lon', 'edge2_lat'], ['lon_v', 'lat_v']]
    for n in node_coord_names:
        if n[0] in gf.variables.keys() and n[1] in gf.variables.keys():
            gt['node_lon'] = n[0]
            gt['node_lat'] = n[1]
            break

    if 'node_lon' not in gt:
        raise NameError('Default node topology names are not in the grid file')

    for n in face_var_names:
        if n in gf.variables.keys():
            gt['faces'] = n
            break

    if 'faces' in gt.keys():
        #UGRID
        return gt
    else:
        for n in center_coord_names:
            if n[0] in gf.variables.keys() and n[1] in gf.variables.keys():
                gt['center_lon'] = n[0]
                gt['center_lat'] = n[1]
                break
        for n in edge1_coord_names:
            if n[0] in gf.variables.keys() and n[1] in gf.variables.keys():
                gt['edge1_lon'] = n[0]
                gt['edge1_lat'] = n[1]
                break
        for n in edge2_coord_names:
            if n[0] in gf.variables.keys() and n[1] in gf.variables.keys():
                gt['edge2_lon'] = n[0]
                gt['edge2_lat'] = n[1]
                break
    return gt
