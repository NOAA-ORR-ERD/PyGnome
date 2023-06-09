
import os
import datetime
import copy
import numpy as np
# import logging
import warnings
from functools import wraps

from colander import (SchemaNode, SequenceSchema,
                      String, Boolean, DateTime,
                      drop, Int)

import gridded
from gridded.utilities import get_dataset
import nucos as uc

from gnome.gnomeobject import combine_signatures
from gnome.persist import base_schema
from gnome.gnomeobject import GnomeId
from gnome.persist import (GeneralGnomeObjectSchema, SchemaNode, SequenceSchema,
                           String, Boolean, DateTime, drop, FilenameSchema)
from gnome.persist.validators import convertible_to_seconds
from gnome.persist.extend_colander import LocalDateTime
from gnome.utilities.inf_datetime import InfDateTime


class TimeSchema(base_schema.ObjTypeSchema):
    filename = FilenameSchema(
        isdatafile=True, test_equal=False, update=False
    )
    varname = SchemaNode(
        String(), read_only=True
    )
    data = SequenceSchema(
        SchemaNode(
            DateTime(default_tzinfo=None)
        )
    )
    min_time = SchemaNode(
        DateTime(default_tzinfo=None), read_only=True
    )
    max_time = SchemaNode(
        DateTime(default_tzinfo=None), read_only=True
    )


class GridSchema(base_schema.ObjTypeSchema):
    name = SchemaNode(String(), test_equal=False)
    filename = FilenameSchema(
        isdatafile=True, test_equal=False, update=False
    )

class DepthSchema(base_schema.ObjTypeSchema):
    filename = FilenameSchema(
        isdatafile=True, test_equal=False, update=False
    )

class S_DepthSchema(DepthSchema):
    vtransform = SchemaNode(Int())
    zero_ref = SchemaNode(String())


class VariableSchemaBase(base_schema.ObjTypeSchema):
    #filename
    #data?
    units = SchemaNode(String())
    time = TimeSchema(
        save_reference=True
    )
    grid = GridSchema(
        save_reference=True
    )
    data_file = FilenameSchema(
        isdatafile=True, test_equal=False, update=False
    )
    grid_file = FilenameSchema(
        isdatafile=True, test_equal=False, update=False
    )
    extrapolation_is_allowed = SchemaNode(Boolean())
    data_start = SchemaNode(LocalDateTime(), read_only=True,
                            validator=convertible_to_seconds)
    data_stop = SchemaNode(LocalDateTime(), read_only=True,
                           validator=convertible_to_seconds)


class VariableSchema(VariableSchemaBase):
    varname = SchemaNode(
        String(), missing=drop, read_only=True
    )


class VectorVariableSchema(VariableSchemaBase):
    varnames = SequenceSchema(
        SchemaNode(String()),
        read_only=True
    )
    variables = SequenceSchema(
        GeneralGnomeObjectSchema(
            acceptable_schemas=[VariableSchema, base_schema.ObjTypeSchema]
        ), save_reference=True
    )


class Time(gridded.time.Time, GnomeId):

    _schema = TimeSchema
    def __repr__(self):
        try:
            return super().__repr__()
        except ValueError:
            return object.__repr__(self)

    @classmethod
    def from_file(cls, filename=None, **kwargs):
        if isinstance(filename, list):
            filename = filename[0]

        t = []

        with open(filename, 'r') as fd:
            for line in fd:
                line = line.rstrip()
                if line is not None:
                    t.append(datetime.datetime.strptime(line, '%c'))

        return Time(t)


class Grid_U(gridded.grids.Grid_U, GnomeId):

    _schema = GridSchema

    def __init__(self, **kwargs):
        super(Grid_U, self).__init__(**kwargs)

        #This is for the COOPS case, where their coordinates go from 0-360 starting at prime meridian
        for lon in [self.node_lon,]:
            if lon is not None and lon.max() > 180:
                self.logger.warning('Detected longitudes > 180 in {0}. Rotating -360 degrees'.format(self.name))
                lon -= 360

    def draw_to_plot(self, ax, features=None, style=None):
        import matplotlib
        def_style = {'color': 'blue',
                     'linestyle': 'solid'}
        s = def_style.copy()

        if style is not None:
            s.update(style)

        lines = self.get_lines()
        lines = matplotlib.collections.LineCollection(lines, **s)

        ax.add_collection(lines)

    @classmethod
    @combine_signatures
    def new_from_dict(cls, dict_):
        rv = cls.from_netCDF(**dict_)

        return rv

    def get_cells(self):
        return self.nodes[self.faces]

    def get_lines(self):
        '''
        Returns an array of lengths, and a list of line arrays.
        The first array sequentially indexes the second array.
        When the second array is split up using the first array
        and the resulting lines are drawn, you should end up with a picture of
        the grid.
        '''
        open_cells = self.nodes[self.faces]
        closed_cells = np.concatenate((open_cells, open_cells[:, None, 0]),
                                      axis=1)
        closed_cells = closed_cells.astype(np.float32)
        lengths = closed_cells.shape[1] * np.ones(closed_cells.shape[0],
                                                  dtype=np.int32)

        return (lengths, [closed_cells])

    def get_nodes(self):
        return self.nodes[:]

    def get_centers(self):
        if self.face_coordinates is None:
            self.build_face_coordinates()
        return self.face_coordinates

    def get_metadata(self):
        json_ = {}
        json_['nodes_shape'] = self.nodes.shape
        json_['num_nodes'] = self.nodes.shape[0]
        json_['num_cells'] = self.faces.shape[0]
        return json_


class Grid_S(GnomeId, gridded.grids.Grid_S):

    _schema = GridSchema

    def __init__(self, use_masked_boundary=True, *args, **kwargs):
        super(Grid_S, self).__init__(*args, use_masked_boundary=use_masked_boundary, **kwargs)

        '''
        #This is for the COOPS case, where their coordinates go from 0-360 starting at prime meridian
        for lon in [self.node_lon, self.center_lon, self.edge1_lon, self.edge2_lon]:
            if lon is not None and lon.max() > 180:
                self.logger.warning('Detected longitudes > 180 in {0}. Rotating -360 degrees'.format(self.name))
                lon -= 360
        '''

    '''hack to avoid problems when registering object in webgnome'''
    @property
    def non_grid_variables(self):
        return None

    def draw_to_plot(self, ax, features=None, style=None):
        def_style = {'node': {'color': 'green',
                              'linestyle': 'dashed',
                              'marker': 'o'},
                     'center': {'color': 'blue',
                                'linestyle': 'solid'},
                     'edge1': {'color': 'purple'},
                     'edge2': {'color': 'olive'}}

        if features is None:
            features = ['node']
        st = def_style.copy()

        if style is not None:
            for k in style.keys():
                st[k].update(style[k])

        for f in features:
            s = st[f]
            lon, lat = self._get_grid_vars(f)

            ax.plot(lon, lat, **s)
            ax.plot(lon.T, lat.T, **s)

    @classmethod
    @combine_signatures
    def new_from_dict(cls, dict_):
        rv = cls.from_netCDF(**dict_)
        return rv

    def get_cells(self):
        if self._cell_tree is None:
            self.build_celltree()

        ns = self._cell_tree[1]
        fs = self._cell_tree[2]

        return ns[fs]

    def get_nodes(self):
        if self._cell_tree is None:
            self.build_celltree()

        n = self._cell_tree[1]

        return n

    def get_centers(self):
        if self.center_lon is None:
            lons = (self.node_lon[0:-1, 0:-1] + self.node_lon[1:, 1:]) / 2
            lats = (self.node_lat[0:-1, 0:-1] + self.node_lat[1:, 1:]) / 2
            return np.stack((lons, lats), axis=-1).reshape(-1, 2)
        else:
            if self._get_geo_mask('center'):
                if not self._cell_tree:
                    self.build_celltree(use_mask=True)
                ctr_padding_slice = self.get_padding_slices(self.center_padding)
                ctr_mask = gridded.utilities.gen_celltree_mask_from_center_mask(self.center_mask, ctr_padding_slice)
                clons = self.center_lon[ctr_padding_slice]
                clats = self.center_lat[ctr_padding_slice]
                clons = np.ma.MaskedArray(clons, mask = ctr_mask)
                clats = np.ma.MaskedArray(clats, mask = ctr_mask)
                return np.stack((clons.compressed(), clats.compressed()), axis=-1).reshape(-1, 2)
            return self.centers.reshape(-1, 2)

    def get_metadata(self):
        if self._cell_tree is None:
            self.build_celltree()
        json_ = {}
        json_['nodes_shape'] = self.nodes.shape
        json_['num_nodes'] = self.nodes.shape[0] * self.nodes.shape[1]
        json_['num_cells'] = self._cell_tree[2].shape[0]
        return json_

    def get_lines(self):
        '''
        Returns an array of lengths, and a list of line arrays.
        The first array sequentially indexes the second array.
        When the second array is split up using the first array
        and the resulting lines are drawn, you should end up with a picture of
        the grid.
        '''
        if self._get_geo_mask('node') is not None:
            #masked sgrid, so use Grid_U style of lines
            if self._cell_tree is None:
                self.build_celltree()
            nodes = self._cell_tree[1]
            faces = self._cell_tree[2]
            open_cells = nodes[faces]
            closed_cells = np.concatenate((open_cells, open_cells[:, None, 0]),
                                          axis=1)
            closed_cells = closed_cells.astype(np.float32, copy=False)
            lengths = closed_cells.shape[1] * np.ones(closed_cells.shape[0],
                                                      dtype=np.int32)
            return (lengths, [closed_cells])

        hor_lines = (np.dstack((self.node_lon[:], self.node_lat[:])).astype(np.float32))
        ver_lines = (hor_lines.transpose((1, 0, 2)).astype(np.float32))

        hor_lens = hor_lines.shape[1] * np.ones(hor_lines.shape[0],
                                                dtype=np.int32)
        ver_lens = ver_lines.shape[1] * np.ones(ver_lines.shape[0],
                                                dtype=np.int32)
        lens = np.concatenate((hor_lens, ver_lens))
        return (lens, [hor_lines, ver_lines])


class Grid_R(gridded.grids.Grid_R, GnomeId):

    _schema = GridSchema

    @classmethod
    def new_from_dict(cls, dict_):
        rv = cls.from_netCDF(**dict_)
        return rv

    def get_metadata(self):
        json_ = {}
        json_['shape'] = self.nodes.shape
        return json_

    def get_nodes(self):
        return self.nodes.reshape(-1, 2)

    def get_centers(self):
        return self.centers.reshape(-1, 2)

    def get_cells(self):
        return np.concatenate(self.node_lon, self.node_lat)

    def get_lines(self):

        lon_lines = np.array([[(lon, self.node_lat[0]),
                               (lon, self.node_lat[len(self.node_lat) // 2]),
                               (lon, self.node_lat[-1])]
                              for lon in self.node_lon], dtype=np.float32)
        lat_lines = np.array([[(self.node_lon[0], lat),
                               (self.node_lon[len(self.node_lon) // 2], lat),
                               (self.node_lon[-1], lat)]
                              for lat in self.node_lat], dtype=np.float32)

        lon_lens = lon_lines.shape[1] * np.ones(lon_lines.shape[0],
                                                dtype=np.int32)
        lat_lens = lat_lines.shape[1] * np.ones(lat_lines.shape[0],
                                                dtype=np.int32)
        lens = np.concatenate((lon_lens, lat_lens))

        return (lens, [lon_lines, lat_lines])


class PyGrid(gridded.grids.Grid):

    @staticmethod
    def from_netCDF(*args, **kwargs):
        kwargs['_default_types'] = (('ugrid', Grid_U),
                                    ('sgrid', Grid_S),
                                    ('rgrid', Grid_R))

        return gridded.grids.Grid.from_netCDF(*args, **kwargs)

    @staticmethod
    def new_from_dict(dict_):
        return PyGrid.from_netCDF(**dict_)

    @staticmethod
    def _get_grid_type(*args, **kwargs):
        kwargs['_default_types'] = (('ugrid', Grid_U),
                                    ('sgrid', Grid_S),
                                    ('rgrid', Grid_R))

        return gridded.grids.Grid._get_grid_type(*args, **kwargs)


class Depth(gridded.depth.Depth):
    @staticmethod
    def from_netCDF(*args, **kwargs):
        kwargs['_default_types'] = (('level', L_Depth),
                                    ('sigma', S_Depth),
                                    ('surface', DepthBase))

        return gridded.depth.Depth.from_netCDF(*args, **kwargs)

    @staticmethod
    def _get_depth_type(*args, **kwargs):
        kwargs['_default_types'] = (('level', L_Depth),
                                    ('sigma', S_Depth),
                                    ('surface', DepthBase))

        return gridded.depth.Depth._get_depth_type(*args, **kwargs)


class Variable(gridded.Variable, GnomeId):
    _schema = VariableSchema

    default_names = []
    cf_names = []

    _default_component_types = copy.deepcopy(gridded.Variable
                                             ._default_component_types)
    _default_component_types.update({'time': Time,
                                     'grid': PyGrid,
                                     'depth': Depth})

    _gnome_unit = None #Default assumption for unit type

    # fixme: shouldn't extrapolation_is_allowed be
    #        in Environment only?
    def __init__(self, extrapolation_is_allowed=False, *args, **kwargs):
        super(Variable, self).__init__(*args, **kwargs)
        self.extrapolation_is_allowed = extrapolation_is_allowed

    def __repr__(self):
        try:
            return super().__repr__()
        except ValueError:
            return object.__repr__(self)

    def init_from_netCDF(self,
                         filename=None,
                         varname=None,
                         grid_topology=None,
                         name=None,
                         units=None,
                         time=None,
                         time_origin=None,
                         grid=None,
                         depth=None,
                         dataset=None,
                         data_file=None,
                         grid_file=None,
                         location=None,
                         load_all=False,
                         fill_value=0,
                         **kwargs
                         ):
        '''
        Initialize a Variable from a netcdf file.

        This is use in subclasses, to it can be done after object creation

        :param filename: Default data source. Has lowest priority.
                         If dataset, grid_file, or data_file are provided,
                         this function uses them first
        :type filename: string

        :param varname: Explicit name of the data in the data source file.
                        Equivalent to the key used to look the item up
                        directly eg 'ds["lon_u"]' for a netCDF4 Dataset.
        :type varname: string

        :param grid_topology: Description of the relationship between grid
                              attributes and variable names that hold the grid.
        :type grid_topology: {string : string, ...}

        :param name: Name of this object (GNome Object name)
        :type name: string

        :param units: string such as 'm/s'
        :type units: string

        :param time: Time axis of the data. May be a constructed ``gridded.Time``
                     object, or collection of datetime.datetime objects
        :type time: [] of datetime.datetime, netCDF4 Variable, or Time object

        :param data: Underlying data object. May be any array-like,
                     including netCDF4 Variable, etc
        :type data: netCDF4.Variable or numpy.array

        :param grid: Grid that the data corresponds to
        :type grid: pysgrid or pyugrid

        :param location: The feature where the data aligns with the grid.
                         e.g. "node", "face"
        :type location: string

        :param depth: Depth axis object from ``gridded.depth``
        :type depth: Depth, S_Depth or L_Depth

        :param dataset: Instance of open netCDF4.Dataset
        :type dataset: netCDF4.Dataset

        :param data_file: Name of data source file, if data and grid files are separate
        :type data_file: string

        :param grid_file: Name of grid source file, if data and grid files are separate
        :type grid_file: string

        :param extrapolation_is_allowed:
        '''
        Grid = self._default_component_types['grid']
        Time = self._default_component_types['time']
        Depth = self._default_component_types['depth']
        if filename is not None:
            try:
                filename = os.fspath(filename)
            except TypeError:
                pass
            data_file = grid_file = filename
        ds = None
        dg = None
        if dataset is None:
            if grid_file == data_file:
                ds = dg = get_dataset(grid_file)
            else:
                ds = get_dataset(data_file)
                dg = get_dataset(grid_file)
        else:
            if grid_file is not None:
                dg = get_dataset(grid_file)
            else:
                dg = dataset
            ds = dataset
        if data_file is None:
            data_file = os.path.split(ds.filepath())[-1]

        if grid is None:
            grid = Grid.from_netCDF(grid_file,
                                    dataset=dg,
                                    grid_topology=grid_topology)
        if varname is None:
            varname = self._gen_varname(data_file,
                                       dataset=ds)
            if varname is None:
                raise NameError('Default current names are not in the data file, '
                                'must supply variable name')
        data = ds[varname]
        if name is None:
            name = self.__class__.__name__ + str(self._def_count)
            self._def_count += 1
        if units is None:
            try:
                units = data.units
            except AttributeError:
                units = None
        if time is None:
            time = Time.from_netCDF(filename=data_file,
                                    dataset=ds,
                                    datavar=data)
            if time_origin is not None:
                time = Time(data=time.data,
                            filename=time.filename,
                            varname=time.varname,
                            origin=time_origin)
        if depth is None:
            if (isinstance(grid, (Grid_S, Grid_R)) and len(data.shape) == 4 or
                    isinstance(grid, Grid_U) and len(data.shape) == 3):
                depth = Depth.from_netCDF(grid_file,
                                          dataset=dg,
                                          )
        if location is None:
            if hasattr(data, 'location'):
                location = data.location
#             if len(data.shape) == 4 or (len(data.shape) == 3 and time is None):
#                 from gnome.environment.environment_objects import S_Depth
#                 depth = S_Depth.from_netCDF(grid=grid,
#                                             depth=1,
#                                             data_file=data_file,
#                                             grid_file=grid_file,
#                                             **kwargs)
        if load_all:
            data = data[:]
        self.__init__(name=name,
                      units=units,
                      time=time,
                      data=data,
                      grid=grid,
                      depth=depth,
                      grid_file=grid_file,
                      data_file=data_file,
                      fill_value=fill_value,
                      location=location,
                      varname=varname,
                      **kwargs)

    @classmethod
    @combine_signatures
    def from_netCDF(cls, *args, **kwargs):
        """
        create a new variable object from a netcdf file

        See init_from_netcdf for signature
        """
        var = cls.__new__(cls)
        var.init_from_netCDF(*args, **kwargs)
        return var

    @combine_signatures
    def at(self, points, time, units=None, *args, **kwargs):
        if ('extrapolate' not in kwargs):
            kwargs['extrapolate'] = False
        if ('unmask' not in kwargs):
            kwargs['unmask'] = True

        value = super(Variable, self).at(points, time, *args, **kwargs)

        data_units = self.units if self.units else self._gnome_unit
        req_units = units if units else data_units
        if data_units is not None and data_units != req_units:
            try:
                value = uc.convert(data_units, req_units, value)
            except uc.NotSupportedUnitError:
                if (not uc.is_supported(data_units)):
                    warnings.warn("{0} units is not supported: {1}"
                                  "Using them unconverted as {2}"
                                  .format(self.name, data_units, req_units))
                elif (not uc.is_supported(req_units)):
                    warnings.warn("Requested unit is not supported: {1}".format(req_units))
                else:
                    raise
        return value

    @classmethod
    @combine_signatures
    def new_from_dict(cls, dict_):
        if 'data' not in dict_:
            return cls.from_netCDF(**dict_)

        return super(Variable, cls).new_from_dict(dict_)

    @classmethod
    def constant(cls, value):
        #Sets a Variable up to represent a constant scalar field. The result
        #will return a constant value for all times and places.
        Grid = Grid_S
        Time = cls._default_component_types['time']
        _data = np.full((3,3), value)
        _node_lon = np.array(([-360, 0, 360], [-360, 0, 360], [-360, 0, 360]))
        _node_lat = np.array(([-89.95, -89.95, -89.95], [0, 0, 0], [89.95, 89.95, 89.95]))
        _grid = Grid(node_lon=_node_lon, node_lat=_node_lat)
        _time = Time.constant_time()
        return cls(grid=_grid, time=_time, data=_data, fill_value=value)

    @property
    def extrapolation_is_allowed(self):
        if self.time is not None:
            return self.time.min_time == self.time.max_time or self._extrapolation_is_allowed
        else:
            return self._extrapolation_is_allowed

    @extrapolation_is_allowed.setter
    def extrapolation_is_allowed(self, e):
        self._extrapolation_is_allowed = e

    @property
    def data_start(self):
        return self.time.min_time.replace(tzinfo=None)

    @property
    def data_stop(self):
        return self.time.max_time.replace(tzinfo=None)

    def save(self, saveloc='.', refs=None, overwrite=True):
        return GnomeId.save(self, saveloc=saveloc, refs=refs, overwrite=overwrite)

class DepthBase(gridded.depth.DepthBase, GnomeId):

    _schema = DepthSchema

    _default_component_types = copy.deepcopy(gridded.depth.DepthBase
                                             ._default_component_types)
    _default_component_types.update({'time': Time,
                                     'grid': PyGrid,
                                     'variable': Variable})

    @classmethod
    def new_from_dict(cls, dict_):
        rv = cls.from_netCDF(**dict_)
        return rv

    def interpolation_alphas(self, points, time, data_shape, _hash=None, **kwargs):
        return (None, None)


class L_Depth(gridded.depth.L_Depth, GnomeId):
    _schema = DepthSchema

    _default_component_types = copy.deepcopy(gridded.depth.L_Depth
                                             ._default_component_types)
    _default_component_types.update({'time': Time,
                                     'grid': PyGrid,
                                     'variable': Variable})

    @classmethod
    def new_from_dict(cls, dict_):
        rv = cls.from_netCDF(**dict_)
        return rv


class S_Depth(gridded.depth.S_Depth, GnomeId):

    _schema = S_DepthSchema

    _default_component_types = copy.deepcopy(gridded.depth.S_Depth
                                             ._default_component_types)
    _default_component_types.update({'time': Time,
                                     'grid': PyGrid,
                                     'variable': Variable})

    def __init__(self,
                 zero_ref = 'surface',
                 **kwargs):
        return super(S_Depth, self).__init__(zero_ref=zero_ref, **kwargs)

    @classmethod
    def new_from_dict(cls, dict_):
        rv = cls.from_netCDF(**dict_)
        return rv

class VectorVariable(gridded.VectorVariable, GnomeId):

    _schema = VectorVariableSchema

    _default_component_types = copy.deepcopy(gridded.VectorVariable
                                             ._default_component_types)
    _default_component_types.update({'time': Time,
                                     'grid': PyGrid,
                                     'depth': Depth,
                                     'variable': Variable})

    _gnome_unit = None #Default assumption for unit type

    def __init__(self,
                 extrapolation_is_allowed=False,
                 *args,
                 **kwargs):
        super(VectorVariable, self).__init__(*args, **kwargs)
        self.extrapolation_is_allowed = extrapolation_is_allowed

    def __repr__(self):
        try:
            return super().__repr__()
        except ValueError:
            return object.__repr__(self)

    def init_from_netCDF(self,
                         filename=None,
                         varnames=None,
                         grid_topology=None,
                         name=None,
                         units=None,
                         time=None,
                         time_origin=None,
                         grid=None,
                         depth=None,
                         data_file=None,
                         grid_file=None,
                         dataset=None,
                         load_all=False,
                         variables=None,
                         **kwargs
                         ):
        '''
        Allows one-function initialization of a VectorVariable from a file.

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
        Grid = self._default_component_types['grid']
        Time = self._default_component_types['time']
        Variable = self._default_component_types['variable']
        Depth = self._default_component_types['depth']
        if filename is not None:
            try:
                filename = os.fspath(filename)
            except TypeError:
                pass
            data_file = grid_file = filename
        ds = None
        dg = None
        if dataset is None:
            if grid_file == data_file:
                ds = dg = get_dataset(grid_file)
            else:
                ds = get_dataset(data_file)
                dg = get_dataset(grid_file)
        else:
            if grid_file is not None:
                dg = get_dataset(grid_file)
            else:
                dg = dataset
            ds = dataset

        if grid is None:
            grid = Grid.from_netCDF(grid_file,
                                    dataset=dg,
                                    grid_topology=grid_topology)
        if varnames is None:
            varnames = self._gen_varnames(data_file,
                                         dataset=ds)
            if all([v is None for v in varnames]):
                raise ValueError('No compatible variable names found!')
        if name is None:
            name = self.__class__.__name__ + str(self._def_count)
            self._def_count += 1
        data = ds[varnames[0]]
        if time is None:
            time = Time.from_netCDF(filename=data_file,
                                    dataset=ds,
                                    datavar=data)
            if time_origin is not None:
                time = Time(data=time.data, filename=data_file, varname=time.varname, origin=time_origin)
        if depth is None:
            if (isinstance(grid, (Grid_S, Grid_R)) and len(data.shape) == 4 or
                    isinstance(grid, Grid_U) and len(data.shape) == 3):
                depth = Depth.from_netCDF(grid_file=grid_file,
                                          grid=grid,
                                          dataset=dg,
                                          )

#         if depth is None:
#             if (isinstance(grid, Grid_S) and len(data.shape) == 4 or
#                         (len(data.shape) == 3 and time is None) or
#                     (isinstance(grid, Grid_U) and len(data.shape) == 3 or
#                         (len(data.shape) == 2 and time is None))):
#                 from gnome.environment.environment_objects import S_Depth
#                 depth = S_Depth.from_netCDF(grid=grid,
#                                             depth=1,
#                                             data_file=data_file,
#                                             grid_file=grid_file,
#                                             **kwargs)
        if variables is None:
            variables = []
            for vn in varnames:
                if vn is not None:
                    # Fixme: We're calling from_netCDF from itself ?!?!?
                    variables.append(Variable.from_netCDF(filename=filename,
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
                                                          location=None,
                                                          **kwargs))
        if units is None:
            units = [v.units for v in variables]
            if all(u == units[0] for u in units):
                units = units[0]

        self.__init__(name=name,
                    filename=filename,
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
    def from_netCDF(cls, *args, **kwargs):
        """
        create a new VectorVariable object from a netcdf file

        See init_from_netcdf for signature
        """
        var = cls.__new__(cls)
        var.init_from_netCDF(*args, **kwargs)
        return var

    @classmethod
    def new_from_dict(cls, dict_, **kwargs):
        if not dict_.get('variables', False):
            return super(VectorVariable, cls).new_from_dict(cls.from_netCDF(**dict_).to_dict(), **kwargs)
        else:
            return super(VectorVariable, cls).new_from_dict(dict_, **kwargs)


    @classmethod
    def constant(cls,
                 values,
                 name=None,
                 units=None):
        '''
        Sets a VectorVariable up to represent a constant vector field. The result
        will return a constant value for all times and places.

        :param values: vector of values
        :type values: array-like
        '''
        
        Grid = Grid_S
        Time = cls._default_component_types['time']
        _node_lon = np.array(([-360, 0, 360], [-360, 0, 360], [-360, 0, 360]))
        _node_lat = np.array(([-89.95, -89.95, -89.95], [0, 0, 0], [89.95, 89.95, 89.95]))
        _grid = Grid(node_lon=_node_lon, node_lat=_node_lat)
        _time = Time.constant_time()
        if isinstance(units, str):
            units = [units,]
        _datas = [np.full((3,3), v) for v in values]
        _vars = [Variable(grid=_grid, units=units[i], time=_time, data=d) for i, d in enumerate(_datas)]
        return cls(name=name, grid=_grid, time=_time, variables=_vars)

    def at(self, points, time, units=None, *args, **kwargs):
        if ('extrapolate' not in kwargs):
            kwargs['extrapolate'] = False
        if ('unmask' not in kwargs):
            kwargs['unmask'] = True
        units = units if units else self._gnome_unit #no need to convert here, its handled in the subcomponents
        value = super(VectorVariable, self).at(points, time, units=units, *args, **kwargs)

        return value

    def get_data_vectors(self):
        '''
        return array of shape (time_slices, len_linearized_data,2)
        first is magnitude, second is direction
        '''

        raw_u = self.variables[0].data[:]
        raw_v = self.variables[1].data[:]

        if self.depth is not None:
            raw_u = raw_u[:, self.depth.surface_index]
            raw_v = raw_v[:, self.depth.surface_index]

        if np.any(np.array(raw_u.shape) != np.array(raw_v.shape)):
            # must be roms-style staggered
            u_padding_slice = (np.s_[:],) + self.grid.get_padding_slices(self.grid.edge1_padding)
            v_padding_slice = (np.s_[:],) + self.grid.get_padding_slices(self.grid.edge2_padding)
            raw_u = np.ma.filled(raw_u[u_padding_slice], 0)
            raw_v = np.ma.filled(raw_v[v_padding_slice], 0)
            raw_u = (raw_u[:, :, 0:-1, ] + raw_u[:, :, 1:]) / 2
            raw_v = (raw_v[:, 0:-1, :] + raw_v[:, 1:, :]) / 2
        #u/v should be interpolated to centers at this point. Now apply appropriate mask

        if isinstance(self.grid, Grid_S) and self.grid.center_mask is not None:
            ctr_padding_slice = self.grid.get_padding_slices(self.grid.center_padding)
            if self.grid._cell_tree_mask is None:
                self.grid.build_celltree()

            x = raw_u[:]
            xt = x.shape[0]
            y = raw_v[:]
            yt = y.shape[0]
            if raw_u.shape[-2:] == raw_v.shape[-2:] and raw_u.shape[-2:] == self.grid.center_mask.shape: 
                #raw u/v are same shape and on center
                #need to padding_slice the variable since they are not interpolated from u/v
                x = x[(np.s_[:],) + ctr_padding_slice]
                y = y[(np.s_[:],) + ctr_padding_slice]
            x = x.reshape(xt, -1)
            y = y.reshape(yt, -1)
            ctr_mask = gridded.utilities.gen_celltree_mask_from_center_mask(self.grid.center_mask, ctr_padding_slice)
            x = np.ma.MaskedArray(x, mask = np.tile(ctr_mask.reshape(-1), xt))
            y = np.ma.MaskedArray(y, mask = np.tile(ctr_mask.reshape(-1), yt))
            x = x.compressed().reshape(xt, -1)
            y = y.compressed().reshape(yt,-1)
        else:
            raw_u = np.ma.filled(raw_u, 0).reshape(raw_u.shape[0], -1)
            raw_v = np.ma.filled(raw_v, 0).reshape(raw_v.shape[0], -1)
            r = np.stack((raw_u, raw_v))
            return np.ascontiguousarray(r, np.float32)

        #r = np.ma.stack((x, y)) change to this when numpy 1.15 becomes norm.
        r = np.ma.concatenate((x[None,:], y[None,:]))
        return np.ascontiguousarray(r.astype(np.float32)) # r.compressed().astype(np.float32)

    def get_metadata(self):
        json_ = {}
        json_['data_location'] = self.grid.infer_location(self.variables[0].data)
        return json_

    @property
    def extrapolation_is_allowed(self):
        if self.time is not None:
            return self.time.min_time == self.time.max_time or self._extrapolation_is_allowed
        else:
            return self._extrapolation_is_allowed

    @extrapolation_is_allowed.setter
    def extrapolation_is_allowed(self, e):
        self._extrapolation_is_allowed = e

    @property
    def data_start(self):
        try:
            return self.time.min_time.replace(tzinfo=None)
        except TypeError: # cftime datetime objects don't have a tzinfo attribute.
            return self.time.min_time

    @property
    def data_stop(self):
        try:
            return self.time.max_time.replace(tzinfo=None)
        except TypeError: # cftime datetime objects don't have a tzinfo attribute.
            return self.time.max_time

    def save(self, saveloc='.', refs=None, overwrite=True):
        return GnomeId.save(self, saveloc=saveloc, refs=refs, overwrite=overwrite)

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
                ds = dg =  None
                if _mod('dataset'):
                    if 'grid_file' in kws and 'data_file' in kws:
                        if kws['grid_file'] == kws['data_file']:
                            ds = dg = gridded.utilities.get_dataset(kws['grid_file'])
                        else:
                            ds = gridded.utilities.get_dataset(kws['data_file'])
                            dg = gridded.utilities.get_dataset(kws['grid_file'])
                    kws['dataset'] = ds
                else:
                    if 'grid_file' in kws and kws['grid_file'] is not None:
                        dg = gridded.utilities.get_dataset(kws['grid_file'])
                    else:
                        dg = kws['dataset']
                    ds = kws['dataset']
                if _mod('grid'):
                    gt = kws.get('grid_topology', None)
                    kws['grid'] = PyGrid.from_netCDF(kws['filename'], dataset=dg, grid_topology=gt)
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
