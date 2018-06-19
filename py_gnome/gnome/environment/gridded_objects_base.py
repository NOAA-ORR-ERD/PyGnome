import datetime
import StringIO
import copy
import numpy as np

from colander import (SchemaNode, SequenceSchema,
                      Sequence, String, DateTime,
                      drop)

import gridded

from gnome.persist import base_schema
from gnome.gnomeobject import GnomeId
from gnome.persist.extend_colander import FilenameSchema
from gnome.persist.base_schema import GeneralGnomeObjectSchema


class TimeSchema(base_schema.ObjTypeSchema):
    filename = FilenameSchema(
        isdatafile=True, test_equal=False
    )
    varname = SchemaNode(
        String(), read_only=True
    )
    data = SequenceSchema(
        SchemaNode(
            DateTime(default_tzinfo=None)
        )
    )


class GridSchema(base_schema.ObjTypeSchema):
    name = SchemaNode(String(), test_equal=False) #remove this once gridded stops using _def_count
    filename = FilenameSchema(
        isdatafile=True, test_equal=False
    )

class DepthSchema(base_schema.ObjTypeSchema):
    filename = FilenameSchema(
        isdatafile=True, test_equal=False
    )


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
        isdatafile=True, test_equal=False
    )
    grid_file = FilenameSchema(
        isdatafile=True, test_equal=False
    )


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
        closed_cells = np.concatenate((open_cells, open_cells[:,None,0]), axis=1)
        closed_cells = closed_cells.astype(np.float32, copy=False)
        lengths = closed_cells.shape[1] * np.ones(closed_cells.shape[0], dtype=np.int32)

        return (lengths, [closed_cells])

    def get_nodes(self):
        return self.nodes[:]

    def get_centers(self):
        if self.face_coordinates == None:
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
    def new_from_dict(cls, dict_):
        rv = cls.from_netCDF(**dict_)
        return rv

    def get_cells(self):
        if not hasattr(self, '_cell_trees'):
            self.build_celltree()

        ns = self._cell_trees['node'][1]
        fs = self._cell_trees['node'][2]

        return ns[fs]

    def get_nodes(self):
        if not hasattr(self, '_cell_trees'):
            self.build_celltree()

        n = self._cell_trees['node'][1]

        return n

    def get_centers(self):
        if self.center_lon is None:
            lons = (self.node_lon[0:-1, 0:-1] + self.node_lon[1:,1:]) /2
            lats = (self.node_lat[0:-1, 0:-1] + self.node_lat[1:,1:]) /2
            return np.stack((lons, lats), axis=-1).reshape(-1,2)
        else:
            return self.centers.reshape(-1,2)

    def get_metadata(self):
        json_ = {}
        json_['nodes_shape'] = self.nodes.shape
        json_['num_nodes'] = self.nodes.shape[0] * self.nodes.shape[1]
        json_['num_cells'] = self._cell_trees['node'][2].shape[0]
        return json_

    def get_lines(self):
        '''
        Returns an array of lengths, and a list of line arrays.
        The first array sequentially indexes the second array.
        When the second array is split up using the first array
        and the resulting lines are drawn, you should end up with a picture of
        the grid.
        '''
        hor_lines = np.dstack((self.node_lon[:], self.node_lat[:])).astype(np.float32, copy=False)
        ver_lines = hor_lines.transpose((1, 0, 2)).astype(np.float32, copy=True)
        hor_lens = hor_lines.shape[1] * np.ones(hor_lines.shape[0], dtype=np.int32)
        ver_lens = ver_lines.shape[1] * np.ones(ver_lines.shape[0], dtype=np.int32)
        lens = np.concatenate((hor_lens, ver_lens))
        return (lens, [hor_lines, ver_lines])


class Grid_R(gridded.grids.Grid_R, GnomeId):

    _schema = GridSchema

    @classmethod
    def new_from_dict(cls, dict_):
        rv = cls.from_netCDF(**dict_)
        return rv

    def get_nodes(self):
        return self.nodes.reshape(-1,2)

    def get_centers(self):
        return self.centers.reshape(-1,2)

    def get_cells(self):
        return np.concatenate(self.node_lon, self.node_lat)

    def get_lines(self):

        lon_lines = np.array([[(lon, self.node_lat[0]), (lon, self.node_lat[len(self.node_lat)/2]), (lon, self.node_lat[-1])] for lon in self.node_lon], dtype=np.float32)
        lat_lines = np.array([[(self.node_lon[0], lat), (self.node_lon[len(self.node_lon)/2], lat), (self.node_lon[-1], lat)] for lat in self.node_lat], dtype=np.float32)
        lon_lens = lon_lines.shape[1] * np.ones(lon_lines.shape[0], dtype=np.int32)
        lat_lens = lat_lines.shape[1] * np.ones(lat_lines.shape[0], dtype=np.int32)
        lens = np.concatenate((lon_lens, lat_lens))
        return (lens, [lon_lines, lat_lines])

class PyGrid(gridded.grids.Grid):

    @staticmethod
    def from_netCDF(*args, **kwargs):
        kwargs['_default_types'] = (('ugrid', Grid_U), ('sgrid', Grid_S), ('rgrid', Grid_R))

        return gridded.grids.Grid.from_netCDF(*args, **kwargs)

    @staticmethod
    def new_from_dict(dict_):
        return PyGrid.from_netCDF(**dict_)

    @staticmethod
    def _get_grid_type(*args, **kwargs):
        kwargs['_default_types'] = (('ugrid', Grid_U), ('sgrid', Grid_S), ('rgrid', Grid_R))

        return gridded.grids.Grid._get_grid_type(*args, **kwargs)

class Depth(gridded.depth.Depth):
    @staticmethod
    def from_netCDF(*args, **kwargs):
        kwargs['_default_types'] = (('level', L_Depth), ('sigma', S_Depth), ('surface', DepthBase))

        return gridded.depth.Depth.from_netCDF(*args, **kwargs)

    @staticmethod
    def _get_depth_type(*args, **kwargs):
        kwargs['_default_types'] = (('level', L_Depth), ('sigma', S_Depth), ('surface', DepthBase))

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

    def __init__(self, extrapolate=False, *args, **kwargs):
        self.extrapolate = extrapolate
        super(Variable, self).__init__(*args, **kwargs)

    def at(self, *args, **kwargs):
        if ('extrapolate' not in kwargs):
            kwargs['extrapolate'] = self.extrapolate
        return super(Variable, self).at(*args, **kwargs)

    @classmethod
    def new_from_dict(cls, dict_):
        if 'data' not in dict_:
            return cls.from_netCDF(**dict_)

        return super(Variable, cls).new_from_dict(dict_)


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

    _schema = DepthSchema

    _default_component_types = copy.deepcopy(gridded.depth.S_Depth
                                             ._default_component_types)
    _default_component_types.update({'time': Time,
                                     'grid': PyGrid,
                                     'variable': Variable})
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
            raw_u = (raw_u[:, 0:-1, :] + raw_u[:, 1:, :]) / 2
            raw_v = (raw_v[:, :, 0:-1] + raw_v[:, :, 1:]) / 2

        raw_u = raw_u.reshape(raw_u.shape[0], -1)
        raw_v = raw_v.reshape(raw_v.shape[0], -1)
        r = np.stack((raw_u, raw_v))

        return np.ascontiguousarray(r, np.float32)

    def get_metadata(self):
        json_ = {}
        json_['data_location'] = self.grid.infer_location(self.variables[0].data)
        return json_

    @classmethod
    def new_from_dict(cls, dict_):
        rv = cls.from_netCDF(**dict_)
        return rv
