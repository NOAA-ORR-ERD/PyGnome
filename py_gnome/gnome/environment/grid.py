"""
grid for wind or current data
"""

import datetime
import copy

import numpy as np

from colander import (SchemaNode, drop, Float)

from gnome.cy_gnome.cy_grid_curv import CyTimeGridWindCurv
from gnome.cy_gnome.cy_grid_rect import CyTimeGridWindRect
from gnome.utilities.time_utils import date_to_sec
from gnome.utilities.serializable import Serializable, Field
from gnome.persist import validators, base_schema

from .environment import Environment

import pyugrid
import pysgrid
import netCDF4 as nc4
from gnome.utilities.file_tools.data_helpers import _get_dataset

class GridSchema(base_schema.ObjType):
    name = 'grid'
    grid_type = SchemaNode(Float(), missing=drop)


class Grid(Environment, Serializable):
    '''
    Defines a grid for a current or wind
    '''

    _update = []

    # used to create new obj or as readonly parameter
    _create = []
    _create.extend(_update)

    _state = copy.deepcopy(Environment._state)
    _state.add(save=_create, update=_update)
    _schema = GridSchema


    def __init__(self, filename, topology_file=None, grid_type=1,
                 extrapolate=False, time_offset=0,
                 **kwargs):
        """
        Initializes a grid object from a file and a grid type

        maybe allow a grid to be passed in eventually, otherwise filename required

        All other keywords are optional. Optional parameters (kwargs):

        :param grid_type: default is 1 - regular grid (eventually figure this out from file)

        """
        self._grid_type = grid_type

        self.filename = filename
        self.topology_file = topology_file

        if self._grid_type == 1:
            self.grid = CyTimeGridWindRect(filename)
        elif self._grid_type == 2:
            self.grid = CyTimeGridWindCurv(filename, topology_file)
        else:
            raise Exception('grid_type not implemented ')

        self.grid.load_data(filename, topology_file)

        super(Grid, self).__init__(**kwargs)

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'timeseries={1}'
                ')').format(self, self_ts)

    def __str__(self):
        return ("Grid ( "
                "grid_type='curvilinear')")

    @property
    def grid_type(self):
        return self._grid_type

    @grid_type.setter
    def grid_type(self, value):
        """
        probably will figure out from the file
        """
        # may want a check on value
        self._grid_type = value


    extrapolate = property(lambda self: self.grid.extrapolate,
                           lambda self, val: setattr(self.grid,
                                                     'extrapolate',
                                                     val))

    time_offset = property(lambda self: self.grid.time_offset / 3600.,
                           lambda self, val: setattr(self.grid,
                                                     'time_offset',
                                                     val * 3600.))

    def prepare_for_model_run(self, model_time):
        """
        Not sure we need to do anything here
        """

        pass


    def prepare_for_model_step(self, model_time):
        """
        Make sure we have the right data loaded
        """
        model_time = date_to_sec(model_time)
        self.grid.set_interval(model_time)

    def get_value(self, time, location):
        '''
        Return the value at specified time and location. 

        '''
        data = self.grid.get_value(time, location)

        return data

    def get_values(self, model_time, positions, velocities):
        '''
        Return the values for the given positions

        '''
        data = self.grid.get_values(model_time, positions, velocities)

        return data

    def serialize(self, json_='webapi'):

        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()

        serial = schema.serialize(toserial)

        return serial

    @classmethod
    def deserialize(cls, json_):

        schema = cls._schema()

        _to_dict = schema.deserialize(json_)

        return _to_dict


class PyGrid(Serializable):
    
    def __init__(self,
                 filename,
                 grid_topology=None,
                 dataset=None,
                 **kwargs):
        if grid_topology is None:
            grid_topology = self._gen_topology(filename, dataset)
        self._grid = self._init_grid(filename, grid_topology, dataset)
        super(PyGrid, self).__init__(**kwargs)
    
    def __new__(self):
        pass
    
    def __getattribute__(self, name):
        if name != '_grid':
            return self._grid.__getattribute__(name)
        else:
            return self._grid
    
    @staticmethod
    def _init_grid(filename,
                   grid_topology=None,
                   dataset=None,):
        gt = grid_topology
        gf = dataset
        if gf is None:
            gf = _get_dataset(filename)
        grid = None
        if gt is None:
            try:
                grid = pyugrid.UGrid.from_nc_dataset(gf)
            except:
                pass
            try:
                grid = pysgrid.SGrid.load_grid(gf)
            except:
                gt = PyGrid._gen_topology(filename)
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
                # UGrid
                faces = gf[gt['faces']]
                if faces.shape[0] == 3:
                    faces = np.ascontiguousarray(np.array(faces).T - 1)
                if nodes is None:
                    nodes = np.column_stack((node_lon, node_lat))
                grid = pyugrid.UGrid(nodes=nodes, faces=faces)
            else:
                # SGrid
                center_lon = center_lat = edge1_lon = edge1_lat = edge2_lon = edge2_lat = None
                if node_lon is None:
                    node_lon = nodes[:, 0]
                if node_lat is None:
                    node_lat = nodes[:, 1]
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
                grid = pysgrid.SGrid(node_lon=node_lon,
                                     node_lat=node_lat,
                                     center_lon=center_lon,
                                     center_lat=center_lat,
                                     edge1_lon=edge1_lon,
                                     edge1_lat=edge1_lat,
                                     edge2_lon=edge2_lon,
                                     edge2_lat=edge2_lat)
        return grid

    @staticmethod
    def _gen_topology(filename,
                      dataset=None):
        '''
        Function to create the correct default topology if it is not provided
    
        :param filename: Name of file that will be searched for variables
        :return: List of default variable names, or None if none are found
        '''
        gf = dataset
        if gf is None:
            gf = _get_dataset(filename)
        gt = {}
        node_coord_names = [['node_lon', 'node_lat'], ['lon', 'lat'], ['lon_psi', 'lat_psi']]
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
            # UGRID
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

    def save(self, saveloc, references=None, name=None):
        '''
        Write Wind timeseries to file or to zip,
        then call save method using super
        '''
        name = (name, 'Wind.json')[name is None]
        ts_name = os.path.splitext(name)[0] + '_data.WND'

        if zipfile.is_zipfile(saveloc):
            self._write_timeseries_to_zip(saveloc, ts_name)
            self._filename = ts_name
        else:
            datafile = os.path.join(saveloc, ts_name)
            self._write_timeseries_to_file(datafile)
            self._filename = datafile
        return super(Wind, self).save(saveloc, references, name)

class PyStructuredGrid(pysgrid.SGrid, Serializable):
    
    def __init__(self, *args, **kwargs):
        super(PyStructuredGrid, self).__init__(*args, **kwargs)
