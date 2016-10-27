"""
grid for wind or current data
"""

import datetime
import copy
import os

import numpy as np

from colander import (SchemaNode, drop, Float, String)

from gnome.cy_gnome.cy_grid_curv import CyTimeGridWindCurv
from gnome.cy_gnome.cy_grid_rect import CyTimeGridWindRect
from gnome.utilities.time_utils import date_to_sec
from gnome.utilities.serializable import Serializable, Field
from gnome.persist import validators, base_schema

from .environment import Environment

import pyugrid
import pysgrid
import zipfile
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
        self_ts = None
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


class GridMeta(type):
    
    def __call__(cls, *args, **kwargs):
        if 'faces' in kwargs:
            cls = PyGrid_U
        else:
            cls = PyGrid_S
        return type.__call__(cls, *args, **kwargs)



class PyGridSchema(base_schema.ObjType):
    filename = SchemaNode(String())


class PyGrid(Serializable):

    __metaclass__ = GridMeta

    _def_count = 0

    _update = ['filename']

    # used to create new obj or as readonly parameter
    _create = ['filename']
    _create.extend(_update)

    _state = copy.deepcopy(Serializable._state)
    _state.add(save=_create, update=_update)
    _schema = PyGridSchema

    def __init__(self,
                 filename=None,
                 *args,
                 **kwargs):
        super(PyGrid, self).__init__(**kwargs)
        self.name = self.name + '_' + str(type(self)._def_count)
        self.filename = filename
        type(self)._def_count += 1

    @classmethod
    def from_netCDF(cls, filename, dataset=None, grid_topology=None, *args, **kwargs):
        gt = grid_topology
        gf = dataset
        if gf is None:
            gf = _get_dataset(filename)
        if gt is None:
            gt = PyGrid._find_topology(filename)
        if gt is None:
            gt = PyGrid._gen_topology(filename)
            if 'nodes' not in gt:
                if 'node_lon' not in gt and 'node_lat' not in gt:
                    raise ValueError('Nodes must be specified with either the "nodes" or "node_lon" and "node_lat" keys')
                if 'node_lon' not in kwargs:
                    kwargs['node_lon'] = gf[gt['node_lon']]
                if 'node_lat' not in kwargs:
                    kwargs['node_lat'] = gf[gt['node_lat']]
            else:
                if 'nodes' not in kwargs:
                    kwargs['nodes'] = gf[gt['nodes']]
            if 'faces' in gt and gf[gt['faces']]:
                # Definitely UGrid
                if 'faces' not in kwargs:
                    kwargs['faces'] = gf[gt['faces']]
                if kwargs['faces'].shape[0] == 3:
                    kwargs['faces'] = np.ascontiguousarray(np.array(kwargs['faces']).T - 1)  # special case for fortran-style, index from 1 faces files
                return cls(filename, *args, **kwargs)
            else:
                # SGrid
                if kwargs['node_lon'] is None:
                    kwargs['node_lon'] = kwargs['nodes'][:, 0]
                if kwargs['node_lat'] is None:
                    kwargs['node_lat'] = kwargs['nodes'][:, 1]
                for k in ('center_lon', 'center_lat', 'edge1_lon', 'edge1_lat', 'edge2_lon', 'edge2_lat'):
                    if k not in kwargs:
                        if k in gt:
                            kwargs[k] = gf[gt[k]]
                        else:
                            kwargs[k] = None
                return cls(filename, *args, **kwargs)
        else:
            # toplogy variable does exist! Read to determine correct course of action
            raise RuntimeError()

    @property
    def shape(self):
        return self.node_lon.shape

    def __eq__(self, o):
        for n in ('nodes', 'faces'):
            if hasattr(self, n) and hasattr(o, n):
                s = getattr(self, n)
                s2 = getattr(o, n)
                if s.shape != s2.shape or np.any(s != s2):
                    return False
        return True

    def _write_grid_to_file(self, pth):
        self.save_as_netCDF(pth)
        

    def save(self, saveloc, references=None, name=None):
        '''
        Write Wind timeseries to file or to zip,
        then call save method using super
        '''
        name = (name, 'Wind.json')[name is None]
        ts_name = os.path.splitext(name)[0] + '_data.GRD'
        
        if zipfile.is_zipfile(saveloc):
            self._write_grid_to_zip(saveloc, ts_name)
            self._filename = ts_name
        else:
            datafile = os.path.join(saveloc, ts_name)
            self._write_grid_to_file(datafile)
            self._filename = datafile
        return super(PyGrid, self).save(saveloc, references, name)
    
    @staticmethod
    def _find_topology(filename,
                       dataset=None):
        gf = dataset
        if gf is None:
            gf = _get_dataset(filename)
        if 'grid_topology' in gf.variables.keys():
            return gf['grid_topology']
        else:
            return None

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
    
PyGrid_U = type('PyGrid_U', (PyGrid, pyugrid.UGrid), {})
PyGrid_S = type('PyGrid_S', (PyGrid, pysgrid.SGrid), {})
