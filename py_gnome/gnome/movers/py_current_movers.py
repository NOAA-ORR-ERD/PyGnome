import movers
import numpy as np
import datetime
import copy
import pytest
from gnome import basic_types
from gnome.environment import GridCurrent, GridVectorPropSchema
from gnome.environment.grid import PyGrid_U
from gnome.utilities import serializable
from gnome.utilities.projections import FlatEarthProjection
from gnome.basic_types import oil_status
from gnome.basic_types import (world_point,
                               world_point_type,
                               spill_type,
                               status_code_type)
from gnome.persist import base_schema
from colander import SchemaNode, Float, Boolean, Sequence, MappingSchema, drop, String, OneOf, SequenceSchema, TupleSchema, DateTime, Bool


class PyCurrentMoverSchema(base_schema.ObjType):
    filename = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String())], missing=drop)
    current_scale = SchemaNode(Float(), missing=drop)
    extrapolate = SchemaNode(Bool(), missing=drop)
    time_offset = SchemaNode(Float(), missing=drop)
    current = GridVectorPropSchema(missing=drop)
    data_start_time = SchemaNode(DateTime(), missing=drop)
    data_end_time = SchemaNode(DateTime(), missing=drop)


class PyCurrentMover(movers.PyMover, serializable.Serializable):

    _state = copy.deepcopy(movers.PyMover._state)

    _state.add_field([serializable.Field('filename',
                                         save=True, read=True, isdatafile=True,
                                         test_for_eq=False),
                      serializable.Field('current', read=True, save_reference=True),
                      serializable.Field('data_start_time', read=True),
                      serializable.Field('data_end_time', read=True),
                      ])
    _state.add(update=['uncertain_duration', 'uncertain_time_delay'],
               save=['uncertain_duration', 'uncertain_time_delay'])
    _schema = PyCurrentMoverSchema

    _ref_as = 'py_current_movers'

    _req_refs = {'current': GridCurrent}
    _def_count = 0

    def __init__(self,
                 filename=None,
                 current=None,
                 name=None,
                 extrapolate=False,
                 time_offset=0,
                 current_scale=1,
                 uncertain_duration=24 * 3600,
                 uncertain_time_delay=0,
                 uncertain_along=.5,
                 uncertain_across=.25,
                 uncertain_cross=.25,
                 default_num_method='Trapezoid',
                 **kwargs
                 ):
        self.filename = filename
        self.current = current
        if self.current is None:
            if filename is None:
                raise ValueError("must provide a filename or current object")
            else:
                self.current = GridCurrent.from_netCDF(filename=self.filename, **kwargs)
        if name is None:
            name = self.__class__.__name__ + str(self.__class__._def_count)
            self.__class__._def_count += 1
        self.extrapolate = extrapolate
        self.current_scale = current_scale
        self.uncertain_along = uncertain_along
        self.uncertain_across = uncertain_across
        self.uncertain_duration = uncertain_duration
        self.uncertain_time_delay = uncertain_time_delay
        self.model_time = 0
        self.positions = np.zeros((0, 3), dtype=world_point_type)
        self.delta = np.zeros((0, 3), dtype=world_point_type)
        self.status_codes = np.zeros((0, 1), dtype=status_code_type)
        if self.current.time is None or len(self.current.time.data) == 1:
            self.extrapolate = True

        # either a 1, or 2 depending on whether spill is certain or not
        self.spill_type = 0

        super(PyCurrentMover, self).__init__(default_num_method=default_num_method,
                                             **kwargs)

    def _attach_default_refs(self, ref_dict):
        pass
        return serializable.Serializable._attach_default_refs(self, ref_dict)

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    name=None,
                    extrapolate=False,
                    time_offset=0,
                    current_scale=1,
                    uncertain_duration=24 * 3600,
                    uncertain_time_delay=0,
                    uncertain_along=.5,
                    uncertain_across=.25,
                    uncertain_cross=.25,
                    **kwargs):
        current = GridCurrent.from_netCDF(filename, **kwargs)
        if name is None:
            name = cls.__name__ + str(cls._def_count)
            cls._def_count += 1
        return cls(name=name,
                   current=current,
                   filename=filename,
                   extrapolate=extrapolate,
                   time_offset=time_offset,
                   current_scale=current_scale,
                   uncertain_along=uncertain_along,
                   uncertain_across=uncertain_across,
                   uncertain_cross=uncertain_cross,
                   **kwargs)

    @property
    def data_start_time(self):
        return self.current.time.min_time

    @property
    def data_end_time(self):
        return self.current.time.max_time

    @property
    def is_data_on_cells(self):
        return self.current.grid.infer_location(self.current.u.data) != 'node'

    def get_grid_data(self):
        """
            The main function for getting grid data from the mover
        """
        if isinstance(self.current.grid, PyGrid_U):
            return self.current.grid.nodes[self.current.grid.faces[:]]
        else:
            lons = self.current.grid.node_lon
            lats = self.current.grid.node_lat
            return np.column_stack((lons.reshape(-1), lats.reshape(-1)))

    def get_center_points(self):
        if hasattr(self.current.grid, 'center_lon') and self.current.grid.center_lon is not None:
            lons = self.current.grid.center_lon
            lats = self.current.grid.center_lat
            return np.column_stack((lons.reshape(-1), lats.reshape(-1)))
        else:
            lons = self.current.grid.node_lon
            lats = self.current.grid.node_lat
            if len(lons.shape) == 1: #ugrid
                triangles = self.current.grid.nodes[self.current.grid.faces[:]]
                centroids = np.zeros((self.current.grid.faces.shape[0], 2))
                centroids[:, 0] = np.sum(triangles[:, :, 0], axis=1) / 3
                centroids[:, 1] = np.sum(triangles[:, :, 1], axis=1) / 3

            else:
                c_lons = (lons[0:-1, :] + lons[1:, :]) /2
                c_lats = (lats[:, 0:-1] + lats[:, 1:]) /2
                centroids = np.column_stack((c_lons.reshape(-1), c_lats.reshape(-1)))
            return centroids


    def get_scaled_velocities(self, time):
        """
        :param model_time=0:
        """
        current = self.current
        lons = current.grid.node_lon
        lats = current.grid.node_lat

        #GridCurrent.at needs Nx3 points [lon, lat, z] and a time T
        points = np.column_stack((lons.reshape(-1), lats.reshape(-1), np.zeros_like(current.grid.node_lon.reshape(-1))))
        vels = current.at(points, time)

        return vels

    def get_move(self, sc, time_step, model_time_datetime, num_method=None):
        """
        Compute the move in (long,lat,z) space. It returns the delta move
        for each element of the spill as a numpy array of size
        (number_elements X 3) and dtype = gnome.basic_types.world_point_type

        Base class returns an array of numpy.nan for delta to indicate the
        get_move is not implemented yet.

        Each class derived from Mover object must implement it's own get_move

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current model time as datetime object

        All movers must implement get_move() since that's what the model calls
        """
        method = None
        if num_method is None:
            method = self.num_methods[self.default_num_method]
        else:
            method = self.num_method[num_method]

        status = sc['status_codes'] != oil_status.in_water
        positions = sc['positions']
        pos = positions[:]

        res = method(sc, time_step, model_time_datetime, pos, self.current)
        if res.shape[1] == 2:
            deltas = np.zeros_like(positions)
            deltas[:, 0:2] = res
        else:
            deltas = res

        deltas = FlatEarthProjection.meters_to_lonlat(deltas, positions)
        deltas[status] = (0, 0, 0)
        return deltas
