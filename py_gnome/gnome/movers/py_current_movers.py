import movers
import numpy as np
import datetime
import copy
from gnome import basic_types
from gnome.environment import GridCurrent, GridVectorPropSchema
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


class PyCurrentMover(movers.PyMover, serializable.Serializable):

    _state = copy.deepcopy(movers.PyMover._state)

    _state.add_field([serializable.Field('filename',
                                         save=True, read=True, isdatafile=True,
                                         test_for_eq=False),
                      serializable.Field('current', save=True, read=True, save_reference=True)])
    _state.add(update=['uncertain_duration', 'uncertain_time_delay'],
               save=['uncertain_duration', 'uncertain_time_delay'])
    _schema = PyCurrentMoverSchema

    _ref_as = 'py_current_movers'

    def __init__(self,
                 current=None,
                 filename=None,
                 extrapolate=False,
                 time_offset=0,
                 current_scale=1,
                 uncertain_duration=24 * 3600,
                 uncertain_time_delay=0,
                 uncertain_along=.5,
                 uncertain_across=.25,
                 uncertain_cross=.25,
                 default_num_method='Trapezoid'
                 ):
        self.current = current
        self.filename = filename
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

        # either a 1, or 2 depending on whether spill is certain or not
        self.spill_type = 0

        movers.PyMover.__init__(self,
                                default_num_method=default_num_method)

    @classmethod
    def from_netCDF(cls,
                    filename=None,
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
        return cls(current=current,
                   filename=filename,
                   extrapolate=extrapolate,
                   time_offset=time_offset,
                   current_scale=current_scale,
                   uncertain_along=uncertain_along,
                   uncertain_across=uncertain_across,
                   uncertain_cross=uncertain_cross,
                   **kwargs)


    def get_scaled_velocities(self, time):
        """
        :param model_time=0:
        """
        points = None
        vels = self.grid.interpolated_velocities(time, points)

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
