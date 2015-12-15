import movers
import numpy as np
import datetime
from gnome import basic_types
from gnome.utilities.projections import FlatEarthProjection
from gnome.environment.vector_field import TriVectorField
from gnome.basic_types import oil_status
from gnome.basic_types import (world_point,
                               world_point_type,
                               spill_type,
                               status_code_type)

class UGridCurrentMover(movers.Mover):

    def __init__(self,
                 grid=None,
                 extrapolate=False,
                 time_offset=0,
                 current_scale=1,
                 uncertain_duration=24*3600,
                 uncertain_time_delay=0,
                 uncertain_along=.5,
                 uncertain_across=.25,
                 uncertain_cross=.25,
                 num_method=0):
        self.grid = grid
        self.current_scale = current_scale
        self.uncertain_along = uncertain_along
        self.uncertain_across = uncertain_across
        self.num_method = num_method
        self.uncertain_duration = uncertain_duration
        self.uncertain_time_delay = uncertain_time_delay
        self.model_time = 0
        self.positions = np.zeros((0, 3), dtype=world_point_type)
        self.delta = np.zeros((0, 3), dtype=world_point_type)
        self.status_codes = np.zeros((0, 1), dtype=status_code_type)

        # either a 1, or 2 depending on whether spill is certain or not
        self.spill_type = 0

        movers.Mover.__init__(self)

    def get_move(self, sc, time_step, model_time_datetime):
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
        status = sc['status_codes'] != oil_status.in_water
        positions = sc['positions']

        vels = self.grid.interpolated_velocities(model_time_datetime, positions[:,0:2])
        deltas = np.zeros_like(positions)
        deltas[:] = 0.
        deltas[:,0:2] = vels * time_step
        deltas = FlatEarthProjection.meters_to_lonlat(deltas, positions)
        deltas[status] = (0,0,0)
        pass
        return deltas