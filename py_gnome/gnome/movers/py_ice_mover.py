import movers
import numpy as np
import datetime
from gnome import basic_types
from gnome.utilities.projections import FlatEarthProjection
from gnome.basic_types import oil_status
from gnome.basic_types import (world_point,
                               world_point_type,
                               spill_type,
                               status_code_type)


class PyIceMover(movers.Mover):

    def __init__(self,
                 filename=None,
                 ice_prop=None,
                 water_prop=None,
                 air_prop=None,
                 extrapolate=False,
                 time_offset=0,
                 current_scale=1,
                 uncertain_duration=24 * 3600,
                 uncertain_time_delay=0,
                 uncertain_along=.5,
                 uncertain_across=.25,
                 uncertain_cross=.25,
                 num_method=0):
        self.ice_prop = ice_prop
        self.water_prop = water_prop
        self.air_prop = air_prop
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

    def get_move(self, sc, time_step, model_time):
        """
        Compute the move in (long,lat,z) space. It returns the delta move
        for each element of the spill as a numpy array of size
        (number_elements X 3) and dtype = gnome.basic_types.world_point_type

        Base class returns an array of numpy.nan for delta to indicate the
        get_move is not implemented yet.

        Each class derived from Mover object must implement it's own get_move

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time: current model time as datetime object

        All movers must implement get_move() since that's what the model calls
        """
        status = sc['status_codes'] != oil_status.in_water
        positions = sc['positions'][:, 0:2] + [180, 0]


        deltas = np.zeros_like(sc['positions'], dtype=np.float64)
        deltas[:, 0:2] = water_vel * time_step
        deltas = FlatEarthProjection.meters_to_lonlat(deltas, sc['positions'])
        deltas[status] = (0, 0, 0)
        pass
        return deltas

    def get_ice_coverage(self, positions, time, indices=None, alphas=None):
        ice_coverage = self.ice_field.interpolate_var(
            positions, time, self.ice_field.coverage, indices=indices, alphas=alphas)
        return ice_coverage
