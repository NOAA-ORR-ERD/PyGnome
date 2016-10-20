import movers
import numpy as np
import datetime
import copy
from gnome import basic_types
from gnome.utilities import serializable, rand
from gnome.utilities.projections import FlatEarthProjection
from gnome.environment import GridWind
from gnome.basic_types import oil_status
from gnome.basic_types import (world_point,
                               world_point_type,
                               spill_type,
                               status_code_type)


class PyWindMover(movers.PyMover, serializable.Serializable):

    _state = copy.deepcopy(movers.Mover._state)
    _state.add(update=['uncertain_duration', 'uncertain_time_delay'],
               save=['uncertain_duration', 'uncertain_time_delay'])

    _ref_as = 'py_wind_movers'

    def __init__(self,
                 wind=None,
                 filename=None,
                 extrapolate=False,
                 time_offset=0,
                 uncertain_duration=3,
                 uncertain_time_delay=0,
                 uncertain_speed_scale=2.,
                 uncertain_angle_scale=0.4,
                 default_num_method='Trapezoid',
                 **kwargs):
        """
        Uses super to call CyMover base class __init__

        :param wind: wind object -- provides the wind time series for the mover

        Remaining kwargs are passed onto WindMoversBase __init__ using super.
        See Mover documentation for remaining valid kwargs.

        .. note:: Can be initialized with wind=None; however, wind must be
            set before running. If wind is not None, toggle make_default_refs
            to False since user provided a valid Wind and does not wish to
            use the default from the Model.
        """
        movers.PyMover.__init__(self,
                                default_num_method=default_num_method)
        self._wind = wind
        self.make_default_refs = False

        self.filename = filename
        self.extrapolate = extrapolate
        self.uncertain_duration = uncertain_duration
        self.uncertain_time_delay = uncertain_time_delay
        self.uncertain_speed_scale = uncertain_speed_scale

        # also sets self._uncertain_angle_units
        self.uncertain_angle_scale = uncertain_angle_scale

        self.array_types.update({'windages',
                                 'windage_range',
                                 'windage_persist'})
        # set optional attributes

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
                    default_num_method='Trapezoid',
                    **kwargs):

        wind = GridWind.from_netCDF(filename, **kwargs)
        return cls(wind=wind,
                   filename=filename,
                   extrapolate=extrapolate,
                   time_offset=time_offset,
                   current_scale=current_scale,
                   uncertain_along=uncertain_along,
                   uncertain_across=uncertain_across,
                   uncertain_cross=uncertain_cross,
                   default_num_method=default_num_method)

    @property
    def wind(self):
        return self._wind

    @wind.setter
    def wind(self, value):
        self._wind = value

    def prepare_for_model_step(self, sc, time_step, model_time_datetime):
        """
        Call base class method using super
        Also updates windage for this timestep

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of model as a date time object
        """
        super(PyWindMover, self).prepare_for_model_step(sc, time_step,
                                                           model_time_datetime)

        # if no particles released, then no need for windage
        # TODO: revisit this since sc.num_released shouldn't be None
        if sc.num_released is None  or sc.num_released == 0:
            return

        rand.random_with_persistance(sc['windage_range'][:, 0],
                                     sc['windage_range'][:, 1],
                                     sc['windages'],
                                     sc['windage_persist'],
                                     time_step)

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

        deltas = method(sc, time_step, model_time_datetime, pos, self.wind)
        deltas[:, 0] *= sc['windages']
        deltas[:, 1] *= sc['windages']

        deltas = FlatEarthProjection.meters_to_lonlat(deltas, positions)
        deltas[status] = (0, 0, 0)
        return deltas
