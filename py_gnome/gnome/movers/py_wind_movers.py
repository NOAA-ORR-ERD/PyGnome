import movers

import numpy as np

from colander import (SchemaNode,
                      Bool, Float, String, Sequence, drop)

from gnome.basic_types import oil_status

from gnome.utilities import rand
from gnome.utilities.projections import FlatEarthProjection

from gnome.environment import GridWind

from gnome.persist.base_schema import ObjTypeSchema
from gnome.persist.validators import convertible_to_seconds
from gnome.persist.extend_colander import LocalDateTime, FilenameSchema
from gnome.persist.base_schema import GeneralGnomeObjectSchema
from gnome.environment.gridded_objects_base import VectorVariableSchema


class PyWindMoverSchema(ObjTypeSchema):
    wind = GeneralGnomeObjectSchema(
        acceptable_schemas=[VectorVariableSchema, GridWind._schema],
        save=True, update=True, save_reference=True
    )
    filename = FilenameSchema(
        missing=drop, save=True, read_only=True, isdatafile=True
    )
    wind_scale = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    extrapolation_is_allowed = SchemaNode(
        Bool(), missing=drop, save=True, update=True
    )
    time_offset = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    on = SchemaNode(
        Bool(), missing=drop, save=True, update=True
    )
    active_start = SchemaNode(
        LocalDateTime(), missing=drop,
        validator=convertible_to_seconds,
        save=True, update=True
    )
    active_stop = SchemaNode(
        LocalDateTime(), missing=drop,
        validator=convertible_to_seconds,
        save=True, update=True
    )
    data_start = SchemaNode(
        LocalDateTime(), validator=convertible_to_seconds, read_only=True
    )
    data_stop = SchemaNode(
        LocalDateTime(), validator=convertible_to_seconds, read_only=True
    )


class PyWindMover(movers.PyMover):
    _schema = PyWindMoverSchema

    _ref_as = 'py_wind_movers'

    _req_refs = {'wind': GridWind}

    def __init__(self,
                 filename=None,
                 wind=None,
                 time_offset=0,
                 uncertain_duration=3,
                 uncertain_time_delay=0,
                 uncertain_speed_scale=2.,
                 uncertain_angle_scale=0.4,
                 wind_scale=1,
                 default_num_method='RK2',
                 extrapolation_is_allowed=False,
                 **kwargs):
        """
        Initialize a PyWindMover

        :param filename: absolute or relative path to the data file(s):
                         could be a string or list of strings in the
                         case of a multi-file dataset
        :param wind: Environment object representing wind to be
                        used. If this is not specified, a GridWind object
                        will attempt to be instantiated from the file
        :param active_start: datetime when the mover should be active
        :param active_stop: datetime after which the mover should be inactive
        :param wind_scale: Value to scale wind data
        :param uncertain_duration: how often does a given uncertain element
                                   get reset
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param uncertain_cross: Scale for uncertainty perpendicular to the flow
        :param uncertain_along: Scale for uncertainty parallel to the flow
        :param time_offset: Time zone shift if data is in GMT
        :param num_method: Numerical method for calculating movement delta.
                           Choices:('Euler', 'RK2', 'RK4')
                           Default: RK2

        """
        self.wind = wind
        self.make_default_refs = False

        self.filename = filename

        if self.wind is None:
            if filename is None:
                raise ValueError("must provide a filename or wind object")
            else:
                self.wind = GridWind.from_netCDF(filename=self.filename,
                                                 **kwargs)

        self.extrapolation_is_allowed = extrapolation_is_allowed
        self.uncertain_duration = uncertain_duration
        self.uncertain_time_delay = uncertain_time_delay
        self.uncertain_speed_scale = uncertain_speed_scale
        self.wind_scale = wind_scale
        self.time_offset = time_offset

        # also sets self._uncertain_angle_units
        self.uncertain_angle_scale = uncertain_angle_scale

        (super(PyWindMover, self)
         .__init__(default_num_method=default_num_method, **kwargs))

        self.array_types.update({'windages',
                                 'windage_range',
                                 'windage_persist'})

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    time_offset=0,
                    wind_scale=1,
                    uncertain_duration=24 * 3600,
                    uncertain_time_delay=0,
                    uncertain_along=.5,
                    uncertain_across=.25,
                    uncertain_cross=.25,
                    default_num_method='RK2',
                    **kwargs):

        wind = GridWind.from_netCDF(filename, **kwargs)

        return cls(wind=wind,
                   filename=filename,
                   time_offset=time_offset,
                   wind_scale=wind_scale,
                   uncertain_along=uncertain_along,
                   uncertain_across=uncertain_across,
                   uncertain_cross=uncertain_cross,
                   default_num_method=default_num_method)

    @property
    def data_start(self):
        return self.wind.data_start

    @property
    def data_stop(self):
        return self.wind.data_stop

    @property
    def is_data_on_cells(self):
        return self.wind.grid.infer_location(self.wind.u.data) != 'node'

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
        if sc.num_released is None or sc.num_released == 0:
            return

        if self.active:
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
        positions = sc['positions']

        if self.active and len(positions) > 0:
            status = sc['status_codes'] != oil_status.in_water
            pos = positions[:]

            deltas = self.delta_method(num_method)(sc, time_step, model_time_datetime, pos, self.wind)
            deltas[:, 0] *= sc['windages'] * self.wind_scale
            deltas[:, 1] *= sc['windages'] * self.wind_scale

            deltas = FlatEarthProjection.meters_to_lonlat(deltas, positions)
            deltas[status] = (0, 0, 0)
        else:
            deltas = np.zeros_like(positions)

        return deltas
