from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from . import movers

import numpy as np

from colander import (SchemaNode,
                      Bool, Float, String, Sequence, drop)

from gnome.basic_types import oil_status
from gnome.array_types import gat

from gnome.utilities import rand
from gnome.utilities.projections import FlatEarthProjection

from gnome.environment import GridWind

from gnome.movers.movers import TimeRangeSchema

from gnome.persist.base_schema import ObjTypeSchema
from gnome.persist.validators import convertible_to_seconds
from gnome.persist.extend_colander import LocalDateTime, FilenameSchema
from gnome.persist.base_schema import GeneralGnomeObjectSchema
from gnome.environment.gridded_objects_base import Grid_U, VectorVariableSchema


class PyWindMoverSchema(ObjTypeSchema):
    wind = GeneralGnomeObjectSchema(save=True, update=True,
                                    save_reference=True,
                                    acceptable_schemas=[VectorVariableSchema,
                                                        GridWind._schema])
    filename = FilenameSchema(save=True, update=False, isdatafile=True,
                              missing=drop)
    scale_value = SchemaNode(Float(), save=True, update=True, missing=drop)
    time_offset = SchemaNode(Float(), save=True, update=True, missing=drop)
    on = SchemaNode(Bool(), save=True, update=True, missing=drop)
    active_range = TimeRangeSchema()
    data_start = SchemaNode(LocalDateTime(), read_only=True,
                            validator=convertible_to_seconds)
    data_stop = SchemaNode(LocalDateTime(), read_only=True,
                           validator=convertible_to_seconds)


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
                 scale_value=1,
                 default_num_method='RK2',
                 **kwargs):
        """
        Initialize a PyWindMover

        :param filename: absolute or relative path to the data file(s):
                         could be a string or list of strings in the
                         case of a multi-file dataset
        :param wind: Environment object representing wind to be
                        used. If this is not specified, a GridWind object
                        will attempt to be instantiated from the file

        :param active_range: Range of datetimes for when the mover should be
                             active
        :type active_range: 2-tuple of datetimes

        :param scale_value: Value to scale wind data
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

        (super(PyWindMover, self).__init__(default_num_method=default_num_method, **kwargs))
        self.wind = wind
        self.make_default_refs = False

        self.filename = filename

        if self.wind is None:
            if filename is None:
                raise ValueError("must provide a filename or wind object")
            else:
                self.wind = GridWind.from_netCDF(filename=self.filename,
                                                 **kwargs)

        self.uncertain_duration = uncertain_duration
        self.uncertain_time_delay = uncertain_time_delay
        self.uncertain_speed_scale = uncertain_speed_scale
        self.scale_value = scale_value
        self.time_offset = time_offset

        # also sets self._uncertain_angle_units
        self.uncertain_angle_scale = uncertain_angle_scale

        self.array_types.update({'windages': gat('windages'),
                                 'windage_range': gat('windage_range'),
                                 'windage_persist': gat('windage_persist')})

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    time_offset=0,
                    scale_value=1,
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
                   scale_value=scale_value,
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

    def get_grid_data(self):
        """
            The main function for getting grid data from the mover
        """
        if isinstance(self.wind.grid, Grid_U):
            return self.wind.grid.nodes[self.wind.grid.faces[:]]
        else:
            lons = self.wind.grid.node_lon
            lats = self.wind.grid.node_lat

            return np.column_stack((lons.reshape(-1), lats.reshape(-1)))

    def get_center_points(self):
        if (hasattr(self.wind.grid, 'center_lon') and
                self.wind.grid.center_lon is not None):
            lons = self.wind.grid.center_lon
            lats = self.wind.grid.center_lat

            return np.column_stack((lons.reshape(-1), lats.reshape(-1)))
        else:
            lons = self.wind.grid.node_lon
            lats = self.wind.grid.node_lat

            if len(lons.shape) == 1:
                # we are ugrid
                triangles = self.wind.grid.nodes[self.wind.grid.faces[:]]
                centroids = np.zeros((self.wind.grid.faces.shape[0], 2))
                centroids[:, 0] = np.sum(triangles[:, :, 0], axis=1) / 3
                centroids[:, 1] = np.sum(triangles[:, :, 1], axis=1) / 3

            else:
                c_lons = (lons[0:-1, :] + lons[1:, :]) / 2
                c_lats = (lats[:, 0:-1] + lats[:, 1:]) / 2
                centroids = np.column_stack((c_lons.reshape(-1),
                                             c_lats.reshape(-1)))

            return centroids

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
            deltas[:, 0] *= sc['windages'] * self.scale_value
            deltas[:, 1] *= sc['windages'] * self.scale_value

            deltas = FlatEarthProjection.meters_to_lonlat(deltas, positions)
            deltas[status] = (0, 0, 0)
        else:
            deltas = np.zeros_like(positions)

        return deltas
