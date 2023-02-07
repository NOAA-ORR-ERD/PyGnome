'''
Movers using wind as the forcing function
'''

import os

import numpy as np

from colander import (SchemaNode, Bool, String, Float, drop)

from gnome.exceptions import ReferencedObjectNotSet

from gnome.basic_types import (world_point,
                               world_point_type,
                               velocity_rec)
from gnome.array_types import gat

from gnome.cy_gnome.cy_wind_mover import CyWindMover
from gnome.cy_gnome.cy_gridwind_mover import CyGridWindMover
from gnome.cy_gnome.cy_ice_wind_mover import CyIceWindMover

from gnome.utilities.time_utils import sec_to_datetime
from gnome.utilities.rand import random_with_persistance


from gnome.environment import Wind, WindSchema
from gnome.environment.wind import constant_wind
from gnome.movers import CyMover, ProcessSchema
from gnome.persist.base_schema import GeneralGnomeObjectSchema
from gnome.persist.extend_colander import FilenameSchema
from gnome.persist.validators import convertible_to_seconds
from gnome.persist.extend_colander import LocalDateTime
from gnome.utilities.inf_datetime import InfTime, MinusInfTime


class WindMoversBaseSchema(ProcessSchema):
    uncertain_duration = SchemaNode(Float(), save=True, update=True,
                                    missing=drop)
    uncertain_time_delay = SchemaNode(Float(), save=True, update=True,
                                      missing=drop)
    uncertain_speed_scale = SchemaNode(Float(), save=True, update=True,
                                       missing=drop)
    uncertain_angle_scale = SchemaNode(Float(), save=True, update=True,
                                       missing=drop)


class WindMoversBase(CyMover):

    _schema = WindMoversBaseSchema

    def __init__(self,
                 uncertain_duration=3,
                 uncertain_time_delay=0,
                 uncertain_speed_scale=2.,
                 uncertain_angle_scale=0.4,
                 **kwargs):
        """
        This is simply a base class for WindMover and c_GridWindMover for the
        common properties.

        The classes that inherit from this should define the self.mover object
        correctly so it has the required attributes.

        Input args with defaults:

        :param uncertain_duration: (seconds) the randomly generated uncertainty
            array gets recomputed based on 'uncertain_duration'
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param uncertain_speed_scale: Scale for uncertainty in wind speed
            non-dimensional number
        :param uncertain_angle_scale: Scale for uncertainty in wind direction.
            Assumes this is in radians

        It calls super in the __init__ method and passes in the optional
        parameters (kwargs)
        """
        super(WindMoversBase, self).__init__(**kwargs)

        self.uncertain_duration = uncertain_duration
        self.uncertain_time_delay = uncertain_time_delay
        self.uncertain_speed_scale = uncertain_speed_scale

        # also sets self._uncertain_angle_units
        self.uncertain_angle_scale = uncertain_angle_scale

        self.array_types.update({'windages': gat('windages'),
                                 'windage_range': gat('windage_range'),
                                 'windage_persist': gat('windage_persist')})

    # no conversion necessary - simply sets/gets the stored value
    uncertain_speed_scale = \
        property(lambda self: self.mover.uncertain_speed_scale,
                 lambda self, val: setattr(self.mover, 'uncertain_speed_scale',
                                           val))

    uncertain_angle_scale = \
        property(lambda self: self.mover.uncertain_angle_scale,
                 lambda self, val: setattr(self.mover, 'uncertain_angle_scale',
                                           val))

    def _seconds_to_hours(self, seconds):
        return seconds / 3600.0

    def _hours_to_seconds(self, hours):
        return hours * 3600.0

    @property
    def uncertain_duration(self):
        return self._seconds_to_hours(self.mover.uncertain_duration)

    @uncertain_duration.setter
    def uncertain_duration(self, val):
        self.mover.uncertain_duration = self._hours_to_seconds(val)

    @property
    def uncertain_time_delay(self):
        return self._seconds_to_hours(self.mover.uncertain_time_delay)

    @uncertain_time_delay.setter
    def uncertain_time_delay(self, val):
        self.mover.uncertain_time_delay = self._hours_to_seconds(val)

    def prepare_for_model_step(self, sc, time_step, model_time_datetime):
        """
        Call base class method using super
        Also updates windage for this timestep

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of model as a date time object
        """
        super(WindMoversBase, self).prepare_for_model_step(sc, time_step,
                                                           model_time_datetime)

        # if no particles released, then no need for windage
        # TODO: revisit this since sc.num_released shouldn't be None
        if sc.num_released is None or sc.num_released == 0:
            return

        if self.active:
            random_with_persistance(sc['windage_range'][:, 0],
                                    sc['windage_range'][:, 1],
                                    sc['windages'],
                                    sc['windage_persist'],
                                    time_step)

    def get_move(self, sc, time_step, model_time_datetime):
        """
        Override base class functionality because mover has a different
        get_move signature

        :param sc: an instance of the gnome.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time
                                    object
        """
        self.prepare_data_for_get_move(sc, model_time_datetime)

        if self.active and len(self.positions) > 0:
            self.mover.get_move(self.model_time, time_step,
                                self.positions, self.delta,
                                sc['windages'],
                                self.status_codes, self.spill_type)

        return (self.delta.view(dtype=world_point_type)
                .reshape((-1, len(world_point))))

    def _state_as_str(self):
        '''
            Returns a string containing properties of object.
            This can be called by __repr__ or __str__ to display props
        '''
        info = ('  uncertain_duration={0.uncertain_duration}\n'
                '  uncertain_time_delay={0.uncertain_time_delay}\n'
                '  uncertain_speed_scale={0.uncertain_speed_scale}\n'
                '  uncertain_angle_scale={0.uncertain_angle_scale}\n'
                '  active_range time={0.active_range}\n'
                '  current on/off status={0.on}\n')
        return info.format(self)


class PointWindMoverSchema(WindMoversBaseSchema):
    """
    Contains properties required by UpdateWindMover and CreateWindMover
    """
    # 'wind' schema node added dynamically
    wind = GeneralGnomeObjectSchema(
        acceptable_schemas=[WindSchema],
        save=True, update=True, save_reference=True
    )
    data_start = SchemaNode(
        LocalDateTime(), validator=convertible_to_seconds, read_only=True
    )
    data_stop = SchemaNode(
        LocalDateTime(), validator=convertible_to_seconds, read_only=True
    )
WindMoverSchema = PointWindMoverSchema

class PointWindMover(WindMoversBase):
    """
    Python wrapper around the Cython wind_mover module.
    This class inherits from CyMover and contains CyWindMover

    The real work is done by the CyWindMover object.  CyMover
    sets everything up that is common to all movers.
    """
    _schema = PointWindMoverSchema

    _ref_as = 'wind_mover'

    _req_refs = {'wind': Wind}

    def __init__(self, wind=None, **kwargs):
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
        self.mover = CyWindMover()

        self._wind = None
        if wind is not None:
            self.wind = wind
            self.name = wind.name
            kwargs['make_default_refs'] = kwargs.pop('make_default_refs',
                                                     False)
            kwargs['name'] = kwargs.pop('name', wind.name)

        # set optional attributes
        super(PointWindMover, self).__init__(**kwargs)

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}(\n{1})'
                .format(self, self._state_as_str()))

    def __str__(self):
        return ('WindMover - current _state. '
                'See "wind" object for wind conditions:\n{0}'
                .format(self._state_as_str()))

    @property
    def wind(self):
        return self._wind

    @wind.setter
    def wind(self, value):
        if not isinstance(value, Wind):
            raise TypeError('wind must be of type environment.Wind')
        else:
            # update reference to underlying cython object
            self._wind = value
            self.mover.set_ossm(self._wind.ossm)

    @property
    def data_start(self):
        return self.wind.data_start

    @property
    def data_stop(self):
        return self.wind.data_stop

    def prepare_for_model_run(self):
        '''
        if wind attribute is not set, raise ReferencedObjectNotSet excpetion
        '''
        super(PointWindMover, self).prepare_for_model_run()

        if self.on and self.wind is None:
            msg = "wind object not defined for WindMover"
            raise ReferencedObjectNotSet(msg)
WindMover = PointWindMover


def point_wind_mover_from_file(filename, **kwargs):
    """
    Creates a wind mover from a wind time-series file (OSM long wind format)

    :param filename: The full path to the data file
    :param kwargs: All keyword arguments are passed on to the WindMover
        constructor

    :returns mover: returns a wind mover, built from the file
    """
    w = Wind(filename=filename, coord_sys='r-theta')

    return PointWindMover(w, name=w.name, **kwargs)


def constant_point_wind_mover(speed, direction, units='m/s'):
    """
    utility function to create a point wind mover with a constant wind

    :param speed: wind speed
    :param direction: wind direction in degrees true
                  (direction from, following the meteorological convention)
    :param units='m/s': the units that the input wind speed is in.
                        options: 'm/s', 'knot', 'mph', others...

    :return: returns a gnome.movers.WindMover object all set up.

    .. note::
        The time for a constant wind timeseries is irrelevant.
        This function simply sets it to datetime.now() accurate to hours.
    """
    return PointWindMover(constant_wind(speed, direction, units=units))


class c_GridWindMoverSchema(WindMoversBaseSchema):
    """
        Similar to WindMover except it doesn't have wind_id
    """
    filename = FilenameSchema(
        missing=drop, save=True, update=True, isdatafile=True, test_equal=False
    )
    topology_file = FilenameSchema(
        missing=drop, save=True, update=True, isdatafile=True, test_equal=False
    )
    wind_scale = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    extrapolate = SchemaNode(
        Bool(), missing=drop, save=True, update=True
    )


class c_GridWindMover(WindMoversBase):

    _schema = c_GridWindMoverSchema

    def __init__(self, filename=None, topology_file=None,
                 extrapolate=False, time_offset=0,
                 **kwargs):
        """
        :param wind_file: file containing wind data on a grid
        :param filename: file containing wind data on a grid
        :param topology_file: Default is None. When exporting topology, it
                              is stored in this file
        :param wind_scale: Value to scale wind data
        :param extrapolate: Allow current data to be extrapolated before and
                            after file data
        :param time_offset: Time zone shift if data is in GMT

        Pass optional arguments to base class
        uses super: ``super(c_GridWindMover,self).__init__(**kwargs)``
        """
        if not os.path.exists(filename):
            raise ValueError('Path for wind file does not exist: {0}'
                             .format(filename))

        if topology_file is not None:
            if not os.path.exists(topology_file):
                raise ValueError('Path for Topology file does not exist: {0}'
                                 .format(topology_file))

        self.mover = CyGridWindMover(wind_scale=kwargs.pop('wind_scale', 1))
        self.mover.text_read(filename, topology_file)

        # Ideally, we would be able to run the base class initialization first
        # because we designed the Movers well.  As it is, we inherit from the
        # CyMover, and the CyMover needs to have a self.mover attribute.
        super(c_GridWindMover, self).__init__(**kwargs)

        # is wind_file and topology_file is stored with cy_gridwind_mover?
        self.name = os.path.split(filename)[1]
        self.filename = filename
        self.topology_file = topology_file

        self.mover.extrapolate_in_time(extrapolate)
        self.mover.offset_time(time_offset * 3600.)

    @property
    def data_start(self):
        return sec_to_datetime(self.mover.get_start_time())

    @property
    def data_stop(self):
        return sec_to_datetime(self.mover.get_end_time())

    def __repr__(self):
        """
        .. todo::
            We probably want to include more information.
        """
        return 'c_GridWindMover(\n{0})'.format(self._state_as_str())

    def __str__(self):
        return ('c_GridWindMover - current _state.\n{0}'
                .format(self._state_as_str()))

    wind_scale = property(lambda self: self.mover.wind_scale,
                          lambda self, val: setattr(self.mover, 'wind_scale',
                                                    val))

    extrapolate = property(lambda self: self.mover.extrapolate,
                           lambda self, val: setattr(self.mover, 'extrapolate',
                                                     val))

    time_offset = property(lambda self: self.mover.time_offset / 3600.,
                           lambda self, val: setattr(self.mover, 'time_offset',
                                                     val * 3600.))

    def get_grid_data(self):
        return self.get_cells()

    def get_cells(self):
        """
            Invokes the GetCellDataHdl method of TimeGridWind_c object.
            Cross-references point data to get cell coordinates.
        """
        cell_data = self.mover._get_cell_data()
        points = self.get_points()

        dtype = cell_data[0].dtype.descr
        unstructured_type = dtype[0][1]
        unstructured = (cell_data.view(dtype=unstructured_type)
                        .reshape(-1, len(dtype))[:, 1:])

        return points[unstructured]

    def get_points(self):
        points = (self.mover._get_points()
                  .astype([('long', '<f8'), ('lat', '<f8')]))
        points['long'] /= 10 ** 6
        points['lat'] /= 10 ** 6

        return points

    def get_cell_center_points(self):
        '''
        Right now the cython mover only gets the triangular center points,
        so we need to calculate centers based on the cells themselves.

        Cells will have the format (tl, tr, bl, br)
        We need to get the rectangular centers
        Center will be: (tl + ((br - tl) / 2.))
        '''
        return (self.mover._get_center_points()
                .view(dtype='<f8').reshape(-1, 2))

    def get_center_points(self):
        return self.get_cell_center_points()

    def get_scaled_velocities(self, time):
        """
        :param model_time=0:
        """
        # regular and curvilinear grids only
        if self.mover._is_regular_grid():
            num_cells = self.mover.get_num_points()
        else:
            num_tri = self.mover.get_num_triangles()
            num_cells = num_tri / 2

        # will need to update this for regular grids
        vels = np.zeros(num_cells, dtype=velocity_rec)
        self.mover.get_scaled_velocities(time, vels)

        return vels

    def export_topology(self, topology_file):
        """
        :param topology_file=None: absolute or relative path where topology
                                   file will be written.
        """
        if topology_file is None:
            raise ValueError('Topology file path required: {0}'.
                             format(topology_file))

        self.mover.export_topology(topology_file)

    def extrapolate_in_time(self, extrapolate):
        """
        :param extrapolate=false: Allow current data to be extrapolated before
                                  and after file data.
        """
        self.mover.extrapolate_in_time(extrapolate)

    def offset_time(self, time_offset):
        """
        :param offset_time=0: Allow data to be in GMT with a time zone offset
                              (hours).
        """
        self.mover.offset_time(time_offset * 3600.)


class IceWindMoverSchema(WindMoversBaseSchema):
    filename = FilenameSchema(
        missing=drop, save=True, isdatafile=True, test_equal=False
    )
    topology_file = FilenameSchema(
        missing=drop, save=True, isdatafile=True, test_equal=False
    )


class IceWindMover(WindMoversBase):

    _schema = IceWindMoverSchema

    def __init__(self,
                 filename=None,
                 topology_file=None,
                 extrapolate=False,
                 time_offset=0,
                 **kwargs):
        """
        Initialize an IceWindMover

        :param filename: absolute or relative path to the data file:
                         could be netcdf or filelist
        :param topology_file=None: absolute or relative path to topology file.
                                   If not given, the IceMover will
                                   compute the topology from the data file.

        :param active_range: Range of datetimes for when the mover should be
                             active
        :type active_range: 2-tuple of datetimes

        :param wind_scale: Value to scale wind data
        :param extrapolate: Allow current data to be extrapolated
                            before and after file data
        :param time_offset: Time zone shift if data is in GMT

        uses super, ``super(IceWindMover,self).__init__(**kwargs)``
        """

        # NOTE: will need to add uncertainty parameters and other dialog fields
        #       use super with kwargs to invoke base class __init__

        # if child is calling, the self.mover is set by child - do not reset
        if type(self) == IceWindMover:
            self.mover = CyIceWindMover()

        if not os.path.exists(filename):
            raise ValueError('Path for current file does not exist: {0}'
                             .format(filename))

        if topology_file is not None:
            if not os.path.exists(topology_file):
                raise ValueError('Path for Topology file does not exist: {0}'
                                 .format(topology_file))

        # check if this is stored with cy_ice_wind_mover?
        self.name = os.path.split(filename)[1]
        self.filename = filename
        self.topology_file = topology_file

        # check if this is stored with cy_ice_wind_mover?

        self.extrapolate = extrapolate

        self.mover.text_read(filename, topology_file)
        self.mover.extrapolate_in_time(extrapolate)
        self.mover.offset_time(time_offset * 3600.)

        super(IceWindMover, self).__init__(**kwargs)

    def __repr__(self):
        return ('IceWindMover('
                'active_range={1.active_range}, '
                'on={1.on})'
                .format(self.mover, self))

    def __str__(self):
        return ('IceWindMover - current _state.\n'
                '  active_range time={1.active_range}\n'
                '  current on/off status={1.on}'
                .format(self.mover, self))

    def get_grid_data(self):
        if self.mover._is_triangle_grid():
            return self.get_triangles()
        else:
            return self.get_cells()

    def get_center_points(self):
        if self.mover._is_triangle_grid():
            return self.get_triangle_center_points()
        else:
            return self.get_cell_center_points()

    def get_scaled_velocities(self, model_time):
        """
        :param model_time=0:
        """
        num_tri = self.mover.get_num_triangles()

        if self.mover._is_triangle_grid():
            num_cells = num_tri
        else:
            num_cells = num_tri / 2

        vels = np.zeros(num_cells, dtype=velocity_rec)
        self.mover.get_scaled_velocities(model_time, vels)

        return vels

    def get_ice_velocities(self, model_time):
        """
        :param model_time=0:
        """
        num_tri = self.mover.get_num_triangles()
        vels = np.zeros(num_tri, dtype=velocity_rec)

        self.mover.get_ice_velocities(model_time, vels)

        return vels

    def get_movement_velocities(self, model_time):
        """
        :param model_time=0:
        """
        num_tri = self.mover.get_num_triangles()
        vels = np.zeros(num_tri, dtype=velocity_rec)

        self.mover.get_movement_velocities(model_time, vels)

        return vels

    def get_ice_fields(self, model_time):
        """
        :param model_time=0:
        """
        num_tri = self.mover.get_num_triangles()
        num_cells = num_tri // 2

        frac_coverage = np.zeros(num_cells, dtype=np.float64)
        thickness = np.zeros(num_cells, dtype=np.float64)

        self.mover.get_ice_fields(model_time, frac_coverage, thickness)

        return frac_coverage, thickness

    def export_topology(self, topology_file):
        """
        :param topology_file=None: absolute or relative path where
                                   topology file will be written.
        """
        if topology_file is None:
            raise ValueError('Topology file path required: {0}'
                             .format(topology_file))

        self.mover.export_topology(topology_file)

    def extrapolate_in_time(self, extrapolate):
        """
        :param extrapolate=false: allow current data to be extrapolated
                                  before and after file data.
        """
        self.mover.extrapolate_in_time(extrapolate)
        self.extrapolate = extrapolate

    def offset_time(self, time_offset):
        """
        :param offset_time=0: allow data to be in GMT with a time zone offset
                              (hours).
        """
        self.mover.offset_time(time_offset * 3600.)

    def get_offset_time(self):
        """
        :param offset_time=0: allow data to be in GMT with a time zone offset
                              (hours).
        """
        return self.mover.get_offset_time() / 3600.
