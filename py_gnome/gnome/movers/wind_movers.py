'''
Movers using wind as the forcing function
'''

import os
import copy
from datetime import datetime
import math

import numpy
np = numpy
from colander import (SchemaNode, String, Float, drop)

from gnome.basic_types import (ts_format,
                               world_point,
                               world_point_type,
                               datetime_value_2d)
from gnome import array_types

from gnome.utilities import serializable, rand

from gnome import environment
from gnome.movers import CyMover, MoverSchema
from gnome.cy_gnome.cy_wind_mover import CyWindMover
from gnome.cy_gnome.cy_gridwind_mover import CyGridWindMover

from gnome.persist.base_schema import ObjType
#from gnome.persist import movers_schema
#from gnome.persist.environment_schema import Wind


class WindMoversBaseSchema(ObjType, MoverSchema):
    uncertain_duration = SchemaNode(Float(), missing=drop)
    uncertain_time_delay = SchemaNode(Float(), missing=drop)
    uncertain_speed_scale = SchemaNode(Float(), missing=drop)
    uncertain_angle_scale = SchemaNode(Float(), missing=drop)
    uncertain_angle_units = SchemaNode(String(), missing=drop)


class WindMoverSchema(WindMoversBaseSchema):
    """
    Contains properties required by UpdateWindMover and CreateWindMover
    """
    # 'wind' schema node added dynamically
    name = 'WindMover'
    description = 'wind mover properties'


class WindMoversBase(CyMover):
    _state = copy.deepcopy(CyMover._state)
    _state.add(update=['uncertain_duration', 'uncertain_time_delay',
                      'uncertain_speed_scale'],
              save=['uncertain_duration', 'uncertain_time_delay',
                      'uncertain_speed_scale', 'uncertain_angle_scale',
                      'uncertain_angle_units'],
              read=['uncertain_angle_scale'])

    def __init__(self,
                 uncertain_duration=3,
                 uncertain_time_delay=0,
                 uncertain_speed_scale=2.,
                 uncertain_angle_scale=0.4,
                 uncertain_angle_units='rad',
                 **kwargs):
        """
        This is simply a base class for WindMover and GridWindMover for the
        common properties.

        The classes that inherit from this should define the self.mover object
        correctly so it has the required attributes.

        Input args with defaults:

        :param uncertain_duration: (seconds) the randomly generated uncertainty
            array gets recomputed based on 'uncertain_duration'
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param uncertain_speed_scale: Scale for uncertainty in wind speed
            non-dimensional number
        :param uncertain_angle_scale: Scale for uncertainty in wind direction
            'deg' or 'rad'
        :param uncertain_angle_units: 'rad' or 'deg'. These are the units for
            the uncertain_angle_scale.

        It calls super in the __init__ method and passes in the optional
        parameters (kwargs)
        """
        super(WindMoversBase, self).__init__(**kwargs)

        self.uncertain_duration = uncertain_duration
        self.uncertain_time_delay = uncertain_time_delay
        self.uncertain_speed_scale = uncertain_speed_scale

        # also sets self._uncertain_angle_units
        self.set_uncertain_angle(uncertain_angle_scale, uncertain_angle_units)

        self.array_types.update({'windages': array_types.windages,
                                 'windage_range': array_types.windage_range,
                                 'windage_persist': array_types.windage_persist
                                 })

    # no conversion necessary - simply sets/gets the stored value
    uncertain_speed_scale = \
        property(lambda self: self.mover.uncertain_speed_scale,
                 lambda self, val: setattr(self.mover,
                                           'uncertain_speed_scale',
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

    @property
    def uncertain_angle_units(self):
        """
        units specified by the user when setting the uncertain_angle:
        set_uncertain_angle()
        """
        return self._uncertain_angle_units

    @property
    def uncertain_angle_scale(self):
        '''
        Read only - this is set when set_uncertain_angle() is called
        It returns the angle in 'uncertain_angle_units'
        '''
        if self.uncertain_angle_units == 'deg':
            return self.mover.uncertain_angle_scale * 180.0 / math.pi
        else:
            return self.mover.uncertain_angle_scale

    def set_uncertain_angle(self, val, units):
        '''
        this must be a function because user must provide units with value
        '''
        if units not in ['deg', 'rad']:
            raise ValueError("units for uncertain angle can be either"
                             " 'deg' or 'rad'")

        if units == 'deg':
            # convert to radians
            self.mover.uncertain_angle_scale = val * math.pi / 180.0
        else:
            self.mover.uncertain_angle_scale = val

        self._uncertain_angle_units = units

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
        if sc.num_released is None  or sc.num_released == 0:
            return

        rand.random_with_persistance(sc['windage_range'][:, 0],
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
                '  uncertain_angle_units="{0.uncertain_angle_units}"\n'
                '  active_start time={0.active_start}\n'
                '  active_stop time={0.active_stop}\n'
                '  current on/off status={0.on}\n')
        return info.format(self)


class WindMover(WindMoversBase, serializable.Serializable):
    """
    Python wrapper around the Cython wind_mover module.
    This class inherits from CyMover and contains CyWindMover

    The real work is done by the CyWindMover object.  CyMover
    sets everything up that is common to all movers.

    In addition to base class array_types.basic, also use the
    array_types.windage dict since WindMover requires a windage array
    """
    _state = copy.deepcopy(WindMoversBase._state)
    #_state.add(read=['wind_id'], save=['wind_id'])
    # todo: probably need to make update=True for 'wind' as well
    _state.add_field(serializable.Field('wind', save=True, update=True,
                                         save_reference=True))
    _schema = WindMoverSchema

    def __init__(self, wind, **kwargs):
        """
        Uses super to call CyMover base class __init__

        :param wind: wind object -- provides the wind time series for the mover

        Remaining kwargs are passed onto WindMoversBase __init__ using super.
        See Mover documentation for remaining valid kwargs.
        """
        self.mover = CyWindMover()
        self.wind = wind
        self.name = wind.name

        # set optional attributes
        super(WindMover, self).__init__(**kwargs)

    def __repr__(self):
        """
        .. todo::
            We probably want to include more information.
        """
        return ('{0.__class__.__module__}.{0.__class__.__name__}(\n'
                '{1}'
                ')'.format(self, self._state_as_str()))

    def __str__(self):
        info = ('WindMover - current _state. '
                'See "wind" object for wind conditions:\n'
                '{0}'.format(self._state_as_str()))
        return info

    @property
    def wind(self):
        return self._wind

    @wind.setter
    def wind(self, value):
        if not isinstance(value, environment.Wind):
            raise TypeError('wind must be of type environment.Wind')
        else:
            # update reference to underlying cython object
            self._wind = value
            self.mover.set_ossm(self.wind.ossm)

    def serialize(self, json_='webapi'):
        """
        Since 'wind' property is saved as a reference when used in save file
        and 'save' option, need to add appropriate node to WindMover schema
        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()
        if json_ == 'webapi':
            # add wind schema
            schema.add(environment.WindSchema())

        serial = schema.serialize(toserial)

        return serial

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for wind object
        """
        schema = cls._schema()
        if 'wind' in json_:
            schema.add(environment.WindSchema())
        _to_dict = schema.deserialize(json_)

        return _to_dict


def wind_mover_from_file(filename, **kwargs):
    """
    Creates a wind mover from a wind time-series file (OSM long wind format)

    :param filename: The full path to the data file
    :param kwargs: All keyword arguments are passed on to the WindMover
        constructor

    :returns mover: returns a wind mover, built from the file
    """
    w = environment.Wind(filename=filename, format='r-theta')
    wm = WindMover(w, **kwargs)

    return wm


def constant_wind_mover(speed, direction, units='m/s'):
    """
    utility function to create a mover with a constant wind

    :param speed: wind speed
    :param direction: wind direction in degrees true
                  (direction from, following the meteorological convention)
    :param units='m/s': the units that the input wind speed is in.
                        options: 'm/s', 'knot', 'mph', others...

    :return: returns a gnome.movers.WindMover object all set up.
    """

    series = np.zeros((1, ), dtype=datetime_value_2d)

    # note: if there is ony one entry, the time is arbitrary

    series[0] = (datetime.now(), (speed, direction))
    wind = environment.Wind(timeseries=series, units=units)
    w_mover = WindMover(wind)
    return w_mover


class GridWindMoverSchema(WindMoversBaseSchema):
    """ Similar to WindMover except it doesn't have wind_id"""
    wind_file = SchemaNode(String(), missing=drop)
    topology_file = SchemaNode(String(), missing=drop)


class GridWindMover(WindMoversBase, serializable.Serializable):
    _state = copy.deepcopy(WindMoversBase._state)
    _state.add(update=['wind_scale'], save=['wind_scale'])
    _state.add_field([serializable.Field('wind_file', save=True,
                    read=True, isdatafile=True, test_for_eq=False),
                    serializable.Field('topology_file', save=True,
                    read=True, isdatafile=True, test_for_eq=False)])

    _schema = GridWindMoverSchema

    def __init__(self, wind_file, topology_file=None,
                 extrapolate=False, time_offset=0,
                 **kwargs):
        """
        :param wind_file: file containing wind data on a grid
        :param topology_file: Default is None. When exporting topology, it
                              is stored in this file
        :param wind_scale: Value to scale wind data
        :param extrapolate: Allow current data to be extrapolated before and
                            after file data
        :param time_offset: Time zone shift if data is in GMT

        Pass optional arguments to base class
        uses super: super(GridWindMover,self).__init__(\*\*kwargs)
        """

        if not os.path.exists(wind_file):
            raise ValueError('Path for wind file does not exist: {0}'
                             .format(wind_file))

        if topology_file is not None:
            if not os.path.exists(topology_file):
                raise ValueError('Path for Topology file does not exist: {0}'
                                 .format(topology_file))

        # is wind_file and topology_file is stored with cy_gridwind_mover?
        self.wind_file = wind_file
        self.topology_file = topology_file
        self.mover = CyGridWindMover(wind_scale=kwargs.pop('wind_scale', 1))
        self.name = os.path.split(wind_file)[1]
        super(GridWindMover, self).__init__(**kwargs)

        self.mover.text_read(wind_file, topology_file)
        self.mover.extrapolate_in_time(extrapolate)
        self.mover.offset_time(time_offset * 3600.)

    def __repr__(self):
        """
        .. todo::
            We probably want to include more information.
        """
        info = 'GridWindMover(\n{0})'.format(self._state_as_str())
        return info

    def __str__(self):
        info = ('GridWindMover - current _state.\n'
                '{0}'.format(self._state_as_str()))
        return info

    wind_scale = property(lambda self: self.mover.wind_scale,
                          lambda self, val: setattr(self.mover,
                                                    'wind_scale',
                                                    val))

    extrapolate = property(lambda self: self.mover.extrapolate,
                           lambda self, val: setattr(self.mover,
                                                     'extrapolate',
                                                     val))

    time_offset = property(lambda self: self.mover.time_offset / 3600.,
                           lambda self, val: setattr(self.mover,
                                                     'time_offset',
                                                     val * 3600.))

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
