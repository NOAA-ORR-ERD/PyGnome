'''
Movers using wind as the forcing function
'''

import os
import copy
from datetime import datetime

import numpy as np

from gnome.utilities import serializable
from gnome.movers import CyMover
from gnome import basic_types
from gnome.cy_gnome.cy_wind_mover import CyWindMover
from gnome.cy_gnome.cy_gridwind_mover import CyGridWindMover
from gnome import environment
from gnome.utilities import rand
from gnome.array_types import ArrayType


class WindMover(CyMover, serializable.Serializable):

    """
    Python wrapper around the Cython wind_mover module.
    This class inherits from CyMover and contains CyWindMover

    The real work is done by the CyWindMover object.  CyMover
    sets everything up that is common to all movers.

    In addition to base class array_types.basic, also use the
    array_types.windage dict since WindMover requires a windage array
    """
    _update = ['uncertain_duration', 'uncertain_time_delay',
               'uncertain_speed_scale', 'uncertain_angle_scale']
    _create = ['wind_id']
    _read = ['wind_id']
    _create.extend(_update)

    state = copy.deepcopy(CyMover.state)
    state.add(read=_read, update=_update, create=_create)

    @classmethod
    def new_from_dict(cls, dict_):
        """
        define in WindMover and check wind_id matches wind

        invokes: super(WindMover,cls).new_from_dict(dict\_)
        """

        wind_id = dict_.pop('wind_id')
        if dict_.get('wind').id != wind_id:
            raise ValueError('id of wind object does not match the wind_id'\
                             ' parameter')

        return super(WindMover, cls).new_from_dict(dict_)

    def wind_id_to_dict(self):
        """
        used only for storing state so no wind_id_from_dict is defined. This
        is not a read/write attribute. Only defined for serializable_state
        """

        return self.wind.id

    def from_dict(self, dict_):
        """
        For updating the object from dictionary

        'wind' object is not part of the state since it is not serialized/
        deserialized; however, user can still update the wind attribute with a
        new Wind object. That must be poped out of the dict() here, then call
        super to process the standard dict\_
        """

        self.wind = dict_.pop('wind', self.wind)

        super(WindMover, self).from_dict(dict_)

    def __init__(self, wind, **kwargs):
        """
        Uses super to call CyMover base class __init__

        :param wind: wind object -- provides the wind time series for the mover

        Optional parameters (kwargs):

        :param uncertain_duration: (seconds) how often does a given uncertain
                                   windage get re-set
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param uncertain_speed_scale: Scale for uncertainty in wind speed
        :param uncertain_angle_scale: Scale for uncertainty in wind direction

        Remaining kwargs are passed onto Mover's __init__ using super.
        See Mover documentation for remaining valid kwargs.
        """

        self.mover = \
        CyWindMover(uncertain_duration=kwargs.pop('uncertain_duration', 10800),
             uncertain_time_delay=kwargs.pop('uncertain_time_delay', 0),
             uncertain_speed_scale=kwargs.pop('uncertain_speed_scale', 2.),
             uncertain_angle_scale=kwargs.pop('uncertain_angle_scale', 0.4))

        self.wind = wind
        super(WindMover, self).__init__(**kwargs)
        self.array_types.update(
                  {'windages': ArrayType((), basic_types.windage_type),
                   'windage_range': ArrayType((2,), np.float64),
                   'windage_persist': ArrayType((), np.int)})

    def __repr__(self):
        """
        .. todo::
            We probably want to include more information.
        """

        info = \
            'WindMover( wind=<wind_object>, uncertain_duration='\
            + '{0.uncertain_duration},' \
            + 'uncertain_time_delay={0.uncertain_time_delay}, '\
            + 'uncertain_speed_scale={0.uncertain_speed_scale}, ' \
            + 'uncertain_angle_scale={0.uncertain_angle_scale}, '\
            + 'active_start={1.active_start}, active_stop={1.active_stop}, '\
            + 'on={1.on})'
        return info.format(self.mover, self)

    def __str__(self):
        info = \
            "WindMover - current state." \
            + " See 'wind' object for wind conditions:\n" \
            + '  uncertain_duration={0.uncertain_duration}\n' \
            + '  uncertain_time_delay={0.uncertain_time_delay}\n' \
            + '  uncertain_speed_scale={0.uncertain_speed_scale}\n' \
            + '  uncertain_angle_scale={0.uncertain_angle_scale}' \
            + '  active_start time={1.active_start}' \
            + '  active_stop time={1.active_stop}' \
            + '  current on/off status={1.on}'
        return info.format(self.mover, self)

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

    # Define properties using lambda functions: uses lambda function,
    # which are accessible via fget/fset as follows:
    uncertain_duration = property(lambda self: \
                                  self.mover.uncertain_duration,
                                  lambda self, val: setattr(self.mover,
                                  'uncertain_duration', val))

    uncertain_time_delay = property(lambda self: \
                                    self.mover.uncertain_time_delay,
                                    lambda self, val: \
                                    setattr(self.mover,
                                    'uncertain_time_delay', val))

    uncertain_speed_scale = property(lambda self: \
            self.mover.uncertain_speed_scale, lambda self, val: \
            setattr(self.mover, 'uncertain_speed_scale', val))

    uncertain_angle_scale = property(lambda self: \
            self.mover.uncertain_angle_scale, lambda self, val: \
            setattr(self.mover, 'uncertain_angle_scale', val))

    def prepare_for_model_step(
        self,
        sc,
        time_step,
        model_time_datetime,
        ):
        """
        Call base class method using super
        Also updates windage for this timestep

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of model as a date time object
        """

        super(WindMover, self).prepare_for_model_step(sc, time_step,
                model_time_datetime)

        # if no particles released, then no need for windage
        # todo: revisit this since sc.num_released shouldn't be None
        if sc.num_released is None  or sc.num_released == 0:
            return

        for spill in sc.spills:
            spill_mask = sc.get_spill_mask(spill)
            if np.any(spill_mask):
                sc['windages'][spill_mask] = \
                    rand.random_with_persistance(spill.windage_range[0],
                        spill.windage_range[1], spill.windage_persist,
                        time_step,
                        array_len=np.count_nonzero(spill_mask))

    def get_move(
        self,
        sc,
        time_step,
        model_time_datetime,
        ):
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
            self.mover.get_move(
                self.model_time,
                time_step,
                self.positions,
                self.delta,
                sc['windages'],
                self.status_codes,
                self.spill_type,
                )

        return self.delta.view(dtype=basic_types.world_point_type).reshape((-1,
                len(basic_types.world_point)))


def wind_mover_from_file(filename, **kwargs):
    """
    Creates a wind mover from a wind time-series file (OSM long wind format)

    :param filename: The full path to the data file
    :param **kwargs: All keyword arguments are passed on to the WindMover
                     constructor

    :returns mover: returns a wind mover, built from the file
    """

    w = environment.Wind(filename=filename,
                         ts_format=basic_types.ts_format.magnitude_direction)
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


    :returns WindMover: returns a gnome.movers.WindMover object all set up.
    """

    series = np.zeros((1, ), dtype=basic_types.datetime_value_2d)

    # note: if there is ony one entry, the time is arbitrary

    series[0] = (datetime.now(), (speed, direction))
    wind = environment.Wind(timeseries=series, units=units)
    w_mover = WindMover(wind)
    return w_mover


class GridWindMover(CyMover, serializable.Serializable):

    _update = ['uncertain_duration', 'uncertain_time_delay',
               'uncertain_speed_scale', 'uncertain_angle_scale']

    # _create = ['wind_file', 'topology_file']
    # _read = ['wind_file', 'topology_file']
    # _create.extend(_update)

    state = copy.deepcopy(CyMover.state)
    state.add(update=_update, create=_update)
    state.add_field([serializable.Field('wind_file', create=True,
                    read=True, isdatafile=True),
                    serializable.Field('topology_file', create=True,
                    read=True, isdatafile=True)])

    def __init__(
        self,
        wind_file,
        topology_file=None,
        uncertain_duration=10800,
        uncertain_time_delay=0,
        uncertain_speed_scale=2.,
        uncertain_angle_scale=0.4,
        **kwargs
        ):
        """
        :param active_start: datetime when the mover should be active
        :param active_stop: datetime after which the mover should be inactive
        :param uncertain_duration: (seconds) how often does a given uncertian
                                   windage get re-set
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param uncertain_speed_scale: Scale for uncertainty in wind speed
        :param uncertain_angle_scale: Scale for uncertainty in wind direction

        uses super: super(GridWindMover,self).__init__(**kwargs)
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
        self.mover = CyGridWindMover()
        self.mover.text_read(wind_file, topology_file)

#         self.mover = CyGridWindMover(uncertain_duration=uncertain_duration,
#                                  uncertain_time_delay=uncertain_time_delay,
#                                  uncertain_speed_scale=uncertain_speed_scale,
#                                  uncertain_angle_scale=uncertain_angle_scale)

        super(GridWindMover, self).__init__(**kwargs)

    def __repr__(self):
        """
        .. todo::
            We probably want to include more information.
        """

        info = \
            'GridWindMover( uncertain_duration={0.uncertain_duration},' \
            + 'uncertain_time_delay={0.uncertain_time_delay}, '\
            + 'uncertain_speed_scale={0.uncertain_speed_scale}, ' \
            + 'uncertain_angle_scale={0.uncertain_angle_scale}, '\
            + 'active_start={1.active_start}, active_stop={1.active_stop}, '\
            + 'on={1.on})'
        return info.format(self.mover, self)

    def __str__(self):
        info = 'GridWindMover - current state.\n' \
            + '  uncertain_duration={0.uncertain_duration}\n' \
            + '  uncertain_time_delay={0.uncertain_time_delay}\n' \
            + '  uncertain_speed_scale={0.uncertain_speed_scale}\n' \
            + '  uncertain_angle_scale={0.uncertain_angle_scale}' \
            + '  active_start time={1.active_start}' \
            + '  active_stop time={1.active_stop}' \
            + '  current on/off status={1.on}'
        return info.format(self.mover, self)

    # Define properties using lambda functions: uses lambda function, which are
    #accessible via fget/fset as follows:
    uncertain_duration = property(lambda self: \
                                  self.mover.uncertain_duration,
                                  lambda self, val: setattr(self.mover,
                                  'uncertain_duration', val))

    uncertain_time_delay = property(lambda self: \
                                    self.mover.uncertain_time_delay,
                                    lambda self, val: \
                                    setattr(self.mover,
                                    'uncertain_time_delay', val))

    uncertain_speed_scale = property(lambda self: \
            self.mover.uncertain_speed_scale, lambda self, val: \
            setattr(self.mover, 'uncertain_speed_scale', val))

    uncertain_angle_scale = property(lambda self: \
            self.mover.uncertain_angle_scale, lambda self, val: \
            setattr(self.mover, 'uncertain_angle_scale', val))

    def prepare_for_model_step(
        self,
        sc,
        time_step,
        model_time_datetime,
        ):
        """
        Call base class method using super and also update windage for this
        timestep

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current model time
        :type model_time: datetime object
        """

        super(GridWindMover, self).prepare_for_model_step(sc,
                time_step, model_time_datetime)

        # if no particles released, then no need for windage

        if len(sc['positions']) == 0:
            return

        for spill in sc.spills:
            spill_mask = sc.get_spill_mask(spill)
            if np.any(spill_mask):
                sc['windages'][spill_mask] = \
                    rand.random_with_persistance(spill.windage_range[0],
                        spill.windage_range[1], spill.windage_persist,
                        time_step,
                        array_len=np.count_nonzero(spill_mask))

    def get_move(
        self,
        sc,
        time_step,
        model_time_datetime,
        ):
        """
        Override base class functionality because mover has a different
        get_move signature

        :param sc: an instance of the gnome.SpillContainer class
        :param time_step: timestep in seconds
        :param model_time_datetime: current model time as datetime object
        :type model_time: datetime
        """

        self.prepare_data_for_get_move(sc, model_time_datetime)

        if self.active and len(self.positions) > 0:
            self.mover.get_move(
                self.model_time,
                time_step,
                self.positions,
                self.delta,
                sc['windages'],
                self.status_codes,
                self.spill_type,
                )

        return self.delta.view(dtype=basic_types.world_point_type).reshape((-1,
                len(basic_types.world_point)))

    def export_topology(self, topology_file):
        """
        :param topology_file=None: absolute or relative path where topology
                                   file will be written.
        """

        if topology_file is None:
            raise ValueError('Topology file path required: {0}'.
                             format(topology_file))

        self.mover.export_topology(topology_file)
