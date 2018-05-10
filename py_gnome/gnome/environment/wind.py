"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""
import datetime
import os
import copy
import StringIO
import zipfile
import gridded

import numpy as np

from colander import (SchemaNode, drop, OneOf,
                      Float, String, Range, Boolean)

import unit_conversion as uc

from gnome.basic_types import datetime_value_2d
from gnome.basic_types import coord_systems
from gnome.basic_types import wind_datasources

from gnome.cy_gnome.cy_ossm_time import ossm_wind_units

from gnome.utilities import serializable
from gnome.utilities.time_utils import sec_to_datetime
from gnome.utilities.timeseries import Timeseries
from gnome.utilities.inf_datetime import InfDateTime
from gnome.utilities.distributions import RayleighDistribution as rayleigh

from gnome.persist.extend_colander import (DefaultTupleSchema,
                                           LocalDateTime,
                                           DatetimeValue2dArraySchema)
from gnome.persist import validators, base_schema

from .environment import Environment
from .. import _valid_units


class MagnitudeDirectionTuple(DefaultTupleSchema):
    speed = SchemaNode(Float(),
                       default=0,
                       validator=Range(min=0,
                                       min_err='wind speed must be '
                                               'greater than or equal to 0'
                                       )
                       )
    direction = SchemaNode(Float(), default=0,
                           validator=Range(0, 360,
                                           min_err='wind direction must be '
                                                   'greater than or equal to '
                                                   '0',
                                           max_err='wind direction must be '
                                                   'less than or equal to '
                                                   '360deg'
                                           )
                           )


class WindTupleSchema(DefaultTupleSchema):
    '''
    Schema for each tuple in WindTimeSeries list
    '''
    datetime = SchemaNode(LocalDateTime(default_tzinfo=None),
                          default=base_schema.now,
                          validator=validators.convertible_to_seconds)
    mag_dir = MagnitudeDirectionTuple()


class WindTimeSeriesSchema(DatetimeValue2dArraySchema):
    '''
    Schema for list of Wind tuples, to make the wind timeseries
    '''
    value = WindTupleSchema(default=(datetime.datetime.now(), (0, 0)))

    def validator(self, node, cstruct):
        '''
        validate wind timeseries numpy array
        '''
        validators.no_duplicate_datetime(node, cstruct)
        validators.ascending_datetime(node, cstruct)


class WindSchema(base_schema.ObjTypeSchema):
    '''
    validate data after deserialize, before it is given back to pyGnome's
    from_dict to set _state of object
    '''
    description = SchemaNode(String(), missing=drop)
    filename = SchemaNode(String(), missing=drop)
    updated_at = SchemaNode(LocalDateTime(), missing=drop)

    latitude = SchemaNode(Float(), missing=drop)
    longitude = SchemaNode(Float(), missing=drop)
    source_id = SchemaNode(String(), missing=drop)
    source_type = SchemaNode(String(),
                             validator=OneOf(wind_datasources._attr),
                             default='undefined', missing='undefined')
    units = SchemaNode(String(), default='m/s')
    speed_uncertainty_scale = SchemaNode(Float(), missing=drop)

    timeseries = WindTimeSeriesSchema(missing=drop)
    extrapolation_is_allowed = SchemaNode(Boolean(), missing=drop)
    name = 'wind'


class Wind(serializable.Serializable, Timeseries, Environment):
    '''
    Defines the Wind conditions for a single point
    '''
    # object is referenced by others using this attribute name
    _ref_as = 'wind'

    # default units for input/output data
    _update = ['description',
               'latitude',
               'longitude',
               'source_type',
               'source_id',  # what is source ID? Buoy ID?
               'updated_at',
               'speed_uncertainty_scale']

    # used to create new obj or as readonly parameter
    _create = []
    _create.extend(_update)

    _state = copy.deepcopy(Environment._state)
    _state.add(save=_create, update=_update)
    _schema = WindSchema

    # add 'filename' as a Field object
    _state.add_field([serializable.Field('filename', isdatafile=True,
                                         save=True, read=True,
                                         test_for_eq=False),
                      serializable.Field('timeseries', save=False,
                                         update=True),
                      serializable.Field('extrapolation_is_allowed', save=True,
                                         update=True),
                      # test for equality of units a little differently
                      serializable.Field('units', save=True,
                                         update=True, test_for_eq=False),
                      ])
    _state['name'].test_for_eq = False

    # list of valid velocity units for timeseries
    valid_vel_units = _valid_units('Velocity')

    def __init__(self,
                 timeseries=None,
                 units=None,
                 filename=None,
                 coord_sys='r-theta',
                 latitude=None,
                 longitude=None,
                 speed_uncertainty_scale=0.0,
                 extrapolation_is_allowed=False,
                 **kwargs):
        """
        todo: update docstrings!
        """
        self.updated_at = kwargs.pop('updated_at', None)
        self.source_id = kwargs.pop('source_id', 'undefined')

        self.longitude = longitude
        self.latitude = latitude

        self.description = kwargs.pop('description', 'Wind Object')
        self.speed_uncertainty_scale = speed_uncertainty_scale

        # TODO: the way we are doing this, super() is not being used
        #       effectively.  We should tailor kwargs in a way that we can
        #       just pass it into the base __init__() function.
        #       As it is, we are losing arguments that we then need to
        #       explicitly handle.
        if filename is not None:
            self.source_type = kwargs.pop('source_type', 'file')

            super(Wind, self).__init__(filename=filename, coord_sys=coord_sys)

            self.name = kwargs.pop('name', os.path.split(self.filename)[1])
            # set _user_units attribute to match user_units read from file.
            self._user_units = self.ossm.user_units

            if units is not None:
                self.units = units
        else:
            if kwargs.get('source_type') in wind_datasources._attr:
                self.source_type = kwargs.pop('source_type')
            else:
                self.source_type = 'undefined'

            # either timeseries is given or nothing is given
            # create an empty default object
            super(Wind, self).__init__(coord_sys=coord_sys)

            self.units = 'mps'  # units for default object

            if timeseries is not None:
                if units is None:
                    raise TypeError('Units must be provided with timeseries')

                self.set_wind_data(timeseries, units, coord_sys)

            self.name = kwargs.pop('name', self.__class__.__name__)

        self.extrapolation_is_allowed = extrapolation_is_allowed

    def _check_units(self, units):
        '''
        Checks the user provided units are in list Wind.valid_vel_units
        '''
        if units not in Wind.valid_vel_units:
            raise uc.InvalidUnitError((units, 'Velocity'))

    def __repr__(self):
        self_ts = self.timeseries.__repr__()
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'description="{0.description}", '
                'source_id="{0.source_id}", '
                'source_type="{0.source_type}", '
                'units="{0.units}", '
                'updated_at="{0.updated_at}", '
                'timeseries={1})'
                .format(self, self_ts))

    @property
    def timeseries(self):
        '''
        returns entire timeseries in 'r-theta' coordinate system in the units
        in which the data was entered or as specified by units attribute
        '''
        return self.get_wind_data(units=self.units)

    @timeseries.setter
    def timeseries(self, value):
        '''
        set the timeseries for wind. The units for value are as specified by
        self.units attribute. Property converts the units to 'm/s' so Cython/
        C++ object stores timeseries in 'm/s'
        '''
        self.set_wind_data(value, units=self.units)

    @property
    def data_start(self):
        """
        The start time of the valid data for this wind timeseries

        If there is one data point -- it's a constant wind
        so data_start is -InfDateTime
        """

        if self.ossm.get_num_values() == 1:
            return InfDateTime("-inf")
        else:
            return sec_to_datetime(self.ossm.get_start_time())

    @property
    def data_stop(self):
        """The stop time of the valid data for this wind timeseries

        If there is one data point -- it's a constant wind
        so data_start is -InfDateTime
        """
        if self.ossm.get_num_values() == 1:
            return InfDateTime("inf")
        else:
            return sec_to_datetime(self.ossm.get_end_time())

    def timeseries_to_dict(self):
        '''
        when serializing data - round it to 2 decimal places
        '''
        ts = self.get_wind_data(units=self.units)
        ts['value'][:] = np.round(ts['value'], 2)

        return ts

    @property
    def units(self):
        '''
            define units in which wind data is input/output
        '''
        return self._user_units

    @units.setter
    def units(self, value):
        """
        User can set default units for input/output data

        These are given as string - derived classes should override
        _check_units() to customize for their data. Base class first checks
        units, then sets it - derived classes can raise an error in
        _check_units if units are incorrect for their type of data
        """
        self._check_units(value)
        self._user_units = value

    def _convert_units(self, data, coord_sys, from_unit, to_unit):
        '''
        method to convert units for the 'value' stored in the
        date/time value pair
        '''
        if from_unit != to_unit:
            data[:, 0] = uc.convert('Velocity', from_unit, to_unit, data[:, 0])

            if coord_sys == coord_systems.uv:
                data[:, 1] = uc.convert('Velocity', from_unit, to_unit,
                                        data[:, 1])

        return data

    def save(self, saveloc, references=None, name=None):
        '''
        Write Wind timeseries to file or to zip,
        then call save method using super
        '''
        name = (name, 'Wind.json')[name is None]
        ts_name = os.path.splitext(name)[0] + '_data.WND'

        if zipfile.is_zipfile(saveloc):
            self._write_timeseries_to_zip(saveloc, ts_name)
            self._filename = ts_name
        else:
            datafile = os.path.join(saveloc, ts_name)
            self._write_timeseries_to_file(datafile)
            self._filename = datafile

        return super(Wind, self).save(saveloc, references, name)

    def _write_timeseries_to_zip(self, saveloc, ts_name):
        '''
        use a StringIO type of file descriptor and write directly to zipfile
        '''
        fd = StringIO.StringIO()
        self._write_timeseries_to_fd(fd)
        self._write_to_zip(saveloc, ts_name, fd.getvalue())

    def _write_timeseries_to_file(self, datafile):
        '''write timeseries data to file '''
        with open(datafile, 'w') as fd:
            self._write_timeseries_to_fd(fd)

    def _write_timeseries_to_fd(self, fd):
        '''
        Takes a general file descriptor as input and writes data to it.

        Writes the "OSSM format" with the full header
        '''
        if self.units in ossm_wind_units.values():
            data_units = self.units
        else:
            # we know C++ understands this unit
            data_units = 'meters per second'

        header = ('Station Name\n'
                  'Position\n'
                  '{0}\n'
                  'LTime\n'
                  '0,0,0,0,0,0,0,0\n').format(data_units)

        data = self.get_wind_data(units=data_units)
        val = data['value']
        dt = data['time'].astype(datetime.datetime)

        fd.write(header)

        for i, idt in enumerate(dt):
            fd.write('{0.day:02}, '
                     '{0.month:02}, '
                     '{0.year:04}, '
                     '{0.hour:02}, '
                     '{0.minute:02}, '
                     '{1:02.2f}, {2:02.2f}\n'
                     .format(idt,
                             round(val[i, 0], 4),
                             round(val[i, 1], 4)))

    def update_from_dict(self, data):
        '''
        update attributes from dict - override base class because we want to
        set the units before updating the data so conversion is done correctly.
        Internally all data is stored in SI units.
        '''
        updated = self.update_attr('units', data.pop('units', self.units))

        if super(Wind, self).update_from_dict(data):
            return True
        else:
            return updated

    def get_wind_data(self, datetime=None, units=None, coord_sys='r-theta'):
        """
        Returns the timeseries in the requested coordinate system.
        If datetime=None, then the original timeseries that was entered is
        returned.
        If datetime is a list containing datetime objects, then the value
        for each of those date times is determined by the underlying
        C++ object and the timeseries is returned.

        The output coordinate system is defined by the strings 'r-theta', 'uv'

        :param datetime: [optional] datetime object or list of datetime
                         objects for which the value is desired
        :type datetime: datetime object

        :param units: [optional] outputs data in these units. Default is to
            output data without unit conversion
        :type units: string. Uses the unit_conversion module.

        :param coord_sys: output coordinate system for the times series:
                          either 'r-theta' or 'uv'
        :type coord_sys: either string or integer value defined by
                         basic_types.ts_format.* (see cy_basic_types.pyx)

        :returns: numpy array containing dtype=basic_types.datetime_value_2d.
                  Contains user specified datetime and the corresponding
                  values in user specified ts_format

        .. note:: Invokes self._convert_units() to do the unit conversion.
            Override this method to define the derived object's unit conversion
            functionality

        todo: return data in appropriate significant digits
        """
        datetimeval = super(Wind, self).get_timeseries(datetime, coord_sys)
        units = (units, self._user_units)[units is None]

        datetimeval['value'] = self._convert_units(datetimeval['value'],
                                                   coord_sys,
                                                   'meter per second',
                                                   units)

        return datetimeval

    def set_wind_data(self, wind_data, units, coord_sys='r-theta'):
        """
        Sets the timeseries of the Wind object to the new value given by
        a numpy array.  The coordinate system for the input data defaults to
        basic_types.format.magnitude_direction but can be changed by the user.
        Units are also required with the data.

        :param datetime_value_2d: timeseries of wind data defined in a
                                  numpy array
        :type datetime_value_2d: numpy array of dtype
                                 basic_types.datetime_value_2d

        :param units: units associated with the data. Valid units defined in
                      Wind.valid_vel_units list

        :param coord_sys: output coordinate system for the times series,
                          as defined by basic_types.format.
        :type coord_sys: either string or integer value defined by
                         basic_types.format.* (see cy_basic_types.pyx)
        """
        if self._check_timeseries(wind_data):
            self._check_units(units)
            self.units = units

            wind_data = self._xform_input_timeseries(wind_data)
            wind_data['value'] = self._convert_units(wind_data['value'],
                                                     coord_sys, units,
                                                     'meter per second')

            super(Wind, self).set_timeseries(wind_data, coord_sys)
        else:
            raise ValueError('Bad timeseries as input')

    def get_value(self, time):
        '''
        Return the value at specified time and location. Wind timeseries are
        independent of location; however, a gridded datafile may require
        location so this interface may get refactored if it needs to support
        different types of wind data. It returns the data in SI units (m/s)
        in 'r-theta' coordinate system (speed, direction)

        :param time: the time(s) you want the data for
        :type time: datetime object or sequence of datetime objects.

        .. note:: It invokes get_wind_data(..) function
        '''
        data = self.get_wind_data(time, 'm/s', 'r-theta')

        return tuple(data[0]['value'])

    def at(self, points, time, coord_sys='r-theta',
           _auto_align=True):
        '''
        Returns the value of the wind at the specified points at the specified
        time. Valid coordinate systems include 'r-theta', 'r', 'theta',
        'uv', 'u' or 'v'. This function is for API compatibility with the new
        environment objects.

        :param points: Nx2 or Nx3 array of positions (lon, lat, [z]).
                       This may not be None. To get wind values
                       position-independently, use get_value(time)
        :param time: Datetime of the time to be queried

        :param coord_sys: String describing the coordinate system.
        '''
        if points is None:
            points = np.array((0, 0)).reshape(-1, 2)

        pts = gridded.utilities._reorganize_spatial_data(points)

        ret_data = np.zeros_like(pts, dtype='float64')

        if coord_sys in ('r-theta', 'uv'):
            data = self.get_wind_data(time, 'm/s', coord_sys)[0]['value']
            ret_data[:, 0] = data[0]
            ret_data[:, 1] = data[1]
        elif coord_sys in ('u', 'v', 'r', 'theta'):
            if coord_sys in ('u', 'v'):
                f = 'uv'
            else:
                f = 'r-theta'

            data = self.get_wind_data(time, 'm/s', f)[0]['value']
            if coord_sys in ('u', 'r'):
                ret_data[:, 0] = data[0]
                ret_data = ret_data[:, 0]
            else:
                ret_data[:, 1] = data[1]
                ret_data = ret_data[:, 1]
        else:
            raise ValueError('invalid coordinate system {0}'.format(coord_sys))

        if _auto_align:
            ret_data = (gridded.utilities
                        ._align_results_to_spatial_data(ret_data, points))

        return ret_data

    def set_speed_uncertainty(self, up_or_down=None):
        '''
        This function shifts the wind speed values in our time series
        based on a single parameter Rayleigh distribution method,
        and scaled by a value in the range [0.0 ... 0.5].
        This range represents a plus-or-minus percent of uncertainty that
        the distribution function should calculate

        For each wind value in our time series:

        * We assume it to be the average speed for that sample time
        * We calculate its respective Rayleigh distribution mode (sigma).
        * We determine either an upper percent uncertainty or a
          lower percent uncertainty based on a passed in parameter.
        * Using the Rayleigh Quantile method and our calculated percent,
          we determine the wind speed that is just at or above the
          fractional area under the Probability distribution.
        * We assign the wind speed to its new calculated value.

        Since we are irreversibly changing the wind speed values,
        we should probably do this only once.
        '''
        if up_or_down not in ('up', 'down'):
            return False

        if (self.speed_uncertainty_scale <= 0.0 or
                self.speed_uncertainty_scale > 0.5):
            return False
        else:
            percent_uncertainty = self.speed_uncertainty_scale

        time_series = self.get_wind_data()

        for tse in time_series:
            sigma = rayleigh.sigma_from_wind(tse['value'][0])

            if up_or_down == 'up':
                tse['value'][0] = rayleigh.quantile(0.5 + percent_uncertainty,
                                                    sigma)
            elif up_or_down == 'down':
                tse['value'][0] = rayleigh.quantile(0.5 - percent_uncertainty,
                                                    sigma)

        self.set_wind_data(time_series, self.units)

        return True

    def __eq__(self, other):
        '''
        invoke super to check equality for all 'save' parameters. Also invoke
        __eq__ for Timeseries object to check equality of timeseries. Super
        is not used in any of the __eq__ methods
        '''
        if not super(Wind, self).__eq__(other):
            return False

        if not Timeseries.__eq__(self, other):
            return False

        return True

    def validate(self):
        '''
        only issues warning - object is always valid
        '''
        msgs = []
        if np.all(self.timeseries['value'][:, 0] == 0.0):
            msg = 'wind speed is 0'
            self.logger.warning(msg)

            msgs.append(self._warn_pre + msg)

        return (msgs, True)


def constant_wind(speed, direction, units='m/s'):
    """
    utility to create a constant wind "timeseries"

    :param speed: speed of wind
    :param direction: direction -- degrees True, direction wind is from
                      (degrees True)
    :param unit='m/s': units for speed, as a string, i.e. "knots", "m/s",
                       "cm/s", etc.

    .. note::
        The time for a constant wind timeseries is irrelevant. This
        function simply sets it to datetime.now() accurate to hours.
    """
    wind_vel = np.zeros((1, ), dtype=datetime_value_2d)

    # just to have a time accurate to minutes
    wind_vel['time'][0] = datetime.datetime.now().replace(microsecond=0,
                                                          second=0,
                                                          minute=0)
    wind_vel['value'][0] = (speed, direction)

    return Wind(timeseries=wind_vel, coord_sys='r-theta', units=units)


def wind_from_values(values, units='m/s'):
    """
    creates a Wind object directly from data.

    :param values: list of (datetime, speed, direction) tuples

    :returns: A Wind timeseries object that can be used for a wind mover, etc.
    """
    wind_vel = np.zeros((len(values), ), dtype=datetime_value_2d)

    for i, record in enumerate(values):
        wind_vel['time'][i] = record[0]
        wind_vel['value'][i] = tuple(record[1:3])

    return Wind(timeseries=wind_vel, coord_sys='r-theta', units=units)
