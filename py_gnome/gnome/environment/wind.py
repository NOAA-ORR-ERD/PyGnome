"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""

import datetime
import os
import copy
import StringIO
import zipfile

import numpy
np = numpy

from colander import (SchemaNode, drop, OneOf,
                      Float, String, Range)
import unit_conversion as uc

from gnome import basic_types

from gnome.utilities import serializable

from gnome.utilities.distributions import RayleighDistribution as rayleigh

from gnome.persist.extend_colander import (DefaultTupleSchema,
                                           LocalDateTime,
                                           DatetimeValue2dArraySchema)
from gnome.persist import validators, base_schema

from .environment import Environment
from gnome.utilities.timeseries import Timeseries
from gnome.cy_gnome.cy_ossm_time import ossm_wind_units
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


class WindSchema(base_schema.ObjType):
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
                             validator=OneOf(basic_types.wind_datasource._attr),
                             default='undefined', missing='undefined')
    units = SchemaNode(String(), default='m/s')
    speed_uncertainty_scale = SchemaNode(Float(), missing=drop)

    timeseries = WindTimeSeriesSchema(missing=drop)
    name = 'wind'


class Wind(serializable.Serializable, Timeseries, Environment):
    '''
    Defines the Wind conditions for a spill
    '''
    # removed 'id' from list below. id, filename and units cannot be updated
    # - read only properties

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
                      # test for equality of units a little differently
                      serializable.Field('units', save=False,
                                         update=True, test_for_eq=False),
                      ])
    _state['name'].test_for_eq = False

    # list of valid velocity units for timeseries
    valid_vel_units = _valid_units('Velocity')

    def __init__(self, timeseries=None, units=None,
                 filename=None, format='r-theta',
                 latitude=None, longitude=None,
                 speed_uncertainty_scale=0.0,
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

        if filename is not None:
            self.source_type = kwargs.pop('source_type', 'file')
            super(Wind, self).__init__(filename=filename, format=format)
            self.name = kwargs.pop('name', os.path.split(self.filename)[1])
            # set _user_units attribute to match user_units read from file.
            self._user_units = self.ossm.user_units

            if units is not None:
                self.units = units
        else:
            if kwargs.get('source_type') in basic_types.wind_datasource._attr:
                self.source_type = kwargs.pop('source_type')
            else:
                self.source_type = 'undefined'

            # either timeseries is given or nothing is given
            # create an empty default object
            super(Wind, self).__init__()

            self.units = 'mps'  # units for default object
            if timeseries is not None:
                if units is None:
                    raise TypeError('Units must be provided with timeseries')

                self.set_wind_data(timeseries, units, format)

            self.name = kwargs.pop('name', self.__class__.__name__)

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
                'timeseries={1}'
                ')').format(self, self_ts)

    # user_units = property( lambda self: self._user_units)

    @property
    def timeseries(self):
        '''
        returns entire timeseries in 'r-theta' format in the units in which
        the data was entered or as specified by units attribute
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

    def _convert_units(self, data, ts_format, from_unit, to_unit):
        '''
        method to convert units for the 'value' stored in the
        date/time value pair
        '''
        if from_unit != to_unit:
            data[:, 0] = uc.convert('Velocity', from_unit, to_unit, data[:, 0])

            if ts_format == basic_types.ts_format.uv:
                # TODO: avoid clobbering the 'ts_format' namespace
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
                     '{1:02.2f}, {2:02.2f}\n'.format(idt,
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

    def get_wind_data(self, datetime=None, units=None, format='r-theta'):
        """
        Returns the timeseries in the requested format. If datetime=None,
        then the original timeseries that was entered is returned.
        If datetime is a list containing datetime objects, then the value
        for each of those date times is determined by the underlying
        C++ object and the timeseries is returned.

        The output format is defined by the strings 'r-theta', 'uv'

        :param datetime: [optional] datetime object or list of datetime
                         objects for which the value is desired
        :type datetime: datetime object
        :param units: [optional] outputs data in these units. Default is to
            output data without unit conversion
        :type units: string. Uses the unit_conversion module.
        :param format: output format for the times series:
                       either 'r-theta' or 'uv'
        :type format: either string or integer value defined by
                      basic_types.ts_format.* (see cy_basic_types.pyx)

        :returns: numpy array containing dtype=basic_types.datetime_value_2d.
                  Contains user specified datetime and the corresponding
                  values in user specified ts_format

        .. note:: Invokes self._convert_units() to do the unit conversion.
            Override this method to define the derived object's unit conversion
            functionality
        """
        datetimeval = super(Wind, self).get_timeseries(datetime, format)
        units = (units, self._user_units)[units is None]

        datetimeval['value'] = self._convert_units(datetimeval['value'],
                                                   format,
                                                   'meter per second',
                                                   units)

        return datetimeval

    def set_wind_data(self, datetime_value_2d, units, format='r-theta'):
        """
        Sets the timeseries of the Wind object to the new value given by
        a numpy array.  The format for the input data defaults to
        basic_types.format.magnitude_direction but can be changed by the user.
        Units are also required with the data.

        :param datetime_value_2d: timeseries of wind data defined in a
                                  numpy array
        :type datetime_value_2d: numpy array of dtype
                                 basic_types.datetime_value_2d
        :param units: units associated with the data. Valid units defined in
                      Wind.valid_vel_units list
        :param format: output format for the times series; as defined by
                       basic_types.format.
        :type format: either string or integer value defined by
                      basic_types.format.* (see cy_basic_types.pyx)
        """
        self._check_units(units)
        self.units = units
        datetime_value_2d = self._xform_input_timeseries(datetime_value_2d)
        datetime_value_2d['value'] = \
            self._convert_units(datetime_value_2d['value'],
                                format, units, 'meter per second')
        super(Wind, self).set_timeseries(datetime_value_2d, format)

    def get_value(self, time):
        '''
        Return the value at specified time and location. Wind timeseries are
        independent of location; however, a gridded datafile may require
        location so this interface may get refactored if it needs to support
        different types of wind data. It returns the data in SI units (m/s)
        in 'r-theta' format (speed, direction)

        :param time: the time(s) you want the data for
        :type time: datetime object or sequence of datetime objects.

        .. note:: It invokes get_wind_data(..) function
        '''
        data = self.get_wind_data(time, 'm/s', 'r-theta')
        return tuple(data[0]['value'])

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


def constant_wind(speed, direction, units='m/s'):
    """
    utility to create a constant wind "timeseries"

    :param speed: speed of wind
    :param direction: direction -- degrees True, direction wind is from
                      (degrees True)
    :param unit='m/s': units for speed, as a string, i.e. "knots", "m/s",
                       "cm/s", etc.

    .. note:: The time for a constant wind timeseries is irrelevant. This
    function simply sets it to datetime.now() accurate to hours.
    """
    wind_vel = np.zeros((1, ), dtype=basic_types.datetime_value_2d)

    # just to have a time accurate to minutes
    wind_vel['time'][0] = datetime.datetime.now().replace(microsecond=0,
                                                          second=0,
                                                          minute=0)
    wind_vel['value'][0] = (speed, direction)

    return Wind(timeseries=wind_vel, format='r-theta', units=units)
