"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""

import datetime
import os
import copy
from itertools import chain
import tempfile

import numpy
np = numpy

from colander import (SchemaNode, drop, OneOf,
                      Float, String, Range)

from .environment import Environment
from gnome.persist.extend_colander import (DefaultTupleSchema,
                                           LocalDateTime,
                                           DatetimeValue2dArraySchema)
from gnome.persist import validators, base_schema

#import gnome
from gnome import basic_types

# TODO: The name 'convert' is doubly defined as
#       hazpy.unit_conversion.convert and...
#       gnome.utilities.convert
#       This will inevitably cause namespace collisions.
from hazpy import unit_conversion
from gnome.utilities.time_utils import sec_to_date, date_to_sec
from hazpy.unit_conversion import (ConvertDataUnits,
                                   GetUnitNames,
                                   InvalidUnitError)

from gnome.utilities import serializable
from gnome.utilities.convert import (to_time_value_pair,
                                     tsformat,
                                     to_datetime_value_2d)

from gnome.cy_gnome.cy_ossm_time import CyOSSMTime


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

    timeseries = WindTimeSeriesSchema(missing=drop)
    name = 'wind'


class Wind(Environment, serializable.Serializable):
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
               'updated_at']

    # used to create new obj or as readonly parameter
    _create = []
    _create.extend(_update)

    _state = copy.deepcopy(Environment._state)
    _state.add(save=_create, update=_update)
    _schema = WindSchema

    # add 'filename' as a Field object
    #'name',    is this for webgnome?
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
    valid_vel_units = list(chain.from_iterable([item[1] for item in
                                    ConvertDataUnits['Velocity'].values()]))
    valid_vel_units.extend(GetUnitNames('Velocity'))

    def __init__(self, timeseries=None, units=None,
                 filename=None, format='r-theta',
                 latitude=None, longitude=None,
                 **kwargs):
        """
        Initializes a wind object from timeseries or datafile
        If both are given, it will read data from the file

        If neither are given, timeseries gets initialized as:

            timeseries = np.zeros((1,), dtype=basic_types.datetime_value_2d)
            units = 'mps'

        If user provides timeseries, they *must* also provide units

        All other keywords are optional. Optional parameters (kwargs):

        :param timeseries: numpy array containing time_value_pair
        :type timeseries: numpy.ndarray[basic_types.time_value_pair, ndim=1]
        :param filename: path to a long wind file from which to read wind data
        :param units: units associated with the timeseries data. If 'filename'
            is given, then units are read in from the file.
            get_timeseries() will use these as default units to
            output data, unless user specifies otherwise.
            These units must be valid as defined in the hazpy
            unit_conversion module:
            unit_conversion.GetUnitNames('Velocity')
        :type units:  string, for example: 'knot', 'meter per second',
            'mile per hour' etc.
            Default units for input/output timeseries data
        :param format: (Optional) default timeseries format is
            magnitude direction: 'r-theta'
        :type format: string 'r-theta' or 'uv'. Default is 'r-theta'.
            Converts string to integer defined by
            gnome.basic_types.ts_format.*
            TODO: 'format' is a python builtin keyword.  We should
            not use it as an argument name
        :param name: (Optional) human readable string for wind object name.
            Default is filename if data is from file or "Wind Object"
        :param source_type: (Optional) Default is undefined, but can be one of
            the following: ['buoy', 'manual', 'undefined', 'file', 'nws']
            If data is read from file, then it is 'file'
        :param latitude: (Optional) latitude of station or location where
            wind data is obtained from NWS
        :param longitude: (Optional) longitude of station or location where
            wind data is obtained from NWS

        """
        if (timeseries is None and filename is None):
            timeseries = np.zeros((1,), dtype=basic_types.datetime_value_2d)
            units = 'mps'

        self._filename = None

        if not filename:
            time_value_pair = self._convert_to_time_value_pair(timeseries,
                units, format)

            # this has same scope as CyWindMover object
            #
            # TODO: move this as a class attribute if we can.
            #       I can see that we are instantiating the class,
            #       but maybe we can find a way to not have to
            #       pickle this attribute when we pickle a Wind instance
            #
            self.ossm = CyOSSMTime(timeseries=time_value_pair)
            self._user_units = units
            self.source_type = (kwargs.pop('source_type')
                                if kwargs.get('source_type')
                                in basic_types.wind_datasource._attr
                                else 'undefined')
            self.name = kwargs.pop('name', self.__class__.__name__)
        else:
            ts_format = tsformat(format)
            self._filename = filename
            self.ossm = CyOSSMTime(filename=self._filename,
                                   file_contains=ts_format)
            self._user_units = self.ossm.user_units

            self.source_type = 'file'  # this must be file
            self.name = kwargs.pop('name', os.path.split(self.filename)[1])

        self.updated_at = kwargs.pop('updated_at', None)
        self.source_id = kwargs.pop('source_id', 'undefined')
        self.longitude = longitude
        self.latitude = latitude
        self.description = kwargs.pop('description', 'Wind Object')
        super(Wind, self).__init__(**kwargs)

    def _convert_units(self, data, ts_format, from_unit, to_unit):
        '''
        Private method to convert units for the 'value' stored in the
        date/time value pair
        '''
        if from_unit != to_unit:
            data[:, 0] = unit_conversion.convert('Velocity',
                                                 from_unit, to_unit,
                                                 data[:, 0])

            if ts_format == basic_types.ts_format.uv:
                # TODO: avoid clobbering the 'ts_format' namespace
                data[:, 1] = unit_conversion.convert('Velocity',
                                                     from_unit, to_unit,
                                                     data[:, 1])

        return data

    def _check_units(self, units):
        '''
        Checks the user provided units are in list Wind.valid_vel_units
        '''
        if units not in Wind.valid_vel_units:
            raise InvalidUnitError('A valid velocity unit must be one of: '
                                   '{0}'.format(Wind.valid_vel_units))

    def _check_timeseries(self, timeseries, units):
        '''
        Run some checks to make sure timeseries is valid
        Also, make the resolution to minutes as opposed to seconds
        '''
        try:
            if timeseries.dtype != basic_types.datetime_value_2d:
                # Both 'is' or '==' work in this case.  There is only one
                # instance of basic_types.datetime_value_2d.
                # Maybe in future we can consider working with a list,
                # but that's a bit more cumbersome for different dtypes
                raise ValueError('timeseries must be a numpy array containing '
                                 'basic_types.datetime_value_2d dtype')
        except AttributeError, err:
            msg = 'timeseries is not a numpy array. {0}'
            raise AttributeError(msg.format(err.message))

        # check to make sure the time values are in ascending order
        if np.any(timeseries['time'][np.argsort(timeseries['time'])]
                  != timeseries['time']):
            raise ValueError('timeseries are not in ascending order. '
                             'The datetime values in the array must be in '
                             'ascending order')

        # check for duplicate entries
        unique = np.unique(timeseries['time'])
        if len(unique) != len(timeseries['time']):
            raise ValueError('timeseries must contain unique time entries. '
                             'Number of duplicate entries: '
                             '{0}'.format(len(timeseries) - len(unique)))

        # make resolution to minutes in datetime
        for ix, tm in enumerate(timeseries['time'].astype(datetime.datetime)):
            timeseries['time'][ix] = tm.replace(second=0)

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

    def __str__(self):
        return "Wind( timeseries=Wind.get_timeseries('uv'), format='uv')"

    def __eq__(self, other):
        '''
        call super to test for equality of objects for all attributes
        except 'units' and 'timeseries' - test 'timeseries' here by converting
        to consistent units
        '''
        # since this has numpy array - need to compare that as well
        # By default, tolerance for comparison is atol=1e-10, rtol=0
        # persisting data requires unit conversions and finite precision,
        # both of which will introduce a difference between two objects

        check = super(Wind, self).__eq__(other)

        if check:
            sts = self.get_timeseries(units=self.units)
            ots = other.get_timeseries(units=self.units)

            if (sts['time'] != ots['time']).all():
                return False
            else:
                return np.allclose(sts['value'], ots['value'], 0, 1e-2)

        return check

    def __ne__(self, other):
        return not self == other

    # user_units = property( lambda self: self._user_units)

    @property
    def units(self):
        return self._user_units

    @units.setter
    def units(self, value):
        """
        User can set default units for input/output data

        These are given as string and must be one of the units defined in
        unit_conversion.GetUnitNames('Velocity')
        or one of the associated abbreviations
        unit_conversion.GetUnitAbbreviation()
        """
        self._check_units(value)
        self._user_units = value

    filename = property(lambda self: self._filename)
    timeseries = property(lambda self: self.get_timeseries(),
                          lambda self, val: self.set_timeseries(val,
                                                            units=self.units)
                          )

    def _convert_to_time_value_pair(self, datetime_value_2d, units, format):
        '''
        format datetime_value_2d so it is a numpy array with
        dtype=basic_types.time_value_pair as the C++ code expects
        '''
        # following fails for 0-d objects so make sure we have a 1-D array
        # to work with
        datetime_value_2d = np.asarray(datetime_value_2d,
            dtype=basic_types.datetime_value_2d)
        if datetime_value_2d.shape == ():
            datetime_value_2d = np.asarray([datetime_value_2d],
                dtype=basic_types.datetime_value_2d)

        self._check_units(units)
        self._check_timeseries(datetime_value_2d, units)
        datetime_value_2d['value'] = \
            self._convert_units(datetime_value_2d['value'],
                                format, units, 'meter per second')

        timeval = to_time_value_pair(datetime_value_2d, format)
        return timeval

    def get_timeseries(self, datetime=None, units=None, format='r-theta'):
        """
        Returns the timeseries in the requested format. If datetime=None,
        then the original timeseries that was entered is returned.
        If datetime is a list containing datetime objects, then the wind value
        for each of those date times is determined by the underlying
        CyOSSMTime object and the timeseries is returned.

        The output format is defined by the strings 'r-theta', 'uv'

        :param datetime: [optional] datetime object or list of datetime
                         objects for which the value is desired
        :type datetime: datetime object
        :param units: [optional] outputs data in these units. Default is to
                      output data in units
        :type units: string. Uses the hazpy.unit_conversion module.
                     hazpy.unit_conversion throws error for invalid units
        :param format: output format for the times series:
                       either 'r-theta' or 'uv'
        :type format: either string or integer value defined by
                      basic_types.ts_format.* (see cy_basic_types.pyx)

        :returns: numpy array containing dtype=basic_types.datetime_value_2d.
                  Contains user specified datetime and the corresponding
                  values in user specified ts_format
        """
        if datetime is None:
            datetimeval = to_datetime_value_2d(self.ossm.timeseries, format)
        else:
            datetime = np.asarray(datetime, dtype='datetime64[s]').reshape(-1)
            timeval = np.zeros((len(datetime), ),
                               dtype=basic_types.time_value_pair)
            timeval['time'] = date_to_sec(datetime)
            timeval['value'] = self.ossm.get_time_value(timeval['time'])
            datetimeval = to_datetime_value_2d(timeval, format)

        if units is None:
            units = self.units

        datetimeval['value'] = self._convert_units(datetimeval['value'],
                                                   format, 'meter per second',
                                                   units)

        return datetimeval

    def set_timeseries(self, datetime_value_2d, units, format='r-theta'):
        """
        Sets the timeseries of the Wind object to the new value given by
        a numpy array.  The format for the input data defaults to
        basic_types.format.magnitude_direction but can be changed by the user

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
        timeval = self._convert_to_time_value_pair(datetime_value_2d, units, format)
        self.ossm.timeseries = timeval

    def save(self, saveloc, references=None, name=None):
        '''
        Write Wind timeseries to file, then call save method using super
        '''
        name = (name, 'Wind.json')[name is None]
        datafile = os.path.join(saveloc,
                                os.path.splitext(name)[0] + '_data.WND')
        self._write_timeseries_to_file(datafile)
        self._filename = datafile
        return super(Wind, self).save(saveloc, references, name)

    def _write_timeseries_to_file(self, datafile):
        '''write to temp file '''

        header = ('Station Name\n'
                  'Position\n'
                  'knots\n'
                  'LTime\n'
                  '0,0,0,0,0,0,0,0\n')
        val = self.get_timeseries(units='knots')['value']
        dt = self.get_timeseries(units='knots')['time'].astype(datetime.datetime)

        with open(datafile, 'w') as file_:
            file_.write(header)

            for i, idt in enumerate(dt):
                file_.write('{0.day:02}, '
                            '{0.month:02}, '
                            '{0.year:04}, '
                            '{0.hour:02}, '
                            '{0.minute:02}, '
                            '{1:02.4f}, {2:02.4f}\n'
                            .format(idt,
                                    round(val[i, 0], 4),
                                    round(val[i, 1], 4))
                            )
        file_.close()   # just incase we get issues on windows


def constant_wind(speed, direction, units='m/s'):
    """
    utility to create a constant wind "timeseries"

    :param speed: speed of wind
    :param direction: direction -- degrees True, direction wind is from
                      (degrees True)
    :param unit='m/s': units for speed, as a string, i.e. "knots", "m/s",
                       "cm/s", etc.
    """
    wind_vel = np.zeros((1, ), dtype=basic_types.datetime_value_2d)

    # just to have a time accurate to minutes
    wind_vel['time'][0] = datetime.datetime.now().replace(microsecond=0,
                                                          second=0)
    wind_vel['value'][0] = (speed, direction)

    return Wind(timeseries=wind_vel, format='r-theta', units=units)
