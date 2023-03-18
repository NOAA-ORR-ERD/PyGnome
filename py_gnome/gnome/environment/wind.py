"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""

import datetime
import os
# import copy
import io
# import zipfile

import numpy as np

import nucos as uc

import gridded

from gnome.basic_types import datetime_value_2d
from gnome.basic_types import ts_format
from gnome.basic_types import wind_datasources

from gnome.cy_gnome.cy_ossm_time import ossm_wind_units

from gnome.utilities.time_utils import (sec_to_datetime, zero_time, sec_to_date)
from gnome.utilities.timeseries import Timeseries
from gnome.utilities.inf_datetime import InfDateTime
from gnome.utilities.distributions import RayleighDistribution as rayleigh

from gnome.persist import (SchemaNode, drop, OneOf, Float, String, Range,
                           Boolean, DefaultTupleSchema, LocalDateTime,
                           DatetimeValue2dArraySchema, FilenameSchema,
                           validators, base_schema)
from gnome.utilities.convert import (to_time_value_pair,
                                     tsformat,
                                     to_datetime_value_2d)
from gnome.persist.validators import convertible_to_seconds

from .environment import Environment
from gnome.environment.gridded_objects_base import Time, TimeSchema
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
    name = SchemaNode(String(), test_equal=False)
    description = SchemaNode(String())

    # Thanks to CyTimeseries
    filename = FilenameSchema(save=False, missing=drop, isdatafile=False, update=False, test_equal=False)

    updated_at = SchemaNode(LocalDateTime())
    latitude = SchemaNode(Float())
    longitude = SchemaNode(Float())
    source_id = SchemaNode(String())
    source_type = SchemaNode(
        String(),
        validator=OneOf(wind_datasources.__members__.keys()),
        default='undefined',
        missing='undefined'
    )
    units = SchemaNode(String(), default='m/s')
    speed_uncertainty_scale = SchemaNode(Float())

    # Because comparing datetimevalue2d arrays does not play nice
    timeseries = WindTimeSeriesSchema(test_equal=False)

    extrapolation_is_allowed = SchemaNode(Boolean())
    data_start = SchemaNode(LocalDateTime(), read_only=True,
                            validator=convertible_to_seconds)
    data_stop = SchemaNode(LocalDateTime(), read_only=True,
                           validator=convertible_to_seconds)
    time = TimeSchema(
        #this is only for duck-typing the new-style environment objects,
        #so only provide to the client
        save=False, update=False, save_reference=False, read_only=True,
    )


class Wind(Timeseries, Environment):
    ## fixme: rename as "PointWind"
    '''
    Provides the a "point wind" -- uniform wind over all space
    '''
    # object is referenced by others using this attribute name
    _ref_as = 'wind'
    _gnome_unit = 'm/s'

    _schema = WindSchema

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
        Create a PointWind object, representing a time series of wind at a single value

        :param timeseries=None:
        :param units=None:
        :param filename=None:
        :param coord_sys='r-theta':
        :param latitude=None:
        :param longitude=None:
        :param speed_uncertainty_scale=0.0:
        :param extrapolation_is_allowed=False:


        """
        self._timeseries = np.array([(sec_to_date(zero_time()), [0.0, 0.0])], dtype=datetime_value_2d)
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

            super(Wind, self).__init__(filename=filename, coord_sys=coord_sys, **kwargs)

            self.name = kwargs.pop('name', os.path.split(self.filename)[1])
            # set _user_units attribute to match user_units read from file.
            self._user_units = self.ossm.user_units
            self._timeseries = self.get_wind_data(units=self._user_units)

            if units is not None:
                self.units = units
        else:
            if kwargs.get('source_type') in wind_datasources.__members__.keys():
                self.source_type = kwargs.pop('source_type')
            else:
                self.source_type = 'undefined'

            # either timeseries is given or nothing is given
            # create an empty default object
            super(Wind, self).__init__(coord_sys=coord_sys, **kwargs)

            self.units = 'mps'  # units for default object

            if timeseries is not None:
                if units is None:
                    raise TypeError('Units must be provided with timeseries')
                self.units = units
                self.new_set_timeseries(timeseries, coord_sys)

        self.extrapolation_is_allowed = extrapolation_is_allowed

    def update_from_dict(self, dict_, refs=None):
        if 'units' in dict_:
            # enforce updating of units before timeseries
            self.units = dict_.pop('units')
        if 'timeseries' in dict_:
            self.timeseries = WindTimeSeriesSchema().deserialize(dict_.pop('timeseries'))
        super(Wind, self).update_from_dict(dict_, refs=refs)

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
    def time(self):
        #This duck-types the API of the timeseries_objecs_base.TimeseriesVector object
        if not hasattr(self, '_time') and hasattr(self, '_timeseries') and self._timeseries is not None:
            self._time = Time()
            self._time.data = self._timeseries['time'].astype(datetime.datetime)

        return self._time

    @time.setter
    def time(self, t):
        if t is None:
            self._time = None
            return
        if self.timeseries is not None and len(t) != len(self.timeseries):
            warnings.warn("Data/time interval mismatch, doing nothing")
            return

        if isinstance(t, Time):
            self._time = t
        elif isinstance(t, collections.Iterable):
            self._time = Time(t)
        else:
            raise ValueError('Object being assigned must be an iterable '
                             'or a Time object')
    @property
    def timeseries(self):
        '''
        returns entire timeseries in 'r-theta' coordinate system in the units
        in which the data was entered or as specified by units attribute
        '''
        return self._timeseries

    @timeseries.setter
    def timeseries(self, value):
        '''
        set the timeseries for wind. The units for value are as specified by
        self.units attribute. Property converts the units to 'm/s' so Cython/
        C++ object stores timeseries in 'm/s'
        '''
        coord_sys ='r-theta'
        self.new_set_timeseries(value, coord_sys)

    def new_set_timeseries(self, value, coord_sys):
        if self._check_timeseries(value):
            units = self.units

            wind_data = self._xform_input_timeseries(value)
            self._timeseries = wind_data.copy()
            wind_data['value'] = self._convert_units(wind_data['value'],
                                                     coord_sys, units,
                                                     'meter per second')

            datetime_value_2d = self._xform_input_timeseries(wind_data)
            timeval = to_time_value_pair(wind_data, coord_sys)
            self.ossm.timeseries = timeval
            if not hasattr(self, '_time') or self._time is None:
                self._time = Time()
            self.time.data = self._timeseries['time'].astype(datetime.datetime)
        else:
            raise ValueError('Bad timeseries as input')


    @property
    def data_start(self):
        """
        The start time of the valid data for this wind timeseries
        """
        return sec_to_datetime(self.ossm.get_start_time())

    @property
    def data_stop(self):
        """
        The stop time of the valid data for this wind timeseries
        """
        return sec_to_datetime(self.ossm.get_end_time())

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

            if coord_sys == ts_format.uv:
                data[:, 1] = uc.convert('Velocity', from_unit, to_unit,
                                        data[:, 1])

        return data

#     def save(self, saveloc, references=None, filename=None, overwrite=True):
#         '''
#         Write Wind timeseries to file or to zip,
#         then call save method using super
#         '''
#         json_, zipfile_, refs = super(Wind, self).save(saveloc, references, overwrite=overwrite)
#         import pdb
#         pdb.set_trace()
#         filename = (filename, self.name + '.json')[filename is None]
#         ts_name = os.path.splitext(filename)[0] + '_data.WND'
#
#         if zipfile.is_zipfile(saveloc):
#             self._write_timeseries_to_zip(saveloc, ts_name)
#             self._filename = ts_name
#         else:
#             datafile = os.path.join(saveloc, ts_name)
#             self._write_timeseries_to_file(datafile)
#             self._filename = datafile


    def _write_timeseries_to_zip(self, saveloc, ts_name):
        '''
        use a StringIO type of file descriptor and write directly to zipfile
        '''
        fd = io.StringIO()
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
        :type units: string. Uses the nucos module.

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
            self._timeseries = wind_data
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

    def at(self, points, time, coord_sys='uv', units=None,
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

class PointWind(Wind):
    """
    Wind at a single point
    """
    # currently an alias for Wind -- but we should probably swap those and
    # make this the real class, and the `Wind` be the alias.
    pass

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

    return PointWind(timeseries=wind_vel, coord_sys='r-theta', units=units)


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
