
import datetime

import numpy as np

from gnome.cy_gnome.cy_ossm_time import CyTimeseries

from gnome import basic_types, GnomeId
from gnome.utilities.time_utils import (zero_time,
                                        date_to_sec,
                                        sec_to_date)
from gnome.utilities.convert import (to_time_value_pair,
                                     tsformat,
                                     to_datetime_value_2d)
from gnome.persist.base_schema import ObjTypeSchema


class TimeseriesError(Exception):
    """
    Error class for a problem with the timeseries check
    """
    pass


class Timeseries(GnomeId):
    _schema = ObjTypeSchema

    def __init__(self, timeseries=None, filename=None, coord_sys='uv',
                 extrapolation_is_allowed=False, **kwargs):
        """
        Initializes a timeseries object from either a timeseries or datafile
        containing the timeseries. If both timeseries and file are given,
        it will read data from the file

        If neither are given, timeseries gets initialized as:

            timeseries = np.zeros((1,), dtype=basic_types.datetime_value_2d)

        If user provides timeseries, the default coordinate system is 'uv'.
        The C++ stores the data in 'uv' coordinates - transformations are done
        in this Python code (set_timeseries(), get_timeseries()).

        C++ code only transforms the data from 'r-theta' to 'uv' coordinates
        if data is read from file. And this happens during initialization
        because C++ stores data in 'uv' coordinates internally.

        Units option are not included - let derived classes manage units since
        the units for CyTimeseries (OSSMTimeValue_c) are limited. No unit
        conversion is performed when get_timeseries, set_timeseries is invoked.
        It does, however convert between 'uv' and 'r-theta' depending on the
        coordinate system specified. Choose coord_sys='uv' if no transformation
        is desired.

        .. note:: For the Wind datafiles, the units will get read from the
           file. These are stored in ossm.user_units. It would be ideal to remove
           units and unit conversion from here, but can't completely do away with
           it since C++ file reading uses/sets it. But, managing units is
           responsibility of derived objects.

        All other keywords are optional

        :param timeseries: numpy array containing time_value_pair
        :type timeseries: numpy.ndarray containing
                          basic_types.datetime_value_2d or
                          basic_types.datetime_value_1d.
                          It gets converted to an array containing
                          basic_types.time_value_pair datatype since that's
                          what the C++ code expects
        :param filename: path to a timeseries file from which to read data.
                         Datafile must contain either a 3 line or a 5 line
                         header with following info:

                         1. Station Name: name of the station as a string
                         2. (long, lat, z): station location as tuple
                            containing floats
                         3. units: for wind this is knots, meteres per second
                            or miles per hour. For datafile containing
                            something other than velocity, this should be
                            'undefined'

        Optional parameters (kwargs):

        :param coord_sys: (Optional) default timeseries coordinate system is
                          magnitude direction: 'r-theta'
        :type coord_sys: string 'r-theta' or 'uv'. Default is 'r-theta'.
            Converts string to integer defined by
            gnome.basic_types.ts_format.*
        """
        super(Timeseries, self).__init__(**kwargs)
        if (timeseries is None and filename is None):
            timeseries = np.array([(sec_to_date(zero_time()), [0.0, 0.0])],
                                  dtype=basic_types.datetime_value_2d)

        self._filename = filename

        if filename is None:
            # will raise an Exception if it fails
            self._check_timeseries(timeseries)

            datetime_value_2d = self._xform_input_timeseries(timeseries)
            time_value_pair = to_time_value_pair(datetime_value_2d, coord_sys)

            self.ossm = CyTimeseries(timeseries=time_value_pair)
        else:
            ts_format = tsformat(coord_sys)
            self.ossm = CyTimeseries(filename=filename,
                                     file_format=ts_format)

        self.extrapolation_is_allowed = extrapolation_is_allowed

    def _check_timeseries(self, timeseries):
        """
        Run some checks to make sure timeseries is valid.

        - We accept numpy arrays

        - We accept a list of timeseries values of the form

          (datetime(...), (N, N))

        - We also accept a constant single timeseries value.

        """
        if not isinstance(timeseries, np.ndarray):
            if self._is_timeseries_value(timeseries):
                return True
            else:
                for i in timeseries:
                    if not self._is_timeseries_value(i):
                        raise TimeseriesError('value: {} '
                                              'is not a timeseries value'
                                              .format(i))
                return True

        if not self._timeseries_is_ascending(timeseries):
            self.logger.error('{0} - timeseries are not in ascending order. '
                              'The datetime values in the array must be in '
                              'ascending order'
                              .format(self._pid))
            raise TimeseriesError("timeseries is not in ascending order")

        return True

    def __len__(self):
        """
        length is the number of data points in the timeseries
        """
        return self.ossm.get_num_values()

    def __eq__(self, o):
        """
        """
        t1 = super(Timeseries, self).__eq__(o)
        return t1 and hasattr(self, 'ossm') and np.all(self.ossm.timeseries == o.ossm.timeseries)

    def get_start_time(self):
        """
        :this will be the real_data_start time (seconds).
        """
        return self.ossm.get_start_time()

    def get_end_time(self):
        """
        :this will be the real_data_stop time (seconds).
        """
        return (self.ossm.get_end_time())

    def _is_timeseries_value(self, value):
        if not isinstance(value, (list, tuple)):
            return False

        if len(value) != 2:
            return False

        if not isinstance(value[0], (datetime.datetime, np.datetime64)):
            return False

        if len(value[1]) not in (1, 2):
            return False

        return True

    def _timeseries_is_ascending(self, timeseries):
        """
        Check if values are monotonically increasing

        This should catch both out of order and duplicate values.
        """

        # FixMe: does this ever have to work for a true scalar??
        if timeseries.shape == () or timeseries.shape == (1,):
            # scalar or single value -- must be OK
            return True

        if np.any(np.diff(timeseries['time']) <= np.timedelta64(0, 's')):
            return False
        else:
            return True

    def _xform_input_timeseries(self, timeseries):
        '''
        Ensure input data is numpy array with correct dtype and check
        timeseries doesn't have invalid data
        Derived classes can use this before updating timeseries, prior to
        converting units.
        '''
        # following fails for 0-d objects so make sure we have a 1-D array
        # to work with
        timeseries = np.asarray(timeseries,
                                dtype=basic_types.datetime_value_2d)

        if timeseries.shape == ():
            timeseries = np.asarray([timeseries],
                                    dtype=basic_types.datetime_value_2d)

        return timeseries

    def __str__(self):
        return '{0.__module__}.{0.__class__.__name__}'.format(self)

    @property
    def filename(self):
        return self._filename

    @property
    def extrapolation_is_allowed(self):
        return self.ossm.extrapolation_is_allowed

    @extrapolation_is_allowed.setter
    def extrapolation_is_allowed(self, value):
        self.ossm.extrapolation_is_allowed = value

    def get_timeseries(self, datetime=None, coord_sys='uv'):
        """
        Returns the timeseries in requested coordinate system.
        If datetime=None, then the original timeseries that was entered is
        returned.
        If datetime is a list containing datetime objects, then the value
        for each of those date times is determined by the underlying
        C++ object and the timeseries is returned.

        The output coordinate system is defined by the strings 'r-theta', 'uv'

        :param datetime: [optional] datetime object or list of datetime
                         objects for which the value is desired
        :type datetime: datetime object

        :param coord_sys: output coordinate system for the times series:
                          either 'r-theta' or 'uv'
        :type coord_sys: either string or integer value defined by
                         basic_types.ts_format.* (see cy_basic_types.pyx)

        :returns: numpy array containing dtype=basic_types.datetime_value_2d.
                  Contains user specified datetime and the corresponding
                  values in user specified ts_format
        """
        if datetime is None:
            datetimeval = to_datetime_value_2d(self.ossm.timeseries, coord_sys)
        else:
            datetime = np.asarray(datetime, dtype='datetime64[s]').reshape(-1)
            timeval = np.zeros((len(datetime), ),
                               dtype=basic_types.time_value_pair)
            timeval['time'] = date_to_sec(datetime)
            (timeval['value'], err) = self.ossm.get_time_value(timeval['time'])

            if err != 0:
                msg = ('No available data in the time interval that is being '
                       'modeled\n'
                       '\tModel time: {}\n'
                       '\tMover: {} of type {}\n'
                       .format(datetime, self.name, self.__class__))

                self.logger.error(msg)
                raise RuntimeError(msg)

            datetimeval = to_datetime_value_2d(timeval, coord_sys)

        return np.copy(datetimeval)

    def set_timeseries(self, datetime_value_2d, coord_sys='uv'):
        """
        Sets the timeseries to the new value given by a numpy array.  The
        coordinate system for the input data defaults to
        basic_types.format.magnitude_direction but can be changed by the user.

        Assumes timeseries is valid so _check_timeseries has been invoked
        and any unit conversions are done. This function simply converts
        datetime_value_2d to time_value_pair and updates the data in underlying
        cython/C++ object

        :param datetime_value_2d: timeseries of wind data defined in a
            numpy array
        :type datetime_value_2d: numpy array of dtype
            basic_types.datetime_value_2d

        :param coord_sys: output coordinate system for the times series,
                          as defined by basic_types.ts_format.
        :type coord_sys: either string or integer value defined by
                         basic_types.ts_format.* (see cy_basic_types.pyx)
        """
        datetime_value_2d = self._xform_input_timeseries(datetime_value_2d)
        timeval = to_time_value_pair(datetime_value_2d, coord_sys)

        self.ossm.timeseries = timeval