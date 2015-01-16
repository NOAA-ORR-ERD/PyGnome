import datetime
import os

import numpy as np

from gnome.cy_gnome.cy_ossm_time import CyTimeseries

from gnome import basic_types
from gnome.utilities.time_utils import date_to_sec
from gnome.utilities.convert import (to_time_value_pair,
                                     tsformat,
                                     to_datetime_value_2d)


class Timeseries(object):
    def __init__(self, timeseries=None,
                 filename=None, format='uv',
                 **kwargs):
        """
        Initializes a timeseries object from either a timeseries or datafile
        containing the timeseries. If both timeseries and file are given,
        it will read data from the file

        If neither are given, timeseries gets initialized as:

            timeseries = np.zeros((1,), dtype=basic_types.datetime_value_2d)

        If user provides timeseries, the default format is 'uv'. The C++
        stores the data in 'uv' format - transformations are done in this
        Python code (set_timeseries(), get_timeseries()).

        C++ code only transforms the data from 'r-theta' to 'uv' format if
        data is read from file. And this happens during initialization because
        C++ stores data in 'uv' format internally.

        Units option are not included - let derived classes manage units since
        the units for CyTimeseries (OSSMTimeValue_c) are limited. No unit
        conversion is performed when get_timeseries, set_timeseries is invoked.
        It does, however convert between 'uv' and 'r-theta' depending on format
        specified. Choose format='uv' if no transformation is desired.

        .. note:: For the Wind datafiles, the units will get read from the
        file. These are stored in ossm.user_units. It would be ideal to remove
        units and unit conversion from here, but can't completely do away with
        it since C++ file reading uses/sets it. But, managing units is
        responsibility of derived objects.

        All other keywords are optional

        :param timeseries: numpy array containing time_value_pair
        :type timeseries: numpy.ndarray containing
            basic_types.datetime_value_2d or basic_types.datetime_value_1d. It
            gets converted to an array containging basic_types.time_value_pair
            datatype since that's what the C++ code expects
        :param filename: path to a timeseries file from which to read data.
            Datafile must contain either a 3 line or a 5 line header with
            following info:

            1. Station Name: name of the station as a string
            2. (long, lat, z): station location as tuple containing floats
            3. units: for wind this is knots, meteres per second
            or miles per hour. For datafile containing something other than
            velocity, this should be 'undefined'

        Optional parameters (kwargs):

        :param format: (Optional) default timeseries format is
            magnitude direction: 'r-theta'
        :type format: string 'r-theta' or 'uv'. Default is 'r-theta'.
            Converts string to integer defined by
            gnome.basic_types.ts_format.*
            TODO: 'format' is a python builtin keyword.  We should
            not use it as an argument name

        Remaining kwargs are passed onto the the next method in the Method
        Resolution Order, so this calls super() at the end.
        """
        if (timeseries is None and filename is None):
            timeseries = np.zeros((1,), dtype=basic_types.datetime_value_2d)

        self._filename = None

        if not filename:
            datetime_value_2d = self._xform_input_timeseries(timeseries)
            time_value_pair = to_time_value_pair(datetime_value_2d, format)
            self.ossm = CyTimeseries(timeseries=time_value_pair)
            self.ossm.user_units = 'undefined'
            self.name = kwargs.pop('name', self.__class__.__name__)
            self.source_type = kwargs.pop('source_type', None)

        else:
            ts_format = tsformat(format)
            self._filename = filename
            self.ossm = CyTimeseries(filename=self._filename,
                                     file_format=ts_format)
            self.source_type = 'file'  # todo: check this must be file
            self.name = kwargs.pop('name', os.path.split(self.filename)[1])

    def _xform_input_timeseries(self, datetime_value_2d):
        '''
        Ensure input data is numpy array with correct dtype and check
        timeseries doesn't have invalid data
        Derived classes can use this before updating timeseries, prior to
        converting units.
        '''
        # following fails for 0-d objects so make sure we have a 1-D array
        # to work with
        datetime_value_2d = np.asarray(datetime_value_2d,
                                       dtype=basic_types.datetime_value_2d)
        if datetime_value_2d.shape == ():
            datetime_value_2d = np.asarray([datetime_value_2d],
                                           dtype=basic_types.datetime_value_2d)
        self._check_timeseries(datetime_value_2d)
        return datetime_value_2d

    def _check_timeseries(self, timeseries):
        '''
        Run some checks to make sure timeseries is valid.
        Also, make the resolution to minutes as opposed to seconds
        todo: update exceptions to logged errors
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

    def __str__(self):
        return '{0.__module__}.{0.__class__.__name__}'.format(self)

    @property
    def filename(self):
        return self._filename

    def get_timeseries(self, datetime=None, format='uv'):
        """
        Returns the timeseries in requested format. If datetime=None,
        then the original timeseries that was entered is returned.
        If datetime is a list containing datetime objects, then the value
        for each of those date times is determined by the underlying
        C++ object and the timeseries is returned.

        The output format is defined by the strings 'r-theta', 'uv'

        :param datetime: [optional] datetime object or list of datetime
                         objects for which the value is desired
        :type datetime: datetime object
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

        return datetimeval

    def set_timeseries(self, datetime_value_2d, format='uv'):
        """
        Sets the timeseries to the new value given by a numpy array.  The
        format for the input data defaults to
        basic_types.format.magnitude_direction but can be changed by the user
        Assumes timeseries is valid so _check_timeseries has been invoked
        and any unit conversions are done. This function simply converts
        datetime_value_2d to time_value_pair and updates the data in underlying
        cython/C++ object

        :param datetime_value_2d: timeseries of wind data defined in a
            numpy array
        :type datetime_value_2d: numpy array of dtype
            basic_types.datetime_value_2d
        :param format: output format for the times series; as defined by
                       basic_types.format.
        :type format: either string or integer value defined by
                      basic_types.format.* (see cy_basic_types.pyx)
        """
        timeval = to_time_value_pair(datetime_value_2d, format)
        self.ossm.timeseries = timeval
