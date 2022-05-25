import warnings
import copy
from numbers import Number
from collections import abc

import numpy as np

import nucos as uc

import gridded

from gnome.persist import (ObjTypeSchema, SchemaNode, String, drop,
                           SequenceSchema,NumpyArraySchema)

from gnome.environment.gridded_objects_base import Time, TimeSchema
from gnome.gnomeobject import GnomeId

from gridded.utilities import _align_results_to_spatial_data, _reorganize_spatial_data


class TimeseriesDataSchema(ObjTypeSchema):
    units = SchemaNode(
        String(), missing=drop, save=True, update=True
    )
    time = TimeSchema(
        save=True, update=True, save_reference=True
    )
    data = NumpyArraySchema(
        missing=drop, save=True, update=True
    )


class TimeseriesData(GnomeId):
    '''
    Base class for data with a single dimension: time
    '''

    _schema = TimeseriesDataSchema

    _gnome_unit = None

    def __init__(self,
                 data=None,
                 time=None,
                 units=None,
                 extrapolate=True,
                 *args,
                 **kwargs):
        '''
            A class that represents a natural phenomenon and provides
            an interface to get the value of the phenomenon at a position
            in space and time.
            EnvProp is the base class, and returns only a single value
            regardless of the time.

            :param name: Name
            :type name: string

            :param units: Units
            :type units: string

            :param time: Time axis of the data
            :type time: [] of datetime.datetime, netCDF4.Variable,
                        or Time object

            :param data: Value of the property
            :type data: array-like
        '''

        self._units = self._time = self._data = None

        self.units = units
        self.data = data
        self.time = time
        self.extrapolate = extrapolate
        super(TimeseriesData, self).__init__(*args, **kwargs)

    #
    # Subclasses should override\add any attribute property function
    # getter/setters as needed
    #

    @classmethod
    def constant(cls,
                 name=None,
                 units=None,
                 data=None,):
        if any(var is None for var in (name, data, units)):
            raise ValueError("name, data, or units may not be None")

        if not isinstance(data, Number):
            raise TypeError('{0} data must be a number'.format(name))

        t = Time.constant_time()

        return cls(name=name, units=units, time=t, data=[data])

    @property
    def timeseries(self):
        '''
        Creates a representation of the time series

        :rtype: list of (datetime, double) tuples
        '''
        return [(x, y) for x, y in zip(self.time.data, self.data)]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        if self.time is not None and len(d) != len(self.time):
            raise ValueError("Data/time interval mismatch")
        else:
            self._data = d

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        if self.data is not None and len(t) != len(self.data):
            warnings.warn("Data/time interval mismatch, doing nothing")
            return

        if isinstance(t, Time) or issubclass(t.__class__, gridded.time.Time):
            self._time = t
        elif isinstance(t, abc.Iterable):
            self._time = Time(t)
        else:
            raise ValueError('Object being assigned must be an iterable '
                             'or a Time object')

    def at(self, points, time, units=None, extrapolate=None, auto_align=True, **kwargs):
        '''
            Interpolates this property to the given points at the given time
            with the units specified.

            :param points: A Nx2 array of lon,lat points

            :param time: A datetime object. May be None; if this is so,
                         the variable is assumed to be gridded but
                         time-invariant

            :param units: The units that the result would be converted to
        '''
        pts = _reorganize_spatial_data(points)
        value = None
        if len(self.time) == 1:
            value = self.data
        else:
            if extrapolate is None:
                extrapolate = self.extrapolate
            if not extrapolate:
                self.time.valid_time(time)

            if extrapolate and time > self.time.max_time:
                value = self.data[-1]
            if extrapolate and time <= self.time.min_time:
                value = self.data[0]

            if value is None:
                t_index = self.time.index_of(time, extrapolate)
                t_alphas = self.time.interp_alpha(time, extrapolate)

                d0 = self.data[max(t_index - 1, 0)]
                d1 = self.data[t_index]

                value = d0 + (d1 - d0) * t_alphas


        data_units = self.units if self.units else self._gnome_unit
        req_units = units if units else data_units
        #Try to convert units. This is the same as in gridded_objects_base.Variable
        if data_units is not None and data_units != req_units:
            try:
                value = uc.convert(data_units, req_units, value)
            except uc.NotSupportedUnitError:
                if (not uc.is_supported(data_units)):
                    warnings.warn("{0} units is not supported: {1}".format(self.name, data_units))
                elif (not uc.is_supported(req_units)):
                    warnings.warn("Requested unit is not supported: {1}".format(req_units))
                else:
                    raise

        if points is None:
            return value
        else:
            rval = np.full((pts.shape[0], 1), value, dtype=np.float64)
            if auto_align:
                return _align_results_to_spatial_data(rval, points)
            else:
                return rval


    def in_units(self, unit):
        '''
        Returns a full cpy of this property in the units specified.
        WARNING: This will cpy the data of the original property!

        :param units: Units to convert to
        :type units: string
        :return: Copy of self converted to new units
        :rtype: Same as self
        '''
        cpy = copy.copy(self)

        if hasattr(cpy.data, '__mul__'):
            cpy.data = uc.convert(cpy.units, unit, cpy.data)
        else:
            warnings.warn('Data was not converted to new units and '
                          'was not copied because it does not support '
                          'multiplication')

        cpy._units = unit

        return cpy

    def is_constant(self):
        return len(self.data) == 1

TimeSeriesProp = TimeseriesData

class TimeseriesVectorSchema(ObjTypeSchema):
    units = SchemaNode(
        String(), missing=drop, save=True, update=True
    )
    time = TimeSchema(
        save=True, update=True, save_reference=True
    )
    variables = SequenceSchema(
        TimeseriesDataSchema(), save=True, update=True, save_reference=True
    )


class TimeseriesVector(GnomeId):
    '''
    Base class for multiple data sources aligned along the same single time dimension
    '''
    _schema = TimeseriesVectorSchema
    _gnome_unit = None

    def __init__(self,
                 variables=None,
                 time=None,
                 units=None,
                 *args,
                 **kwargs):
        '''
            A class that represents a vector natural phenomenon and provides
            an interface to get the value of the phenomenon at a position
            in space and time.

            :param name: Name of the Property
            :type name: string

            :param units: Unit of the underlying data
            :type units: string

            :param time: Time axis of the data
            :type time: [] of datetime.datetime, netCDF4.Variable,
                        or Time object

            :param variables: component data arrays
            :type variables: [] of TimeseriesData or numpy.array (Max len=2)
        '''

        self._units = self._time = self._variables = None

        if all([isinstance(v, TimeseriesData) for v in variables]):
            if time is not None and not isinstance(time, Time):
                time = Time(time)

        units = variables[0].units if units is None else units
        time = variables[0].time if time is None else time



        if variables is None or len(variables) < 2:
            raise ValueError('variables must be an array-like of 2 or more '
                             'TimeseriesData objects')

        self.variables = variables
        self.units = units
        self.time = time

        super(TimeseriesVector, self).__init__(*args, **kwargs)

    @property
    def time(self):
        '''
        Time axis of data. I

        :rtype: gnome.environment.property.Time
        '''
        if self.variables is not None:
            times = [v.time for v in self.variables]
            otherTimes = [t for t in times if t is not times[0]]
            if len(otherTimes) == 0:
                return times[0]
            else:
                warnings.warn('variables have different time objects')
                return times
        else:
            raise ValueError('variables is None, cannot get .time')

    @time.setter
    def time(self, time_data):
        '''
        Expected: array-like of datetime objects, or Time instance
        '''
        if isinstance(time_data, Time):
            for v in self.variables:
                v.time = time_data
        else:
            t = self.variables[0].time
            t.data = time_data
            for v in self.variables:
                v.time.data = t.data

    @property
    def units(self):
        '''
        Units of underlying data

        :rtype: string
        '''
        if hasattr(self._units, '__iter__'):
            if len(set(self._units)) > 1:
                return self._units
            else:
                return self._units[0]
        else:
            return self._units

    @units.setter
    def units(self, unit):
        if unit is not None:
            if not uc.is_supported(unit):
                raise ValueError('Units of {0} are not supported'.format(unit))

        self._units = unit

        if self.variables is not None:
            for v in self.variables:
                v.units = unit

    @property
    def varnames(self):
        '''
        Names of underlying variables

        :rtype: [] of strings
        '''
        return [v.varname if hasattr(v, 'varname') else v.name
                for v in self.variables]

    def _check_consistency(self):
        '''
            Checks that the attributes of each GriddedProp in varlist
            are the same as the GridVectorProp
        '''
        raise NotImplementedError()

    def at(self, points, time, units=None, *args, **kwargs):
        '''
            Find the value of the property at positions P at time T

            TODO: What are the argument names for time and time level really?

            :param points: Coordinates to be queried (P)
            :type points: Nx2 array of double

            :param time: The time at which to query these points (T)
            :type time: datetime.datetime object

            :param time: Specifies the time level of the variable
            :type time: integer

            :param units: units the values will be returned in
                          (or converted to)
            :type units: string such as ('m/s', 'knots', etc)

            :param extrapolate: if True, extrapolation will be supported
            :type extrapolate: boolean (True or False)

            :return: returns a Nx2 array of interpolated values
            :rtype: double
        '''
        units = units if units else self._gnome_unit #no need to convert here, its handled in the subcomponents
        val = np.column_stack([var.at(points, time,  units=units, *args, **kwargs) for var in self.variables])

        # No need to unit convert since that should be handled by the individual variable objects
        if points is None:
            return val[0]
        else:
            return val

