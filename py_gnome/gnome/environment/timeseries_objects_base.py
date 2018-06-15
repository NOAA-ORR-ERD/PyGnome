import warnings
import copy
from numbers import Number
import collections

import numpy as np

from colander import SchemaNode, String, Float, drop, SequenceSchema, Sequence

import unit_conversion

from gnome.persist import base_schema

from gnome.environment.gridded_objects_base import Time, TimeSchema
from gnome.gnomeobject import GnomeId
from gnome.persist.extend_colander import NumpyArraySchema


class TimeseriesDataSchema(base_schema.ObjTypeSchema):
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
        return map(lambda x, y: (x, y), self.time.data, self.data)

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

        if isinstance(t, Time):
            self._time = t
        elif isinstance(t, collections.Iterable):
            self._time = Time(t)
        else:
            raise ValueError('Object being assigned must be an iterable '
                             'or a Time object')

    def at(self, points, time, units=None, extrapolate=None, **kwargs):
        '''
            Interpolates this property to the given points at the given time
            with the units specified.

            :param points: A Nx2 array of lon,lat points

            :param time: A datetime object. May be None; if this is so,
                         the variable is assumed to be gridded but
                         time-invariant

            :param units: The units that the result would be converted to
        '''
        value = None
        if extrapolate is None:
            extrapolate = self.extrapolate

        if len(self.time) == 1:
            # single time time series (constant)
            value = np.full((points.shape[0], 1), self.data, dtype=np.float64)

            if units is not None and units != self.units:
                value = unit_conversion.convert(self.units, units, value)

            return value

        if not extrapolate:
            self.time.valid_time(time)

        t_index = self.time.index_of(time, extrapolate)

        if time > self.time.max_time:
            value = self.data[-1]

        if time <= self.time.min_time:
            value = self.data[0]

        if value is None:
            t_alphas = self.time.interp_alpha(time, extrapolate)

            d0 = self.data[t_index - 1]
            d1 = self.data[t_index]

            value = d0 + (d1 - d0) * t_alphas

        if units is not None and units != self.units:
            value = unit_conversion.convert(self.units, units, value)

        return np.full((points.shape[0], 1), value, dtype=np.float64)

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
            cpy.data = unit_conversion.convert(cpy.units, unit, cpy.data)
        else:
            warnings.warn('Data was not converted to new units and '
                          'was not copied because it does not support '
                          'multiplication')

        cpy._units = unit

        return cpy

    def is_constant(self):
        return len(self.data) == 1


class TimeseriesVectorSchema(base_schema.ObjTypeSchema):
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
            otherTimes = filter(lambda t: t is not times[0], times)
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
            if len(set(self._units) > 1):
                return self._units
            else:
                return self._units[0]
        else:
            return self._units

    @units.setter
    def units(self, unit):
        if unit is not None:
            if not unit_conversion.is_supported(unit):
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

    def at(self, *args, **kwargs):
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
        return np.column_stack([var.at(*args, **kwargs)
                                for var in self.variables])

