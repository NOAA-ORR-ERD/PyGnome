import warnings
import os
import copy
import StringIO
import zipfile
import pytest

import netCDF4 as nc4
import numpy as np

from datetime import datetime, timedelta
from colander import SchemaNode, Float, Boolean, Sequence, MappingSchema, drop, String, OneOf, SequenceSchema, TupleSchema, DateTime
from gnome.persist.base_schema import ObjType
from gnome.utilities import serializable
from gnome.persist import base_schema
from gnome.utilities.file_tools.data_helpers import _get_dataset

import pyugrid
import pysgrid
import unit_conversion
import collections
from collections import OrderedDict
from gnome.gnomeobject import GnomeId


class TimeSchema(base_schema.ObjType):
#     time = SequenceSchema(SchemaNode(DateTime(default_tzinfo=None), missing=drop), missing=drop)
    filename = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String())], missing=drop)
    varname = SchemaNode(String(), missing=drop)
    data = SchemaNode(typ=Sequence(), children=[SchemaNode(DateTime(None))], missing=drop)


class PropertySchema(base_schema.ObjType):
    name = SchemaNode(String(), missing=drop)
    units = SchemaNode(String(), missing=drop)
#     units = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String(), missing=drop), SchemaNode(String(), missing=drop)])
    time = TimeSchema(missing=drop)  # SequenceSchema(SchemaNode(DateTime(default_tzinfo=None), missing=drop), missing=drop)


class EnvProp(serializable.Serializable):

    _state = copy.deepcopy(serializable.Serializable._state)
    _schema = PropertySchema

    _state.add_field([serializable.Field('units', save=True, update=True),
                      serializable.Field('time', save=True, update=True, save_reference=True)])

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 data=None,
                 **kwargs):
        '''
        A class that represents a natural phenomenon and provides an interface to get
        the value of the phenomenon at a position in space and time. EnvProp is the base
        class, and returns only a single value regardless of the time.

        :param name: Name
        :param units: Units
        :param time: Time axis of the data
        :param data: Value of the property
        :type name: string
        :type units: string
        :type time: [] of datetime.datetime, netCDF4.Variable, or Time object
        :type data: netCDF4.Variable or numpy.array
        '''

        self.name = self._units = self._time = self._data = None

        self.name = name
        self.units = units
        self.data = data
        self.time = time
        for k in kwargs:
            setattr(self, k, kwargs[k])

    '''
    Subclasses should override\add any attribute property function getter/setters as needed
    '''

#     @property
#     def data(self):
#         '''
#         Underlying data
#
#         :rtype: netCDF4.Variable or numpy.array
#         '''
#         return self._data

    @property
    def units(self):
        '''
        Units of underlying data

        :rtype: string
        '''
        return self._units

    @units.setter
    def units(self, unit):
        if unit is not None:
            if not unit_conversion.is_supported(unit):
                raise ValueError('Units of {0} are not supported'.format(unit))
        self._units = unit

    @property
    def time(self):
        '''
        Time axis of data

        :rtype: gnome.environment.property.Time
        '''
        return self._time

    @time.setter
    def time(self, t):
        if t is None:
            self._time = None
        elif isinstance(t, Time):
            self._time = t
        elif isinstance(t, collections.Iterable):
            self._time = Time(t)
        else:
            raise ValueError("Object being assigned must be an iterable or a Time object")

    def at(self, *args, **kwargs):
        '''
        Find the value of the property at positions P at time T

        :param points: Coordinates to be queried (P)
        :param time: The time at which to query these points (T)
        :param time: Specifies the time level of the variable
        :param units: units the values will be returned in (or converted to)
        :param extrapolate: if True, extrapolation will be supported
        :type points: Nx2 array of double
        :type time: datetime.datetime object
        :type time: integer
        :type units: string such as ('m/s', 'knots', etc)
        :type extrapolate: boolean (True or False)
        :return: returns a Nx1 array of interpolated values
        :rtype: double
        '''

        raise NotImplementedError()

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
            warnings.warn('Data was not converted to new units and was not copied because it does not support multiplication')
        cpy._units = unit
        return cpy


class VectorPropSchema(base_schema.ObjType):
    units = SchemaNode(String(), missing=drop)
    time = TimeSchema(missing=drop)


class VectorProp(serializable.Serializable):

    _state = copy.deepcopy(serializable.Serializable._state)
    _schema = VectorPropSchema

    _state.add_field([serializable.Field('units', save=True, update=True),
                      serializable.Field('time', save=True, update=True, save_reference=True)])

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 **kwargs):
        '''
        A class that represents a vector natural phenomenon and provides an interface to get the value of
        the phenomenon at a position in space and time. VectorProp is the base class

        :param name: Name of the Property
        :param units: Unit of the underlying data
        :param time: Time axis of the data
        :param variables: component data arrays
        :type name: string
        :type units: string
        :type time: [] of datetime.datetime, netCDF4.Variable, or Time object
        :type variables: [] of EnvProp or numpy.array (Max len=2)
        '''

        self.name = self._units = self._time = self._variables = None

        self.name = name

        if all([isinstance(v, EnvProp) for v in variables]):
            if time is not None and not isinstance(time, Time):
                time = Time(time)
            units = variables[0].units if units is None else units
            time = variables[0].time if time is None else time
        if units is None:
            units = variables[0].units
        self._units = units
        if variables is None or len(variables) < 2:
            raise ValueError('Variables must be an array-like of 2 or more Property objects')
        self.variables = variables
        self._time = time
        unused_args = kwargs.keys() if kwargs is not None else None
        if len(unused_args) > 0:
#             print(unused_args)
            kwargs = {}
        super(VectorProp, self).__init__(**kwargs)

    @property
    def time(self):
        '''
        Time axis of data

        :rtype: gnome.environment.property.Time
        '''
        return self._time

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
        return [v.varname if hasattr(v, 'varname') else v.name for v in self.variables ]

    def _check_consistency(self):
        '''
        Checks that the attributes of each GriddedProp in varlist are the same as the GridVectorProp
        '''
        raise NotImplementedError()

    def at(self, *args, **kwargs):
        '''
        Find the value of the property at positions P at time T

        :param points: Coordinates to be queried (P)
        :param time: The time at which to query these points (T)
        :param time: Specifies the time level of the variable
        :param units: units the values will be returned in (or converted to)
        :param extrapolate: if True, extrapolation will be supported
        :type points: Nx2 array of double
        :type time: datetime.datetime object
        :type time: integer
        :type units: string such as ('m/s', 'knots', etc)
        :type extrapolate: boolean (True or False)
        :return: returns a Nx2 array of interpolated values
        :rtype: double
        '''
        return np.column_stack([var.at(*args, **kwargs) for var in self.variables])


class Time(serializable.Serializable):

    _state = copy.deepcopy(serializable.Serializable._state)
    _schema = TimeSchema

    _state.add_field([serializable.Field('filename', save=True, update=True, isdatafile=True),
                      serializable.Field('varname', save=True, update=True),
                      serializable.Field('data', save=True, update=True)])

    _const_time = None

    def __init__(self,
                 time=(datetime.now()),
                 filename=None,
                 varname=None,
                 tz_offset=None,
                 origin=None,
                 displacement=timedelta(seconds=0),
                 **kwargs):
        '''
        Representation of a time axis. Provides interpolation alphas and indexing.

        :param time: Ascending list of times to use
        :param tz_offset: offset to compensate for time zone shifts
        :param displacement: displacement to apply to the time data. Allows shifting entire time interval into future or past
        :param origin: shifts the time interval to begin at the time specified
        :type time: netCDF4.Variable or [] of datetime.datetime
        :type tz_offset: datetime.timedelta

        '''
        if isinstance(time, (nc4.Variable, nc4._netCDF4._Variable)):
            self.time = nc4.num2date(time[:], units=time.units)
        else:
            self.time = np.array(time)

        if origin is not None:
            diff = self.time[0] - origin
            self.time -= diff

        self.time += displacement

        self.filename = filename
        self.varname = varname

#         if self.filename is None:
#             self.filename = self.id + '_time.txt'

        if tz_offset is not None:
            self.time += tz_offset

        if not self._timeseries_is_ascending(self.time):
            raise ValueError("Time sequence is not ascending")
        if self._has_duplicates(self.time):
            raise ValueError("Time sequence has duplicate entries")

        self.name = time.name if hasattr(time, 'name') else None

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    dataset=None,
                    varname=None,
                    datavar=None,
                    tz_offset=None,
                    **kwargs):
        if dataset is None:
            dataset = _get_dataset(filename)
        if datavar is not None:
            if hasattr(datavar, 'time') and datavar.time in dataset.dimensions.keys():
                varname = datavar.time
            else:
                varname = datavar.dimensions[0] if 'time' in datavar.dimensions[0] else None
                if varname is None:
                    return cls.constant_time()
        time = cls(time=dataset[varname],
                   filename=filename,
                   varname=varname,
                   tz_offset=tz_offset,
                   **kwargs
                       )
        return time

    @staticmethod
    def constant_time():
        if Time._const_time is None:
            Time._const_time = Time([datetime.now()])
        return Time._const_time

    @classmethod
    def from_file(cls, filename=None, **kwargs):
        if isinstance(filename, list):
            filename = filename[0]
        fn = open(filename, 'r')
        t = []
        for l in fn:
            l = l.rstrip()
            if l is not None:
                t.append(datetime.strptime(l, '%c'))
        fn.close()
        return Time(t)

    def save(self, saveloc, references=None, name=None):
        '''
        Write Wind timeseries to file or to zip,
        then call save method using super
        '''
#         if self.filename is None:
#             self.filename = self.id + '_time.txt'
#             if zipfile.is_zipfile(saveloc):
#                 self._write_time_to_zip(saveloc, self.filename)
#             else:
#                 datafile = os.path.join(saveloc, self.filename)
#                 self._write_time_to_file(datafile)
#             rv = super(Time, self).save(saveloc, references, name)
#             self.filename = None
#         else:
#             rv = super(Time, self).save(saveloc, references, name)
#         return rv
        super(Time, self).save(saveloc, references, name)

    def _write_time_to_zip(self, saveloc, ts_name):
        '''
        use a StringIO type of file descriptor and write directly to zipfile
        '''
        fd = StringIO.StringIO()
        self._write_time_to_fd(fd)
        self._write_to_zip(saveloc, ts_name, fd.getvalue())

    def _write_time_to_file(self, datafile):
        '''write timeseries data to file '''
        with open(datafile, 'w') as fd:
            self._write_time_to_fd(fd)

    def _write_time_to_fd(self, fd):
        for t in self.time:
            fd.write(t.strftime('%c') + '\n')

    @classmethod
    def new_from_dict(cls, dict_):
        if 'varname' not in dict_:
            dict_['time'] = dict_['data']
#             if 'filename' not in dict_:
#                 raise ValueError
            return cls(**dict_)
        else:
            return cls.from_netCDF(**dict_)

    @property
    def data(self):
        return self.time

    def __len__(self):
        return len(self.time)

    def __iter__(self):
        return self.time.__iter__()

    def __eq__(self, other):
        r = self.time == other.time
        return all(r) if hasattr(r, '__len__') else r

    def __ne__(self, other):
        return not self.__eq__(other)

    def _timeseries_is_ascending(self, ts):
        return all(np.sort(ts) == ts)

    def _has_duplicates(self, time):
        return len(np.unique(time)) != len(time) and len(time) != 1

    @property
    def min_time(self):
        '''
        First time in series

        :rtype: datetime.datetime
        '''
        return self.time[0]

    @property
    def max_time(self):
        '''
        Last time in series

        :rtype: datetime.datetime
        '''
        return self.time[-1]

    def get_time_array(self):
        return self.time[:]

    def time_in_bounds(self, time):
        '''
        Checks if time provided is within the bounds represented by this object.

        :param time: time to be queried
        :type time: datetime.datetime
        :rtype: boolean
        '''
        return not time < self.min_time or time > self.max_time

    def valid_time(self, time):
        if time < self.min_time or time > self.max_time:
            raise ValueError('time specified ({0}) is not within the bounds of the time ({1} to {2})'.format(
                time.strftime('%c'), self.min_time.strftime('%c'), self.max_time.strftime('%c')))

    def index_of(self, time, extrapolate=False):
        '''
        Returns the index of the provided time with respect to the time intervals in the file.

        :param time: Time to be queried
        :param extrapolate:
        :type time: datetime.datetime
        :type extrapolate: boolean
        :return: index of first time before specified time
        :rtype: integer
        '''
        if not (extrapolate or len(self.time) == 1):
            self.valid_time(time)
        index = np.searchsorted(self.time, time)
        return index

    def interp_alpha(self, time, extrapolate=False):
        '''
        Returns interpolation alpha for the specified time

        :param time: Time to be queried
        :param extrapolate:
        :type time: datetime.datetime
        :type extrapolate: boolean
        :return: interpolation alpha
        :rtype: double (0 <= r <= 1)
        '''
        if not len(self.time) == 1 or not extrapolate:
            self.valid_time(time)
        i0 = self.index_of(time, extrapolate)
        if i0 > len(self.time) - 1:
            return 1
        if i0 == 0:
            return 0
        t0 = self.time[i0 - 1]
        t1 = self.time[i0]
        return (time - t0).total_seconds() / (t1 - t0).total_seconds()
