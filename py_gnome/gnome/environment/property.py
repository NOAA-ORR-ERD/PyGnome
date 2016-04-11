import warnings
import copy

import netCDF4 as nc4
import numpy as np

from gnome.utilities.geometry.cy_point_in_polygon import points_in_polys
from datetime import datetime, timedelta
from dateutil import parser
from colander import SchemaNode, Float, Boolean, Sequence, MappingSchema, drop, String, OneOf, SequenceSchema, TupleSchema, DateTime
from gnome.persist.base_schema import ObjType
from gnome.utilities import serializable
from gnome.movers import ProcessSchema
from gnome.persist import base_schema

from gnome.utilities.timeseries_generic import DataTimeSeries

import pyugrid
import pysgrid
import unit_conversion


class EnvProp(object):
    '''
    A class that represents a natural phenomenon and provides an interface to get
    the value of the phenomenon with respect to space and time. EnvProp is the base
    class, and returns only a single value regardless of the time
    '''
    
    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 data=None,
                 extrapolate=False):
        
        
        self.name = name
        if units in unit_conversion.unit_data.supported_units:
            self._units = units
        else:
            raise ValueError('Units of {0} are not supported'.format(units))
        self.time = time
        self.data = data
        self.extrapolate = extrapolate

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, unit):
        warnings.warn('Setting units directly does not change data values. If this is desired, use convert_units()')
        self._units = unit

    def at(self, points, time):
        return np.full((points.shape[0], 1), data)

    def in_units(self, unit):
        '''
        Returns a full cpy of this property in the units specified. 
        WARNING: This will cpy the data of the original property!
        '''
        cpy = copy.copy(self)
        if hasattr(cpy.data, '__mul__'):
            cpy.data = unit_conversion.Convert(None, cpy.units, unit, cpy.data)
        else:
            warnings.warn('Data was not converted to new units and was not copied because it does not support multiplication')
        cpy._units = unit
        return cpy


class TimeSeriesProp(EnvProp):
    '''
    This class represents a phenomenon using a time series
    '''
    
    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 data=None,
                 extrapolate=False):
        if len(time) != len(data):
            raise ValueError("Time and data sequences are of different length.\n\
            len(time) == {0}, len(data) == {1}".format(len(time), len(data)))
        super(TimeSeriesProp, self).__init__(name, units, time, data, extrapolate)
        self.time = Time(time)
        self._time = self.time.time

    @property
    def timeseries(self):
        return map(lambda x,y:(x,y), self.time.time, self.data)

    def at(self, points, time, units=None):
        '''
        Interpolates this property to the given points at the given time with the units specified
        :param points: A Nx2 array of lon,lat points
        :param time: A datetime object. May be None; if this is so, the variable is assumed to be gridded
        but time-invariant
        :param units: The units that the result would be converted to
        '''
        value = None
        if len(self.time) == 1:
            #single time time series (constant)
            return np.full((points.shape[0], 1), data)
        t_index = self.time.indexof(time)
        if self.extrapolate and t_index == len(self.time):
            value = self.data[t_index]

        else:
            t_alphas = self.time.interp_alpha(time)
            d0 = self.data[t_index]
            d1 = self.data[t_index + 1]
            value = d0 + (d1 - d0) * t_alphas
        if units is not None and units != self.units:
            value = unit_conversion.convert(None, self.units, units, value)

        return np.full((points.shape[0], 1), value)


class GriddedProp(EnvProp):
    '''
    This class represents a phenomenon using gridded data
    '''

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 data=None,
                 grid=None,
                 extrapolate=False,
                 data_file=None,
                 grid_file=None):
        if grid is None or data is None:
            raise ValueError('Must provide a grid and data that can fit to the grid')
        if grid.infer_grid(data) is None:
            raise ValueError('Data must be able to fit to the grid')
        super(GriddedProp, self).__init__(name=name, units=units, data=data, extrapolate=extrapolate)
        self.grid = grid
        self.time = Time(time)
        self.data_file = data_file
        self.grid_file = grid_file

    def at(self, points, time, units=None):
        '''
        Interpolates this property to the given points at the given time.
        :param points: A Nx2 array of lon,lat points
        :param time: A datetime object. May be None; if this is so, the variable is assumed to be gridded
        but time-invariant
        '''
        t_alphas = t_index = s0 = s1 = value = None
        if self.time is not None:
            t_index = self.time.indexof(time)
            if self.extrapolate and t_index == len(self.time.time):
                s0 = [t_index]
                value = self.grid.interpolate_var_to_points(points, self.data, slices=s0, memo=True)
            else:
                t_alphas = self.time.interp_alpha(time)
                s0 = [t_index]
                s1 = [t_index + 1]
                if len(self.data.shape) == 4:
                    s1.append(depth)
                    s2.append(depth)
                v0 = self.grid.interpolate_var_to_points(points, self.data, slices=s0, memo=True)
                v1 = self.grid.interpolate_var_to_points(points, self.data, slices=s1, memo=True)
                value = v0 + (v1 - v0) * t_alphas
        else:
            s0 = None
            value = self.grid.interpolate_var_to_points(points, self.data, slices=s0, memo=True)

        if units is not None and units != self.units:
            value = unit_conversion.convert(None, self.units, units, value)
        return value


class VectorProp(object):

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 extrapolate=False):
        self.extrapolate = extrapolate
        self.name = name
        if units is None:
            units = variables[0].units
        self._units = units
        if time is None:
            self._time = variables[0].time
        elif isinstance(time, Time):
            self._time = Time
        else:
            self._time = Time(time)

        self.data_file = None
        self.grid_file = None
        if not isinstance(variables[0], GriddedProp):
            self.data_format = 'timeseries'
        else:
            self.data_format = 'gridded'
            self.data_file = variables[0].data_file
            self.grid_file = variables[0].grid_file
            
        self.check_variables(variables)
        self.variables = variables
        
        
    def check_variables(self, variables):
        for i, var in enumerate(variables):
            if self.data_format == 'timeseries':
                if not isinstance(var, TimeSeriesProp):
                    if isinstance(var, iterable) and len(var) == len(self.time):
                        variables[i] = TimeSeriesProp(name='var{0}'.format(i),
                                                      units=self.units, time=self.time,
                                                      extrapolate=self.extrapolate)
                    else:
                        raise ValueError('Variables must contain an iterable or TimeSeriesProp')
            if self.data_format == 'gridded':
                if not isinstance(var, GriddedProp):
                    raise ValueError('All variables must either be gridded or time series')
                if var.data_file != self.data_file:
                    raise ValueError("""Data filename for component property {0} is different 
                                     than reference datafile {1}""".format(var.data_file, self.data_file))
                if var.grid_file != self.grid_file:
                    raise ValueError("""Grid filename for component property {0} is different 
                                     than reference gridfile {1}""".format(var.grid_file, self.grid_file))
            
            if var.time != self._time:
                raise ValueError('All variables must share the same time series')
            if var.units != self.units:
                raise ValueError('Units of {0} for component property {1} are not the same as \
                units specified for compound proprety {2}'.format(var.name, var.units, self.units))
            if var.extrapolate != self.extrapolate:
                raise ValueError("""VectorProp extrapolation is {0}, 
                                 but component property {1} 
                                 extrapolation is {2}""".format('on' if self.extrapolate else 'off',
                                                                var.name,
                                                                'on' if var.extrapolate else 'off'))
    def at(self, points, time, units):
        val = [v.at(points, time, units) for v in self.variables]
        return val
        

    @property
    def units(self):
        return self._units

    @property
    def time(self):
        return self._time

    def at(self, points, time, units=None):
        return np.column_stack((var.at(points, time, units) for var in self.variables))

class VelocitySchema(base_schema.ObjType):
    
    name = SchemaNode(String(), missing='default')
    units = SchemaNode(String(), missing='none')
    data_format = SchemaNode(String(), missing='gridded')
    extrapolate = SchemaNode(Boolean(), missing='False')
    data_file = SchemaNode(String(), missing=drop)
    grid_file = SchemaNode(String(), missing=drop)
    time = SequenceSchema(SchemaNode(DateTime(default_tzinfo=None), missing=drop), missing=drop)
    timeseries = SequenceSchema(TupleSchema(children=[SchemaNode(DateTime(default_tzinfo=None), missing=drop),
                                                      TupleSchema(children=[
                                                                            SchemaNode(Float(), missing=0),
                                                                            SchemaNode(Float(), missing=0)
                                                                            ]
                                                                 )
                                                      ]
                                            , missing=drop))
    varnames = SequenceSchema(SchemaNode(String(), missing=drop))
    

class Velocity(VectorProp, serializable.Serializable):
    
    _state = copy.deepcopy(serializable.Serializable._state)
    _schema=VelocitySchema
    
    _state.add_field([serializable.Field('units', save=True, update=True), 
                      serializable.Field('time', save=True, update=True),
                      serializable.Field('timeseries', save=True, update=True),
                      serializable.Field('varnames', save=True, update=True),
                      serializable.Field('extrapolate', save=True, update=True),
                      serializable.Field('data_format', save=True, update=True),
                      serializable.Field('data_file', save=True, update=True),
                      serializable.Field('grid_file', save=True, update=True)])
    
    def __init__(self, name=None, units=None, time = None, components = None, extrapolate=False, mag_dir = False):
        VectorProp.__init__(self, name, units, time=time, variables=components, extrapolate=extrapolate)
        if self.data_format == 'timeseries':
            if mag_dir:
                self.data_format = 'mag_dir_timeseries'
        

    @property
    def timeseries(self):
        if self.data_format != 'gridded':
            x = self.variables[0].data
            y = self.variables[1].data
            if self.data_format == 'mag_dir_timeseries':
                direction = -(np.arctan2(y,x)*180/np.pi + 90)
                magnitude = np.sqrt(x**2 + y**2)
                return map(lambda t,x,y:(t,(x,y)), self._time, magnitude, direction)
            else:
                return map(lambda t,x,y:(t,(x,y)), self._time, x, y)
        return None

    @property
    def varnames(self):
        return [v.name for v in self.variables]

    @property
    def time(self):
        if self.data_format != 'gridded':
            return None
        else:
            return self._time
    @classmethod
    def from_file(cls, filename, name):
        pass

    @classmethod
    def deserialize(cls, json_):
        d = super(Velocity, cls).deserialize(json_)
        ts, ds = zip(*d['timeseries'])
        del d['timeseries']
        d['time'] = np.array(ts)
        d['data'] = ds
        return d
    
    @classmethod
    def new_from_dict(cls, dict_):
        data = np.array(dict_.pop('data'))
        x_data, y_data = np.hsplit(data,2)
        if dict_['data_format'] == 'mag_dir_timeseries':
            y_data = ((-y_data - 90) * np.pi/180)
            x_t = x_data *np.cos(y_data)
            y_data = x_data * np.sin(y_data)
            x_data = x_t
        if 'timeseries' in dict_['data_format']:
            x_ts = TimeSeriesProp(name = dict_['varnames'][0],
                              units = dict_['units'],
                              time = dict_['time'],
                              data = x_data,
                              extrapolate = dict_['extrapolate'])
            y_ts = TimeSeriesProp(name = dict_['varnames'][1],
                              units = dict_['units'],
                              time = dict_['time'],
                              data = y_data,
                              extrapolate = dict_['extrapolate'])
        dict_['components'] = [x_ts,y_ts]
        #undo mag/direction here x_x
        return super(Velocity, cls).new_from_dict(dict_)

class Time(object):

    def __init__(self, time_seq, extrapolate=False):
        '''
        Functions for a time array
        :param time_seq: An ascending array of datetime objects of length N
        '''
        if isinstance(time_seq, nc4.Variable):
            self.time = nc4.num2date(time_seq[:], units=time_seq.units)
        else:
            self.time = time_seq

#         if not self._timeseries_is_ascending(self.time):
#             raise ValueError("Time sequence is not ascending")
#         if self._has_duplicates(self.time):
#             raise ValueError("Time sequence has duplicate entries")
        self.extrapolate = False

    @classmethod
    def time_from_nc_var(cls, var):
        return cls(nc4.num2date(var[:], units=var.units))

    def __len__(self):
        return len(self.time)
    
    def __iter__(self):
        return self.time.__iter__()
    
    def __eq__(self, other):
        return (self.time == other.time).all()
    
    def __ne__(self, other):
        return (self.time != other.time).all()

    def _timeseries_is_ascending(self, ts):
        return all(np.sort(ts) == ts)

    def _has_duplicates(self, ts):
        return len(np.unique(ts)) != len(ts)

    @property
    def min_time(self):
        return self.time[0]

    @property
    def max_time(self):
        return self.time[-1]

    def get_time_array(self):
        return self.time[:]

    def time_in_bounds(self, time):
        return not time < self.min_time or time > self.max_time

    def valid_time(self, time):
        if time < self.min_time or time > self.max_time:
            raise ValueError('time specified ({0}) is not within the bounds of the time ({1} to {2})'.format(
                time.strftime('%c'), self.min_time.strftime('%c'), self.max_time.strftime('%c')))

    def indexof(self, time):
        '''
        Returns the index of the provided time with respect to the time intervals in the file.
        :param time:
        :return:
        '''
        if not self.extrapolate:
            self.valid_time(time)
        index = np.searchsorted(self.time, time) - 1
        return index

    def interp_alpha(self, time):
        if not self.extrapolate:
            self.valid_time(time)
        i0 = self.indexof(time)
        if i0 == len(self.time):
            return 1
        t0 = self.time[i0]
        t1 = self.time[i0 + 1]
        return (time - t0).total_seconds() / (t1 - t0).total_seconds()

if __name__ == "__main__":
    import datetime as dt
    dates = np.array([dt.datetime(1, 1, 1, 0), dt.datetime(1, 1, 1, 2), dt.datetime(1, 1, 1, 4)])
    u_data = np.array([3, 4, 5])
    v_data = np.array([4, 3, 12])
    u = TimeSeriesProp('u', 'm/s', dates, u_data)
    v = TimeSeriesProp('v', 'm/s', dates, v_data)

    print u.at(np.array([(1, 1), (1, 2)]), dt.datetime(1, 1, 1, 1))

    vprop = VectorProp('velocity', 'm/s', variables=[u, v])
    print vprop.at(np.array([(1, 1), (1, 2)]), dt.datetime(1, 1, 1, 3))
    
    vel = Velocity('test_vel', components = [u,v], mag_dir=True)
    print vel.at(np.array([(1, 1), (1, 2)]), dt.datetime(1, 1, 1, 3))
    
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vel.serialize())
    pp.pprint(Velocity.deserialize(vel.serialize()))
    
    velfromdict = Velocity.new_from_dict(Velocity.deserialize(vel.serialize()))
    
    pp.pprint(vel.serialize())
    pp.pprint(velfromdict.serialize())
    
    url = ('http://geoport.whoi.edu/thredds/dodsC/clay/usgs/users/jcwarner/Projects/Sandy/triple_nest/00_dir_NYB05.ncml')
    test_grid = pysgrid.load_grid(url)
    grid_u = test_grid.u
    grid_v = test_grid.v
    grid_time = test_grid.ocean_time._data
    
    u2 = GriddedProp('u','m/s', time=grid_time, data=grid_u, grid=test_grid, data_file=url, grid_file=url)
    v2 = GriddedProp('v','m/s', time=grid_time, data=grid_v, grid=test_grid, data_file=url, grid_file=url) 
    
    print "got here"
    vel2 = Velocity(name='gridvel', components=[u2, v2])
    
#     pp.pprint(vel2.serialize())
    
    pass
