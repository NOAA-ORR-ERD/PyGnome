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


def curv_field(filename=None, dataset=None):
    if dataset is None:
        dataset = nc4.Dataset(filename)
    node_lon = dataset['lonc']
    node_lat = dataset['latc']
    u = dataset['water_u']
    v = dataset['water_v']
    dims = node_lon.dimensions[0] + ' ' + node_lon.dimensions[1]

    grid = pysgrid.SGrid(node_lon=node_lon,
                         node_lat=node_lat,
                         node_dimensions=dims)
    grid.u = pysgrid.variables.SGridVariable(data=u)
    grid.v = pysgrid.variables.SGridVariable(data=v)
    time = Time(dataset['time'])
    variables = {'u': grid.u,
                 'v': grid.v,
                 'time': time}
    return SField(grid, time=time, variables=variables)

class EnvPropSchema(base_schema.ObjType):
    
    name = SchemaNode(String(), missing='default')
    units = SchemaNode(String(), missing='none')
    data = SchemaNode(Float(), missing=drop)
    extrapolate = SchemaNode(Boolean(), missing='False')

class EnvProp(serializable.Serializable):
    '''
    A class that represents a natural phenomenon and provides an interface to get
    the value of the phenomenon with respect to space and time. EnvProp is the base
    class, and returns only a single value regardless of the time
    '''
    
    
    _state = copy.deepcopy(serializable.Serializable._state)
    _schema = EnvPropSchema

    # add 'filename' as a Field object
    _state.add_field([serializable.Field('units', save=True, update=True), 
                      serializable.Field('time', save=True, update=True),
                      serializable.Field('data', save=True, update=True),
                      serializable.Field('extrapolate', save=True, update=True)])
    
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


class TimeSeriesPropSchema(EnvPropSchema):
    
    _time = SchemaNode(DateTime(default_tzinfo=None), missing=drop)
    data = SchemaNode(Float(), missing=drop)
    timeseries = SequenceSchema(TupleSchema(children=[_time, data], missing=drop))

class TimeSeriesProp(EnvProp):
    '''
    This class represents a phenomenon using a time series
    '''
        
    _state = copy.deepcopy(EnvProp._state)
    
    _schema = TimeSeriesPropSchema

    # add 'filename' as a Field object
    _state.remove('data')
    _state.add_field([serializable.Field('timeseries', save=False, update=True)])
    

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

    @classmethod
    def deserialize(cls, json_):
        d = super(TimeSeriesProp, cls).deserialize(json_)
        ts, ds = zip(*d['timeseries'])
        del d['timeseries']
        d['time'] = ts
        d['data'] = ds
        return d


class GriddedPropSchema(EnvPropSchema):
    
    data_file = SchemaNode(String(), missing=drop)
    grid_file = SchemaNode(String(), missing=drop)
    time = SequenceSchema(SchemaNode(DateTime(default_tzinfo=None), missing=drop), missing=drop)
    

class GriddedProp(EnvProp):
    '''
    This class represents a phenomenon using gridded data
    '''
        
    _state = copy.deepcopy(EnvProp._state)
    
    _schema =GriddedPropSchema

    # add 'filename' as a Field object
    _state.remove('data')
    _state.remove('time')
    _state.add_field([serializable.Field('data_file', isdatafile=True, save=True, read=True,update=True, test_for_eq=False),
                      serializable.Field('grid_file', isdatafile=True, save=True, read=True, test_for_eq=False),
                      serializable.Field('time', save=False, update=True)])
    
    
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
        self._time = self.time.time
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
            self.time = variables[0].time
        elif isinstance(time, Time):
            self.time = Time
        else:
            self.time = Time(time)
            
        if not isinstance(variables[0], GriddedProp):
            self.data_format = 'timeseries'
        else:
            self.data_format = 'gridded'
            
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
            
            if var.time != self.time:
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
            
            if var.time != self.time:
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

    def at(self, points, time, units=None):
        return np.column_stack((var.at(points, time, units) for var in self.variables))

class VelocitySchema(base_schema.ObjType):
    
    name = SchemaNode(String(), missing='default')
    units = SchemaNode(String(), missing='none')
    data_format = SchemaNode(String(), missing='gridded')
    extrapolate = SchemaNode(Boolean(), missing='False')
    data_files = SchemaNode(String(), missing=drop)
    grid_file = SchemaNode(String(), missing=drop),
    time = SequenceSchema(SchemaNode(DateTime(default_tzinfo=None), missing=drop), missing=drop)
    timeseries = SequenceSchema(TupleSchema(children=[SchemaNode(DateTime(default_tzinfo=None), missing=drop),
                                                      TupleSchema(children=[
                                                                            SchemaNode(Float(), missing=0),
                                                                            SchemaNode(Float(), missing=0)
                                                                            ]
                                                                 )
                                                      ]
                                            , missing=drop))
    

class Velocity(VectorProp, serializable.Serializable):
    
    _state = copy.deepcopy(serializable.Serializable._state)
    _schema=VelocitySchema
    
    _state.add_field([serializable.Field('units', save=True, update=True), 
                      serializable.Field('time', save=True, update=True),
                      serializable.Field('extrapolate', save=True, update=True)])
    
    def __init__(self, name=None, time = None, components = None, mag_dir = False):
        VectorProp.__init__(self, name, time=time, variables=components)
        if self.data_format == 'timeseries':
            if mag_dir:
                self.data_format = 'mag_dir_timeseries'
            self._state.remove(['time', 'data_file', 'grid_file'])
            
        if self.data_format == 'gridded':
            self._state.remove(['timeseries'])
        

    @property
    def timeseries(self):
        x = self.variables[0].data
        y = self.variables[1].data
        if self.data_format == 'mag_dir_timeseries':
            direction = -(np.arctan2(y,x)*180/np.pi + 90)
            magnitude = np.sqrt(x**2 + y**2)
            return map(lambda t,x,y:(t,(x,y)), self.time, magnitude, direction)
        else:
            return map(lambda t,x,y:(t,(x,y)), self.time, x, y)
    
    @property
    def data_files(self):
        return [fn.data_file for fn in self.variables]
    
    @property
    def grid_file(self):
        return [fn.grid_file for fn in self.variables]
    
    @classmethod
    def from_file(cls, filename, name):
        pass

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
        return (self.time != other.time).any()

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
    
    url = ('http://geoport.whoi.edu/thredds/dodsC/clay/usgs/users/jcwarner/Projects/Sandy/triple_nest/00_dir_NYB05.ncml')
    test_grid = pysgrid.load_grid(url)
    grid_u = test_grid.u
    grid_v = test_grid.v
    grid_time = test_grid.ocean_time._data
    
    vel2 = Velocity('gridvel', components=[GriddedProp('u','m/s', time=grid_time, data=grid_u, grid=test_grid, data_file=url, grid_file=url),
                                                 GriddedProp('v','m/s', time=grid_time, data=grid_v, grid=test_grid, data_file=url, grid_file=url)])
    
    pp.pprint(vel2.serialize())
    
    pass
