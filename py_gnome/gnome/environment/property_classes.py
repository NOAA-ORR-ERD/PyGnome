import warnings
import copy

import netCDF4 as nc4
import numpy as np

from datetime import datetime, timedelta
from colander import SchemaNode, Float, Boolean, Sequence, MappingSchema, drop, String, OneOf, SequenceSchema, TupleSchema, DateTime
from gnome.persist.base_schema import ObjType
from gnome.utilities import serializable
from gnome.persist import base_schema

import pyugrid
import pysgrid
import unit_conversion
from gnome.environment.property import Time, PropertySchema
from gnome.environment.ts_property import TSVectorProp, TimeSeriesProp
from gnome.environment.grid_property import GridVectorProp, GriddedProp, GridPropSchema, init_grid, _get_dataset
from gnome.environment import Environment


class TemperatureTSSchema(PropertySchema):
    timeseries = SequenceSchema(
                                TupleSchema(
                                            children=[SchemaNode(DateTime(default_tzinfo=None), missing=drop),
                                                      SchemaNode(Float(), missing=0)
                                                      ],
                                            missing=drop)
                                )
    varnames = SequenceSchema(SchemaNode(String(), missing=drop))

class VelocityTSSchema(PropertySchema):
    timeseries = SequenceSchema(
                                TupleSchema(
                                            children=[SchemaNode(DateTime(default_tzinfo=None), missing=drop),
                                                      TupleSchema(children=[
                                                                            SchemaNode(Float(), missing=0),
                                                                            SchemaNode(Float(), missing=0)
                                                                            ]
                                                                 )
                                                      ],
                                            missing=drop)
                                )
    varnames = SequenceSchema(SchemaNode(String(), missing=drop))


class VelocityTS(TSVectorProp, serializable.Serializable):

    _state = copy.deepcopy(serializable.Serializable._state)
    _schema = VelocityTSSchema

    _state.add_field([serializable.Field('units', save=True, update=True),
                      serializable.Field('timeseries', save=True, update=True),
                      serializable.Field('varnames', save=True, update=True)])

    def __init__(self,
                 name=None,
                 units=None,
                 time = None,
                 variables = None,
                 **kwargs):

        if len(variables) > 2:
            raise ValueError('Only 2 dimensional velocities are supported')
        TSVectorProp.__init__(self, name, units, time=time, variables=variables)

    def __eq__(self, o):
        if o is None:
            return False
        t1 = (self.name == o.name and
              self.units == o.units and
              self.time == o.time)
        t2 = True
        for i in range(0, len(self._variables)):
            if self._variables[i] != o._variables[i]:
                t2=False
                break

        return t1 and t2

    def __str__(self):
        return self.serialize(json_='save').__repr__()

    @property
    def timeseries(self):
        x = self.variables[0].data
        y = self.variables[1].data
        return map(lambda t,x,y:(t,(x,y)), self._time, x, y)

    def serialize(self, json_='webapi'):
        dict_ = serializable.Serializable.serialize(self, json_=json_)
        #The following code is to cover the needs of webapi
        if json_ == 'webapi':
            dict_.pop('timeseries')
            dict_.pop('units')
            x = self.variables[0].data
            y = self.variables[1].data
            direction = -(np.arctan2(y,x)*180/np.pi + 90)
            magnitude = np.sqrt(x**2 + y**2)
            ts = (unicode(tx.isoformat()) for tx in self._time)
            dict_['timeseries'] = map(lambda t,x,y:(t,(x,y)), ts, magnitude, direction)
            dict_['units'] = (unicode(self.variables[0].units), u'degrees')
            dict_['varnames'] = [u'magnitude', u'direction', dict_['varnames'][0], dict_['varnames'][1]]
        return dict_


    @classmethod
    def deserialize(cls, json_):
        dict_ = super(VelocityTS, cls).deserialize(json_)

        ts, data = zip(*dict_.pop('timeseries'))
        ts = np.array(ts)
        data = np.array(data).T
        units = dict_['units']
        if len(units) > 1 and units[1] == 'degrees':
            u_data, v_data = data
            v_data = ((-v_data - 90) * np.pi/180)
            u_t = u_data *np.cos(v_data)
            v_data = u_data * np.sin(v_data)
            u_data = u_t
            data = np.array((u_data, v_data))
            dict_['varnames'] = dict_['varnames'][2:]

        units = units[0]
        dict_['units'] = units
        dict_['time'] = ts
        dict_['data'] = data
        return dict_

    @classmethod
    def new_from_dict(cls, dict_):
        varnames = dict_['varnames']
        vars = []
        for i, varname in enumerate(varnames):
            vars.append(TimeSeriesProp(name= varname,
                                       units= dict_['units'],
                                       time = dict_['time'],
                                       data = dict_['data'][i]))
        dict_.pop('data')
        dict_['variables'] = vars
        return super(VelocityTS, cls).new_from_dict(dict_)


class WindTS(VelocityTS, Environment):

    _ref_as = 'wind'

    def __init__(self,
                 name=None,
                 units=None,
                 time = None,
                 variables = None,
                 **kwargs):
        if 'timeseries' in kwargs:
            ts = kwargs['timeseries']

            time = map(lambda e:e[0], ts)
            mag = np.array(map(lambda e:e[1][0], ts))
            dir = np.array(map(lambda e:e[1][1], ts))
            dir = dir * -1 - 90
            u = mag * np.cos(dir * np.pi/180)
            v = mag * np.sin(dir * np.pi/180)
            variables = [u, v]
        VelocityTS.__init__(self,name, units, time, variables)

    @classmethod
    def constant_wind(cls,
                      name='',
                      speed = 0,
                      direction = 0,
                      units='m/s'):
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
        t = datetime.now().replace(microsecond=0, second=0, minute=0)
        direction = direction * -1 - 90
        u = speed * np.cos(direction * np.pi/180)
        v = speed * np.sin(direction * np.pi/180)
        return cls(name=name, units=units, time = [t], variables = [[u],[v]])

class WaterTemperature(GriddedProp):
    default_names = ['water_t', 'temp']


class IceConcentration(GriddedProp, serializable.Serializable):
    _state = copy.deepcopy(serializable.Serializable._state)

    _schema = GridPropSchema

    _state.add_field([serializable.Field('units', save=True, update=True),
                serializable.Field('varname', save=True, update=False),
                serializable.Field('time', save=True, update=True),
                serializable.Field('data_file', save=True, update=True),
                serializable.Field('grid_file', save=True, update=True)])

    default_names = ['ice_fraction',]

    def __eq__(self, o):
        t1 = (self.name == o.name and
              self.units == o.units and
              self.time == o.time and
              self.varname == o.varname)
        t2 = self.data == o.data
        return t1 and t2

    def __str__(self):
        return self.serialize(json_='save').__repr__()


class VelocityGridSchema(PropertySchema):
    data_file = SchemaNode(String(), missing=drop)
    grid_file = SchemaNode(String(), missing=drop)


class VelocityGrid(GridVectorProp, serializable.Serializable):
    _state = copy.deepcopy(serializable.Serializable._state)

    _schema = VelocityGridSchema

    _state.add_field([serializable.Field('units', save=True, update=True),
                serializable.Field('varnames', save=True, update=True),
                serializable.Field('time', save=True, update=True),
                serializable.Field('data_file', save=True, update=True),
                serializable.Field('grid_file', save=True, update=True)])

    def __init__(self,
                 name=None,
                 units=None,
                 time = None,
                 grid = None,
                 variables = None,
                 data_file=None,
                 grid_file=None,
                 dataset=None,
                 **kwargs):

        if len(variables) > 2:
            raise ValueError('Only 2 dimensional velocities are supported')
        GridVectorProp.__init__(self,
                                name=name,
                                units=units,
                                time=time,
                                grid=grid,
                                variables=variables,
                                data_file = data_file,
                                grid_file = grid_file,
                                dataset=dataset)

    def __eq__(self, o):
        if o is None:
            return False
        t1 = (self.name == o.name and
              self.units == o.units and
              self.time == o.time)
        t2 = True
        for i in range(0, len(self._variables)):
            if self._variables[i] != o._variables[i]:
                t2=False
                break

        return t1 and t2

    def __str__(self):
        return self.serialize(json_='save').__repr__()


class GridCurrent(VelocityGrid, Environment):
    _ref_as = 'current'

    default_names = [['u', 'v'], ['U', 'V'], ['water_u', 'water_v'], ['curr_ucmp', 'curr_vcmp']]

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 grid=None,
                 grid_file=None,
                 data_file=None,
                 dataset=None):
        VelocityGrid.__init__(self,
                              name=name,
                              units=units,
                              time=time,
                              variables=variables,
                              grid=grid,
                              grid_file=grid_file,
                              data_file=data_file,
                              dataset=dataset)
        self.angle = None
        df = None
        if dataset is not None:
            df = dataset
        elif data_file is not None:
            df = _get_dataset(data_file)
        if df is not None and 'angle' in df.variables.keys():
            #Unrotated ROMS Grid!
            self.angle = GriddedProp(name='angle',units='radians',time=None, grid=self.grid, data=df['angle'])

    def at(self, points, time, units=None, depth=-1, extrapolate=False):
        '''
        Find the value of the property at positions P at time T

        :param points: Coordinates to be queried (P)
        :param time: The time at which to query these points (T)
        :param depth: Specifies the depth level of the variable
        :param units: units the values will be returned in (or converted to)
        :param extrapolate: if True, extrapolation will be supported
        :type points: Nx2 array of double
        :type time: datetime.datetime object
        :type depth: integer
        :type units: string such as ('m/s', 'knots', etc)
        :type extrapolate: boolean (True or False)
        :return: returns a Nx2 array of interpolated values
        :rtype: double
        '''
        value = super(GridCurrent,self).at(points, time, units, extrapolate=extrapolate)
        if self.angle is not None:
            angs = self.angle.at(points, time, extrapolate=extrapolate,)
            x = value[:,0] * np.cos(angs) - value[:,1] * np.sin(angs)
            y = value[:,0] * np.sin(angs) + value[:,1] * np.cos(angs)
            value[:,0] = x
            value[:,1] = y
        return value


class GridWind(VelocityGrid, Environment):

    _ref_as = 'wind'

    default_names = [['air_u', 'air_v'], ['Air_U', 'Air_V'], ['air_ucmp', 'air_vcmp'], ['wind_u', 'wind_v']]

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 grid=None,
                 grid_file=None,
                 data_file=None,
                 dataset=None):
        VelocityGrid.__init__(self,
                              name=name,
                              units=units,
                              time=time,
                              variables=variables,
                              grid=grid,
                              grid_file=grid_file,
                              data_file=data_file,
                              dataset=dataset)
        self.angle = None
        df = None
        if dataset is not None:
            df = dataset
        elif data_file is not None:
            df = _get_dataset(data_file)
        if df is not None and 'angle' in df.variables.keys():
            #Unrotated ROMS Grid!
            self.angle = GriddedProp(name='angle',units='radians',time=None, grid=self.grid, data=df['angle'])

    def at(self, points, time, units=None, depth=-1, extrapolate=False):
        '''
        Find the value of the property at positions P at time T

        :param points: Coordinates to be queried (P)
        :param time: The time at which to query these points (T)
        :param depth: Specifies the depth level of the variable
        :param units: units the values will be returned in (or converted to)
        :param extrapolate: if True, extrapolation will be supported
        :type points: Nx2 array of double
        :type time: datetime.datetime object
        :type depth: integer
        :type units: string such as ('m/s', 'knots', etc)
        :type extrapolate: boolean (True or False)
        :return: returns a Nx2 array of interpolated values
        :rtype: double
        '''
        value = super(GridWind,self).at(points, time, units, extrapolate=extrapolate)
        if self.angle is not None:
            angs = self.angle.at(points, time, extrapolate=extrapolate)
            x = value[:,0] * np.cos(angs) - value[:,1] * np.sin(angs)
            y = value[:,0] * np.sin(angs) + value[:,1] * np.cos(angs)
            value[:,0] = x
            value[:,1] = y
        return value


class IceVelocity(VelocityGrid):

    default_names = [['ice_u', 'ice_v',],]

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 grid=None,
                 grid_file=None,
                 data_file=None,
                 dataset=None):
        VelocityGrid.__init__(self,
                              name=name,
                              units=units,
                              time=time,
                              variables=variables,
                              grid=grid,
                              grid_file=grid_file,
                              data_file=data_file,
                              dataset=dataset)


class IceAwareProp(serializable.Serializable):
    _state = copy.deepcopy(serializable.Serializable._state)

    _schema = VelocityGridSchema

    _state.add_field([serializable.Field('units', save=True, update=True),
#                 serializable.Field('varnames', save=True, update=True),
                serializable.Field('time', save=True, update=True),
                serializable.Field('data_file', save=True, update=True),
                serializable.Field('grid_file', save=True, update=True)])

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 ice_var=None,
                 water_var=None,
                 ice_conc_var=None,
                 grid=None,
                 grid_file=None,
                 data_file=None):
        self.name=name
        self.units=units
        self.time=time
        self.ice_var=ice_var
        self.water_var=water_var
        self.ice_conc_var=ice_conc_var
        self.grid=grid
        self.grid_file=grid_file
        self.data_file=data_file

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    grid_topology=None,
                    name=None,
                    units=None,
                    time=None,
                    ice_var=None,
                    water_var=None,
                    ice_conc_var=None,
                    grid=None,
                    dataset=None,
                    grid_file=None,
                    data_file=None):
        if filename is not None:
            data_file = filename
            grid_file = filename

        ds = None
        dg = None
        if dataset is None:
            if grid_file == data_file:
                ds = dg = _get_dataset(grid_file)
            else:
                ds = _get_dataset(data_file)
                dg = _get_dataset(grid_file)
        else:
            ds = dg = dataset

        if grid is None:
            grid = init_grid(grid_file,
                             grid_topology=grid_topology,
                             dataset=dg)
        if ice_var is None:
            ice_var = IceVelocity.from_netCDF(filename,
                                              grid=grid,
                                              dataset=ds)
        if time is None:
            time = ice_var.time
        if water_var is None:
            water_var = GridCurrent.from_netCDF(filename,
                                                time=time,
                                                grid=grid,
                                                dataset=ds)
        if ice_conc_var is None:
            ice_conc_var = IceConcentration.from_netCDF(filename,
                                                        time=time,
                                                        grid=grid,
                                                        dataset=ds)
        if name is None:
            name = 'IceAwareCurrent'
        if units is None:
            units = water_var.units
        return cls(name=name,
                   units=units,
                   time=time,
                   ice_var=ice_var,
                   water_var=water_var,
                   ice_conc_var=ice_conc_var,
                   grid=grid,
                   grid_file=grid_file,
                   data_file=data_file)

    def _scaling_function(self, *args, **kwargs):
        raise NotImplementedError()

class IceAwareCurrent(serializable.Serializable):
    _state = copy.deepcopy(serializable.Serializable._state)

    _schema = VelocityGridSchema

    _state.add_field([serializable.Field('units', save=True, update=True),
#                 serializable.Field('varnames', save=True, update=True),
                serializable.Field('time', save=True, update=True),
                serializable.Field('data_file', save=True, update=True),
                serializable.Field('grid_file', save=True, update=True)])
    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 ice_var=None,
                 water_var=None,
                 ice_conc_var=None,
                 grid=None,
                 grid_file=None,
                 data_file=None):
        self.name=name
        self.units=units
        self.time=time
        self.ice_var=ice_var
        self.water_var=water_var
        self.ice_conc_var=ice_conc_var
        self.grid=grid
        self.grid_file=grid_file
        self.data_file=data_file

    def _check_consistency(self):
        """
        Checks that the attributes of each GriddedProp in varlist are the same as the GridVectorProp
        """
        if self.units is None or self.time is None or self.grid is None:
            return
        for v in [self.ice_var, self.water_var]:
            if v.units != self.units:
                raise ValueError("Variable {0} did not have units consistent with what was specified. Got: {1} Expected {2}".format(v.name,v.units, self.units))
            if v.time != self.time:
                raise ValueError("Variable {0} did not have time consistent with what was specified Got: {1} Expected {2}".format(v.name,v.time, self.time))
            if v.grid != self.grid:
                raise ValueError("Variable {0} did not have grid consistent with what was specified Got: {1} Expected {2}".format(v.name,v.grid, self.grid))
            if v.grid_file != self.grid_file:
                raise ValueError("Variable {0} did not have grid_file consistent with what was specified Got: {1} Expected {2}".format(v.name,v.grid_file, self.grid_file))
            if v.data_file != self.data_file:
                raise ValueError("Variable {0} did not have data_file consistent with what was specified Got: {1} Expected {2}".format(v.name,v.data_file, self.data_file))

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    grid_topology=None,
                    name=None,
                    units=None,
                    time=None,
                    ice_var=None,
                    water_var=None,
                    ice_conc_var=None,
                    grid=None,
                    dataset=None,
                    grid_file=None,
                    data_file=None):
        if filename is not None:
            data_file = filename
            grid_file = filename

        ds = None
        dg = None
        if dataset is None:
            if grid_file == data_file:
                ds = dg = _get_dataset(grid_file)
            else:
                ds = _get_dataset(data_file)
                dg = _get_dataset(grid_file)
        else:
            ds = dg = dataset

        if grid is None:
            grid = init_grid(grid_file,
                             grid_topology=grid_topology,
                             dataset=dg)
        if ice_var is None:
            ice_var = IceVelocity.from_netCDF(filename,
                                              grid=grid,
                                              dataset=ds)
        if time is None:
            time = ice_var.time
        if water_var is None:
            water_var = GridCurrent.from_netCDF(filename,
                                                time=time,
                                                grid=grid,
                                                dataset=ds)
        if ice_conc_var is None:
            ice_conc_var = IceConcentration.from_netCDF(filename,
                                                        time=time,
                                                        grid=grid,
                                                        dataset=ds)
        if name is None:
            name = 'IceAwareCurrent'
        if units is None:
            units = water_var.units
        return cls(name=name,
                   units=units,
                   time=time,
                   ice_var=ice_var,
                   water_var=water_var,
                   ice_conc_var=ice_conc_var,
                   grid=grid,
                   grid_file=grid_file,
                   data_file=data_file)

    def at(self, points, time, units=None, extrapolate=False):
        interp = self.ice_conc_var.at(points, time, extrapolate=extrapolate)
        interp_mask = np.logical_and(interp >= 0.2, interp < 0.8)
        if len(np.where(interp_mask)[0]) != 0:
            ice_mask = interp >= 0.8

            water_v = self.water_var.at(points, time, units, extrapolate)
            ice_v = self.ice_var.at(points, time, units, extrapolate)
            interp -= 0.2
            interp *= 1.25
            interp *= 1.3333333333

            vels = water_v.copy()
            vels[ice_mask] = ice_v[ice_mask]
            diff_v = ice_v
            diff_v -= water_v
            vels[interp_mask] += diff_v[interp_mask]*interp[interp_mask][:,np.newaxis]
            return vels
        else:
            return self.water_var.at(points, time, units, extrapolate)


class IceAwareWind(serializable.Serializable, Environment):
    _state = copy.deepcopy(serializable.Serializable._state)

    _schema = VelocityGridSchema

    _state.add_field([serializable.Field('units', save=True, update=True),
#                 serializable.Field('varnames', save=True, update=True),
                serializable.Field('time', save=True, update=True),
                serializable.Field('data_file', save=True, update=True),
                serializable.Field('grid_file', save=True, update=True)])
    _ref_as = 'wind'
    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 ice_var=None,
                 wind_var=None,
                 ice_conc_var=None,
                 grid=None,
                 grid_file=None,
                 data_file=None):
        self.name=name
        self.units=units
        self.time=time
        self.ice_var=ice_var
        self.wind_var=wind_var
        self.ice_conc_var=ice_conc_var
        self.grid=grid
        self.grid_file=grid_file
        self.data_file=data_file

    def _check_consistency(self):
        """
        Checks that the attributes of each GriddedProp in varlist are the same as the GridVectorProp
        """
        if self.units is None or self.time is None or self.grid is None:
            return
        for v in [self.ice_var, self.wind_var]:
            if v.units != self.units:
                raise ValueError("Variable {0} did not have units consistent with what was specified. Got: {1} Expected {2}".format(v.name,v.units, self.units))
            if v.time != self.time:
                raise ValueError("Variable {0} did not have time consistent with what was specified Got: {1} Expected {2}".format(v.name,v.time, self.time))
            if v.grid != self.grid:
                raise ValueError("Variable {0} did not have grid consistent with what was specified Got: {1} Expected {2}".format(v.name,v.grid, self.grid))
            if v.grid_file != self.grid_file:
                raise ValueError("Variable {0} did not have grid_file consistent with what was specified Got: {1} Expected {2}".format(v.name,v.grid_file, self.grid_file))
            if v.data_file != self.data_file:
                raise ValueError("Variable {0} did not have data_file consistent with what was specified Got: {1} Expected {2}".format(v.name,v.data_file, self.data_file))

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    grid_topology=None,
                    name=None,
                    units=None,
                    time=None,
                    ice_var=None,
                    wind_var=None,
                    ice_conc_var=None,
                    grid=None,
                    dataset=None,
                    grid_file=None,
                    data_file=None):
        if filename is not None:
            data_file = filename
            grid_file = filename

        ds = None
        dg = None
        if dataset is None:
            if grid_file == data_file:
                ds = dg = _get_dataset(grid_file)
            else:
                ds = _get_dataset(data_file)
                dg = _get_dataset(grid_file)
        else:
            ds = dg = dataset
        if grid is None:
            grid = init_grid(grid_file,
                             grid_topology=grid_topology,
                             dataset=dg)
        if ice_var is None:
            ice_var = IceVelocity.from_netCDF(filename,
                                              grid=grid,
                                              dataset=ds)
        if time is None:
            time = ice_var.time
        if wind_var is None:
            wind_var = GridWind.from_netCDF(filename,
                                                time=time,
                                                grid=grid,
                                                dataset=ds)
        if ice_conc_var is None:
            ice_conc_var = IceConcentration.from_netCDF(filename,
                                                        time=time,
                                                        grid=grid,
                                                        dataset=ds)
        if name is None:
            name = 'IceAwareCurrent'
        if units is None:
            units = wind_var.units
        return cls(name=name,
                   units=units,
                   time=time,
                   ice_var=ice_var,
                   wind_var=wind_var,
                   ice_conc_var=ice_conc_var,
                   grid=grid,
                   grid_file=grid_file,
                   data_file=data_file)

    def at(self, points, time, units=None, extrapolate=False):
        interp = self.ice_conc_var.at(points, time, extrapolate=extrapolate)
        interp_mask = np.logical_and(interp >= 0.2, interp < 0.8)
        if len(np.where(interp_mask)[0]) != 0:
            ice_mask = interp >= 0.8

            wind_v = self.wind_var.at(points, time, units, extrapolate)
            interp -= 0.2
            interp *= 1.25
            interp *= 1.3333333333

            vels = wind_v.copy()
            vels[ice_mask] = 0
            vels[interp_mask] *= (1 - interp[interp_mask][:,np.newaxis]) # scale winds from 100-0% depending on ice coverage
            return vels
        else:
            return self.wind_var.at(points, time, units, extrapolate)

if __name__ == "__main__":
    import datetime as dt
    dates = np.array([dt.datetime(2000, 1, 1, 0), dt.datetime(2000, 1, 1, 2), dt.datetime(2000, 1, 1, 4)])
    u_data = np.array([3, 4, 5])
    v_data = np.array([4, 3, 12])
    u = TimeSeriesProp('u', 'm/s', dates, u_data)
    v = TimeSeriesProp('v', 'm/s', dates, v_data)

    print u.at(np.array([(1, 1), (1, 2)]), dt.datetime(2000, 1, 1, 1))

    vprop = TSVectorProp('velocity', 'm/s', variables=[u, v])
    print vprop.at(np.array([(1, 1), (1, 2)]), dt.datetime(2000, 1, 1, 3))

    vel = VelocityTS('test_vel', variables = [u,v])
    print vel.at(np.array([(1, 1), (1, 2)]), dt.datetime(2000, 1, 1, 3))

    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vel.serialize())
    pp.pprint(VelocityTS.deserialize(vel.serialize()))

    velfromweb = VelocityTS.new_from_dict(VelocityTS.deserialize(vel.serialize()))
    velfromweb.name = 'velfromweb'

    pp.pprint(vel.serialize(json_='save'))
    pp.pprint(velfromweb.serialize(json_='save'))

    velfromsave = VelocityTS.new_from_dict(VelocityTS.deserialize(velfromweb.serialize(json_='save')))
    pp.pprint(velfromsave)

    velfromsave.at(np.array([(0,0)]), datetime(2000,1,1,0,0))

    url = ('http://geoport.whoi.edu/thredds/dodsC/clay/usgs/users/jcwarner/Projects/Sandy/triple_nest/00_dir_NYB05.ncml')
    test_grid = pysgrid.load_grid(url)
    grid_u = test_grid.u
    grid_v = test_grid.v
    grid_time = test_grid.ocean_time._data

    u2 = GriddedProp('u','m/s', time=grid_time, data=grid_u, grid=test_grid, data_file=url, grid_file=url)
    v2 = GriddedProp('v','m/s', time=grid_time, data=grid_v, grid=test_grid, data_file=url, grid_file=url)

    print "got here"
    vel2 = Velocity(name='gridvel', variables=[u2, v2])

#     pp.pprint(vel2.serialize())

    pass
