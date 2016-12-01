import warnings
import copy

import netCDF4 as nc4
import numpy as np

from numbers import Number

from datetime import datetime, timedelta
from colander import SchemaNode, Float, Boolean, Sequence, MappingSchema, drop, String, OneOf, SequenceSchema, TupleSchema, DateTime
from gnome.persist.base_schema import ObjType
from gnome.utilities import serializable
from gnome.persist import base_schema

import pyugrid
import pysgrid
import unit_conversion
from .. import _valid_units
from gnome.environment import Environment
from gnome.environment.property import Time, PropertySchema, VectorProp, EnvProp
from gnome.environment.ts_property import TSVectorProp, TimeSeriesProp
from gnome.environment.grid_property import GridVectorProp, GriddedProp, GridPropSchema
from gnome.utilities.file_tools.data_helpers import _init_grid, _get_dataset


class Depth(object):

    def __init__(self,
                 surface_index=-1):
        self.surface_index = surface_index
        self.bottom_index = surface_index

    @classmethod
    def from_netCDF(cls,
                    surface_index=-1):
        return cls(surface_index)

    def interpolation_alphas(self, points, data_shape, _hash=None):
        return None, None


class S_Depth(object):

    default_terms = [['Cs_w', 's_w', 'hc', 'Cs_r', 's_rho']]

    def __init__(self,
                 bathymetry,
                 data_file=None,
                 dataset=None,
                 terms={},
                 **kwargs):
        ds = dataset
        if ds is None:
            if data_file is None:
                data_file = bathymetry.data_file
                if data_file is None:
                    raise ValueError("Need data_file or dataset containing sigma equation terms")
            ds = _get_dataset(data_file)
        self.bathymetry = bathymetry
        self.terms = terms
        if len(terms) == 0:
            for s in S_Depth.default_terms:
                for term in s:
                    self.terms[term] = ds[term][:]

    @classmethod
    def from_netCDF(cls,
                    **kwargs
                    ):
        bathymetry = Bathymetry.from_netCDF(**kwargs)
        data_file = bathymetry.data_file,
        if 'dataset' in kwargs:
            dataset = kwargs['dataset']
        if 'data_file' in kwargs:
            data_file = kwargs['data_file']
        return cls(bathymetry,
                   data_file=data_file,
                   dataset=dataset)

    @property
    def surface_index(self):
        return -1
    
    @property
    def bottom_index(self):
        return 0

    @property
    def num_w_levels(self):
        return len(self.terms['s_w'])

    @property
    def num_r_levels(self):
        return len(self.terms['s_rho'])

    def _w_level_depth_given_bathymetry(self, depths, lvl):
        s_w = self.terms['s_w'][lvl]
        Cs_w = self.terms['Cs_w'][lvl]
        hc = self.terms['hc']
        return -(hc * (s_w - Cs_w) + Cs_w * depths)

    def _r_level_depth_given_bathymetry(self, depths, lvl):
        s_rho = self.terms['s_rho'][lvl]
        Cs_r = self.terms['Cs_r'][lvl]
        hc = self.terms['hc']
        return -(hc * (s_rho - Cs_r) + Cs_r * depths)
    
    def interpolation_alphas(self, points, data_shape, _hash=None):
        underwater = points[:, 2] > 0.0
        if len(np.where(underwater)[0]) == 0:
            return None, None
        indices = -np.ones((len(points)), dtype=np.int64)
        alphas = -np.ones((len(points)), dtype=np.float64)
        depths = self.bathymetry.at(points, datetime.now(), _hash=_hash)[underwater]
        pts = points[underwater]
        und_ind = -np.ones((len(np.where(underwater)[0])))
        und_alph = und_ind.copy()

        if data_shape[0] == self.num_w_levels:
            num_levels = self.num_w_levels
            ldgb = self._w_level_depth_given_bathymetry
        elif data_shape[0] == self.num_r_levels:
            num_levels = self.num_r_levels
            ldgb = self._r_level_depth_given_bathymetry
        else:
            raise ValueError('Cannot get depth interpolation alphas for data shape specified; does not fit r or w depth axis')
        blev_depths = ulev_depths = None
        for ulev in range(0, num_levels):
            ulev_depths = ldgb(depths, ulev)
#             print ulev_depths[0]
            within_layer = np.where(np.logical_and(ulev_depths < pts[:, 2], und_ind == -1))[0]
#             print within_layer
            und_ind[within_layer] = ulev
            if ulev == 0:
                und_alph[within_layer] = -2
            else:
                a = ((pts[:, 2].take(within_layer) - blev_depths.take(within_layer)) / 
                     (ulev_depths.take(within_layer) - blev_depths.take(within_layer)))
                und_alph[within_layer] = a
            blev_depths = ulev_depths

        indices[underwater] = und_ind
        alphas[underwater] = und_alph
        return indices, alphas


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
                 time=None,
                 variables=None,
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
                t2 = False
                break

        return t1 and t2

    def __str__(self):
        return self.serialize(json_='save').__repr__()

    @classmethod
    def constant(cls,
                 name='',
                 speed=0,
                 direction=0,
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
        direction = direction * -1 - 90
        u = speed * np.cos(direction * np.pi / 180)
        v = speed * np.sin(direction * np.pi / 180)
        return super(VelocityTS, self).constant(name, units, variables=[[u], [v]])

    @property
    def timeseries(self):
        x = self.variables[0].data
        y = self.variables[1].data
        return map(lambda t, x, y: (t, (x, y)), self._time, x, y)

    def serialize(self, json_='webapi'):
        dict_ = serializable.Serializable.serialize(self, json_=json_)
        # The following code is to cover the needs of webapi
        if json_ == 'webapi':
            dict_.pop('timeseries')
            dict_.pop('units')
            x = np.asanyarray(self.variables[0].data)
            y = np.asanyarray(self.variables[1].data)
            direction = -(np.arctan2(y, x) * 180 / np.pi + 90)
            magnitude = np.sqrt(x ** 2 + y ** 2)
            ts = (unicode(tx.isoformat()) for tx in self._time)
            dict_['timeseries'] = map(lambda t, x, y: (t, (x, y)), ts, magnitude, direction)
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
            v_data = ((-v_data - 90) * np.pi / 180)
            u_t = u_data * np.cos(v_data)
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
        vs = []
        for i, varname in enumerate(varnames):
            vs.append(TimeSeriesProp(name=varname,
                                     units=dict_['units'],
                                     time=dict_['time'],
                                     data=dict_['data'][i]))
        dict_.pop('data')
        dict_['variables'] = vs
        return super(VelocityTS, cls).new_from_dict(dict_)


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
                 time=None,
                 grid=None,
                 depth=None,
                 variables=None,
                 data_file=None,
                 grid_file=None,
                 dataset=None,
                 **kwargs):

        GridVectorProp.__init__(self,
                                name=name,
                                units=units,
                                time=time,
                                grid=grid,
                                depth=depth,
                                variables=variables,
                                data_file=data_file,
                                grid_file=grid_file,
                                dataset=dataset,
                                **kwargs)
        if len(variables) == 2:
            self.variables.append(TimeSeriesProp(name='constant w', data=[0.0], time=[datetime.now()], units='m/s'))

    def __eq__(self, o):
        if o is None:
            return False
        t1 = (self.name == o.name and
              self.units == o.units and
              self.time == o.time)
        t2 = True
        for i in range(0, len(self._variables)):
            if self._variables[i] != o._variables[i]:
                t2 = False
                break

        return t1 and t2

    def __str__(self):
        return self.serialize(json_='save').__repr__()


class WindTS(VelocityTS, Environment):

    _ref_as = 'wind'

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 **kwargs):
        if 'timeseries' in kwargs:
            ts = kwargs['timeseries']

            time = map(lambda e: e[0], ts)
            mag = np.array(map(lambda e: e[1][0], ts))
            d = np.array(map(lambda e: e[1][1], ts))
            d = d * -1 - 90
            u = mag * np.cos(d * np.pi / 180)
            v = mag * np.sin(d * np.pi / 180)
            variables = [u, v]
        VelocityTS.__init__(self, name, units, time, variables)

    @classmethod
    def constant_wind(cls,
                      name='Constant Wind',
                      speed=None,
                      direction=None,
                      units='m/s'):
        """
        :param speed: speed of wind
        :param direction: direction -- degrees True, direction wind is from
                          (degrees True)
        :param unit='m/s': units for speed, as a string, i.e. "knots", "m/s",
                           "cm/s", etc.
        """
        return super(WindTS, self).constant(name=name, speed=speed, direction=direction, units=units)


class CurrentTS(VelocityTS, Environment):

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 **kwargs):
        if 'timeseries' in kwargs:
            ts = kwargs['timeseries']
            time = map(lambda e: e[0], ts)
            mag = np.array(map(lambda e: e[1][0], ts))
            direction = np.array(map(lambda e: e[1][1], ts))
            direction = direction * -1 - 90
            u = mag * np.cos(direction * np.pi / 180)
            v = mag * np.sin(direction * np.pi / 180)
            variables = [u, v]
        VelocityTS.__init__(self, name, units, time, variables)

    @classmethod
    def constant_wind(cls,
                      name='Constant Current',
                      speed=None,
                      direction=None,
                      units='m/s'):
        """
        :param speed: speed of wind
        :param direction: direction -- degrees True, direction wind is from
                          (degrees True)
        :param unit='m/s': units for speed, as a string, i.e. "knots", "m/s",
                           "cm/s", etc.

        """
        return cls.constant(name=name, speed=speed, direction=direction, units=units)


class TemperatureTS(TimeSeriesProp, Environment):

    def __init__(self,
                 name=None,
                 units='K',
                 time=None,
                 data=None,
                 **kwargs):
        if 'timeseries' in kwargs:
            ts = kwargs['timeseries']

            time = map(lambda e: e[0], ts)
            data = np.array(map(lambda e: e[1], ts))
        TimeSeriesProp.__init__(self, name, units, time, data=data)

    @classmethod
    def constant_temperature(cls,
                             name='Constant Temperature',
                             temperature=None,
                             units='K'):
        return cls.constant(name=name, data=temperature, units=units)


class GridTemperature(GriddedProp, Environment):
    default_names = ['water_t', 'temp']


class SalinityTS(TimeSeriesProp, Environment):

    @classmethod
    def constant_salinity(cls,
                          name='Constant Salinity',
                          salinity=None,
                          units='ppt'):
        return cls.constant(name=name, data=salinity, units=units)


class GridSalinity(GriddedProp, Environment):
    default_names = ['salt']


class WaterDensityTS(TimeSeriesProp, Environment):

    def __init__(self,
                 name=None,
                 units='kg/m^3',
                 temperature=None,
                 salinity=None):
        if temperature is None or salinity is None or not isinstance(temperature, TemperatureTS) or not isinstance(salinity, SalinityTS):
            raise ValueError('Must provide temperature and salinity time series Environment objects')
        density_times = temperature.time if len(temperature.time.time) > len(salinity.time.time) else salinity.time
        dummy_pt = np.array([[0, 0], ])
        import gsw
        from gnome import constants
        data = [gsw.rho(salinity.at(dummy_pt, t), temperature.at(dummy_pt, t, units='C'), constants.atmos_pressure * 0.0001) for t in density_times.time]
        TimeSeriesProp.__init__(self, name, units, time=density_times, data=data)


class GridSediment(GriddedProp, Environment):
    default_names = ['sand_06']


class IceConcentration(GriddedProp, Environment, serializable.Serializable):
    _state = copy.deepcopy(serializable.Serializable._state)

    _schema = GridPropSchema

    _state.add_field([serializable.Field('units', save=True, update=True),
                      serializable.Field('varname', save=True, update=False),
                      serializable.Field('time', save=True, update=True),
                      serializable.Field('data_file', save=True, update=True),
                      serializable.Field('grid_file', save=True, update=True)])

    default_names = ['ice_fraction', ]

    def __eq__(self, o):
        t1 = (self.name == o.name and
              self.units == o.units and
              self.time == o.time and
              self.varname == o.varname)
        t2 = self.data == o.data
        return t1 and t2

    def __str__(self):
        return self.serialize(json_='save').__repr__()


class Bathymetry(GriddedProp):
    default_names = ['h']


class GridCurrent(VelocityGrid, Environment):
    _ref_as = 'current'

    default_names = [['u', 'v', 'w'],
                     ['U', 'V', 'W'],
                     ['u', 'v'],
                     ['U', 'V'],
                     ['water_u', 'water_v'],
                     ['curr_ucmp', 'curr_vcmp']]

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 grid=None,
                 depth=None,
                 grid_file=None,
                 data_file=None,
                 dataset=None,
                 **kwargs):
        VelocityGrid.__init__(self,
                              name=name,
                              units=units,
                              time=time,
                              variables=variables,
                              grid=grid,
                              depth=depth,
                              grid_file=grid_file,
                              data_file=data_file,
                              dataset=dataset)
        self.angle = None
        df = None
        if dataset is not None:
            df = dataset
        elif grid_file is not None:
            df = _get_dataset(grid_file)
        if df is not None and 'angle' in df.variables.keys():
            # Unrotated ROMS Grid!
            self.angle = GriddedProp(name='angle', units='radians', time=None, grid=self.grid, data=df['angle'])
        self.depth = depth

    def at(self, points, time, units=None, depth=-1, extrapolate=False, **kwargs):
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
        mem = kwargs['memoize'] if 'memoize' in kwargs else True
        _hash = kwargs['_hash'] if '_hash' in kwargs else None
        if _hash is None:
            _hash = self._get_hash(points, time)
            if '_hash' not in kwargs:
                kwargs['_hash'] = _hash

        if mem:
            res = self._get_memoed(points, time, self._result_memo, _hash=_hash)
            if res is not None:
                return res

        value = super(GridCurrent, self).at(points, time, units, extrapolate=extrapolate, **kwargs)
        if self.angle is not None:
            angs = self.angle.at(points, time, extrapolate=extrapolate, **kwargs)
            x = value[:, 0] * np.cos(angs) - value[:, 1] * np.sin(angs)
            y = value[:, 0] * np.sin(angs) + value[:, 1] * np.cos(angs)
            value[:, 0] = x
            value[:, 1] = y
        z = value[:, 2]
        z[points[:, 2] == 0.0] = 0
        if mem:
            self._memoize_result(points, time, value, self._result_memo, _hash=_hash)
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
                 dataset=None,
                 **kwargs):
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
        elif grid_file is not None:
            df = _get_dataset(grid_file)
        if df is not None and 'angle' in df.variables.keys():
            # Unrotated ROMS Grid!
            self.angle = GriddedProp(name='angle', units='radians', time=None, grid=self.grid, data=df['angle'])

    def at(self, points, time, units=None, depth=-1, extrapolate=False, **kwargs):
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
        mem = kwargs['memoize'] if 'memoize' in kwargs else True
        _hash = kwargs['_hash'] if '_hash' in kwargs else None
        if _hash is None:
            _hash = self._get_hash(points, time)
            if '_hash' not in kwargs:
                kwargs['_hash'] = _hash

        if mem:
            res = self._get_memoed(points, time, self._result_memo, _hash=_hash)
            if res is not None:
                return res

        value = super(GridWind, self).at(points, time, units, extrapolate=extrapolate, **kwargs)
        if self.angle is not None:
            angs = self.angle.at(points, time, extrapolate=extrapolate, **kwargs)
            x = value[:, 0] * np.cos(angs) - value[:, 1] * np.sin(angs)
            y = value[:, 0] * np.sin(angs) + value[:, 1] * np.cos(angs)
            value[:, 0] = x
            value[:, 1] = y
        if mem:
            self._memoize_result(points, time, value, self._result_memo, _hash=_hash)
        return value


class IceVelocity(VelocityGrid, Environment):

    default_names = [['ice_u', 'ice_v', ], ]

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 grid=None,
                 grid_file=None,
                 data_file=None,
                 dataset=None,
                 **kwargs):
        VelocityGrid.__init__(self,
                              name=name,
                              units=units,
                              time=time,
                              variables=variables,
                              grid=grid,
                              grid_file=grid_file,
                              data_file=data_file,
                              dataset=dataset,
                              **kwargs)


class IceAwareProp(serializable.Serializable, Environment):
    _state = copy.deepcopy(serializable.Serializable._state)
    _schema = VelocityGridSchema
    _state.add_field([serializable.Field('units', save=True, update=True),
                      serializable.Field('time', save=True, update=True),
                      serializable.Field('data_file', save=True, update=True),
                      serializable.Field('grid_file', save=True, update=True)])

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 ice_var=None,
                 ice_conc_var=None,
                 grid=None,
                 grid_file=None,
                 data_file=None,
                 **kwargs):
        self.name = name
        self.units = units
        self.time = time
        self.ice_var = ice_var
        self.ice_conc_var = ice_conc_var
        self.grid = grid
        self.grid_file = grid_file
        self.data_file = data_file

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    grid_topology=None,
                    name=None,
                    units=None,
                    time=None,
                    ice_var=None,
                    ice_conc_var=None,
                    grid=None,
                    dataset=None,
                    grid_file=None,
                    data_file=None,
                    **kwargs):
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
            grid = _init_grid(grid_file,
                              grid_topology=grid_topology,
                              dataset=dg)
        if ice_var is None:
            ice_var = IceVelocity.from_netCDF(filename,
                                              grid=grid,
                                              dataset=ds,
                                              **kwargs)
        if time is None:
            time = ice_var.time

        if ice_conc_var is None:
            ice_conc_var = IceConcentration.from_netCDF(filename,
                                                        time=time,
                                                        grid=grid,
                                                        dataset=ds,
                                                        **kwargs)
        if name is None:
            name = 'IceAwareProp'
        if units is None:
            units = ice_var.units
        return cls(name='foo',
                   units=units,
                   time=time,
                   ice_var=ice_var,
                   ice_conc_var=ice_conc_var,
                   grid=grid,
                   grid_file=grid_file,
                   data_file=data_file,
                   **kwargs)


class IceAwareCurrent(IceAwareProp):

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 ice_var=None,
                 water_var=None,
                 ice_conc_var=None,
                 grid=None,
                 grid_file=None,
                 data_file=None,
                 **kwargs):
        IceAwareProp.__init__(self,
                              name=name,
                              units=units,
                              time=time,
                              ice_var=ice_var,
                              ice_conc_var=ice_conc_var,
                              grid=grid,
                              grid_file=grid_file,
                              data_file=data_file)
        self.water_var = water_var
        if self.name == 'IceAwareProp':
            self.name = 'IceAwareCurrent'

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
                    data_file=None,
                    **kwargs):

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
            grid = _init_grid(grid_file,
                              grid_topology=grid_topology,
                              dataset=dg)
        if water_var is None:
            water_var = GridCurrent.from_netCDF(filename,
                                                time=time,
                                                grid=grid,
                                                dataset=ds,
                                                **kwargs)

        return super(IceAwareCurrent, cls).from_netCDF(grid_topology=grid_topology,
                                                       name=name,
                                                       units=units,
                                                       time=time,
                                                       ice_var=ice_var,
                                                       water_var=water_var,
                                                       ice_conc_var=ice_conc_var,
                                                       grid=grid,
                                                       dataset=ds,
                                                       grid_file=grid_file,
                                                       data_file=data_file,
                                                       **kwargs)

    def at(self, points, time, units=None, extrapolate=False):
        interp = self.ice_conc_var.at(points, time, extrapolate=extrapolate).copy()
        interp_mask = np.logical_and(interp >= 0.2, interp < 0.8)
        if len(interp > 0.2):
            ice_mask = interp >= 0.8

            water_v = self.water_var.at(points, time, units, extrapolate)
            ice_v = self.ice_var.at(points, time, units, extrapolate).copy()
            interp = (interp * 10) / 6 - 0.2

            vels = water_v.copy()
            vels[ice_mask] = ice_v[ice_mask]
            diff_v = ice_v
            diff_v -= water_v
            vels[interp_mask] += diff_v[interp_mask] * interp[interp_mask][:, np.newaxis]
            return vels
        else:
            return self.water_var.at(points, time, units, extrapolate)


class IceAwareWind(IceAwareProp):

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 ice_var=None,
                 wind_var=None,
                 ice_conc_var=None,
                 grid=None,
                 grid_file=None,
                 data_file=None,
                 **kwargs):
        IceAwareProp.__init__(self,
                              name=name,
                              units=units,
                              time=time,
                              ice_var=ice_var,
                              ice_conc_var=ice_conc_var,
                              grid=grid,
                              grid_file=grid_file,
                              data_file=data_file)
        self.wind_var = wind_var
        if self.name == 'IceAwareProp':
            self.name = 'IceAwareWind'

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
                    data_file=None,
                    **kwargs):

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
            grid = _init_grid(grid_file,
                              grid_topology=grid_topology,
                              dataset=dg)
        if wind_var is None:
            wind_var = GridWind.from_netCDF(filename,
                                            time=time,
                                            grid=grid,
                                            dataset=ds,
                                            **kwargs)

        return super(IceAwareWind, cls).from_netCDF(grid_topology=grid_topology,
                                                    name=name,
                                                    units=units,
                                                    time=time,
                                                    ice_var=ice_var,
                                                    wind_var=wind_var,
                                                    ice_conc_var=ice_conc_var,
                                                    grid=grid,
                                                    dataset=ds,
                                                    grid_file=grid_file,
                                                    data_file=data_file,
                                                    **kwargs)

    def at(self, points, time, units=None, extrapolate=False):
        interp = self.ice_conc_var.at(points, time, extrapolate=extrapolate)
        interp_mask = np.logical_and(interp >= 0.2, interp < 0.8)
        if len(interp >= 0.2) != 0:
            ice_mask = interp >= 0.8

            wind_v = self.wind_var.at(points, time, units, extrapolate)
            interp = (interp * 10) / 6 - 0.2

            vels = wind_v.copy()
            vels[ice_mask] = 0
            vels[interp_mask] = vels[interp_mask] * (1 - interp[interp_mask][:, np.newaxis])  # scale winds from 100-0% depending on ice coverage
            return vels
        else:
            return self.wind_var.at(points, time, units, extrapolate)


_valid_temp_units = _valid_units('Temperature')
_valid_dist_units = _valid_units('Length')
_valid_kvis_units = _valid_units('Kinematic Viscosity')
_valid_density_units = _valid_units('Density')
_valid_salinity_units = ('psu',)
_valid_sediment_units = _valid_units('Concentration In Water')


class WaterConditions(Environment, serializable.Serializable):

    _ref_as = 'water'
    _state = copy.deepcopy(Environment._state)
    _units_type = {'temperature': ('temperature', _valid_temp_units),
                   'salinity': ('salinity', _valid_salinity_units),
                   'sediment': ('concentration in water',
                                _valid_sediment_units),
                   'wave_height': ('length', _valid_dist_units),
                   'fetch': ('length', _valid_dist_units),
                   'kinematic_viscosity': ('kinematic viscosity',
                                           _valid_kvis_units),
                   'density': ('density', _valid_density_units),
                   }

    # keep track of valid SI units for properties - these are used for
    # conversion since internal code uses SI units. Don't expect to change
    # these so make it a class level attribute
    _si_units = {'temperature': 'K',
                 'salinity': 'psu',
                 'sediment': 'kg/m^3'}

    def __init__(self,
                 temperature=300.,
                 salinity=35.0,
                 sediment=.005,
                 fetch=0,
                 name='WaterConditions',
                 **kwargs):
        '''
        Assume units are SI for all properties. 'units' attribute assumes SI
        by default. This can be changed, but initialization takes SI.
        '''
        if isinstance(temperature, (Number)):
            self.temperature = TemperatureTS.constant(data=temperature)
        elif isinstance(temperature, (EnvProp)):
            self.temperature = temperature
        else:
            raise TypeError('Temperature is not an environment object or number')
        if isinstance(salinity, (Number)):
            self.salinity = TimeSeriesProp.constant(name='Salinity', units='psu', data=salinity)
        elif isinstance(salinity, (EnvProp)):
            self.salinity = salinity
        else:
            raise TypeError('Salinity is not an environment object or number')
        if isinstance(sediment, (Number)):
            self.sediment = TimeSeriesProp.constant(name='Sediment', units='kg/m^3', data=sediment)
        elif isinstance(sediment, (EnvProp)):
            self.sediment = sediment
        else:
            raise TypeError('Sediment is not an environment object or number')
#         self.wave_height = wave_height
        self.fetch = fetch
        self.kinematic_viscosity = 0.000001
        self.name = 'WaterConditions'
#         self._units = dict(self._si_units)
#         self.units = units

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    grid_topology=None,
                    name=None,
                    temperature=None,
                    salinity=None,
                    sediment=None,
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
            grid = _init_grid(grid_file,
                             grid_topology=grid_topology,
                             dataset=dg)

        if time is None:
            time = ice_var.time

        if temperature is None:
            try:
                temperature = GridTemperature.from_netCDF(filename,
                                                          time=time,
                                                          grid=grid,
                                                          dataset=ds)
            except:
                temperature = 300.
        if salinity is None:
            try:
                salinity = GridSalinity.from_netCDF(filename,
                                                    time=time,
                                                    grid=grid,
                                                    dataset=ds)
            except:
                salinity = 35.
        if sediment is None:
            try:
                sediment = GridSediment.from_netCDF(filename,
                                                    time=time,
                                                    grid=grid,
                                                    dataset=ds)
            except:
                sediment = .005
        if name is None:
            name = 'WaterConditions'
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

    def get(self, attr, unit=None, points=None, time=None, extrapolate=True):  # Arguments should be reorganized eventually
        var = getattr(self, attr)
        if isinstance(var, TimeSeriesProp, TSVectorProp):
            if var.is_constant() or (time is not None):
                return var.at(points, time, units=unit)
            else:
                raise ValueError("Time must be specified to get value from non-constant time series property")
        elif isinstance(var, GriddedProp, GridVectorProp):
            if points is None:
                raise ValueError("Points must be defined to get value from gridded property")
            if time is None:
                raise ValueError("Time must be defined to get value from gridded property")
            return var.at(points, time, units=unit)
        else:
            raise ValueError("var is not a property object")


def Wind(*args, **kwargs):
    '''
    Wind environment object factory function
    '''
    units = kwargs['units'] if 'units' in kwargs else 'm/s'
    name = kwargs['name'] if 'name' in kwargs else 'Wind'

    # Constant wind - Wind(speed=s, direction=d)
    if ('speed' in kwargs and 'direction' in kwargs) or len(args) == 2:
        speed = direction = 0
        if len(args) == 2:
            speed = args[0]
            direction = args[1]
        else:
            speed = kwargs['speed']
            direction = kwargs['direction']
        name = 'Constant Wind' if name is 'Wind' else name
        if isinstance(speed, Number):
            return WindTS.constant(name, units, speed, direction)
        else:
            raise TypeError('speed must be a single value. For a timeseries, use timeseries=[(t0, (mag,dir)),(t1, (mag,dir))]')

    # Time-varying wind - Wind(timeseries=[(t0, (mag,dir)),(t1, (mag,dir)),...])
    if ('timeseries' in kwargs):
        name = 'Wind Time Series' if name is 'Wind' else name
        return WindTS(name=name, units=units, timeseries=kwargs['timeseries'])

    # Gridded Wind - Wind(filename=fn, dataset=ds)
    if ('filename' in kwargs or 'dataset' in kwargs):
        if 'ice_aware' in kwargs:
            # Ice Aware Gridded Wind - Wind(filename=fn, dataset=ds, ice_aware=True)
            name = 'IceAwareWind' if name is 'Wind' else name
            return IceAwareWind.from_netCDF(**kwargs)
        name = 'GridWind' if name is 'Wind' else name
        return GridWind.from_netCDF(**kwargs)

    # Arguments do not trigger a factory route. Attempt construction using default class __init__s
    w = None
    warnings.warn('Attempting default Wind constructions')
    return _attempt_construction([WindTS, GridWind], **kwargs)


def Current(*args, **kwargs):
    '''
    Wind environment object factory function
    '''
    units = kwargs['units'] if 'units' in kwargs else 'm/s'
    name = kwargs['name'] if 'name' in kwargs else 'Current'

    # Constant wind - Wind(speed=s, direction=d)
    if ('speed' in kwargs and 'direction' in kwargs) or len(args) == 2:
        speed = direction = 0
        if len(args) == 2:
            speed = args[0]
            direction = args[1]
        else:
            speed = kwargs['speed']
            direction = kwargs['direction']
        name = 'Constant Current' if name is 'Wind' else name
        if isinstance(speed, Number):
            return CurrentTS.constant_current(name, units, speed, direction)
        else:
            raise TypeError('speed must be a single value. For a timeseries, use timeseries=[(t0, (mag,dir)),(t1, (mag,dir))]')

    # Time-varying wind - Wind(timeseries=[(t0, (mag,dir)),(t1, (mag,dir)),...])
    if ('timeseries' in kwargs):
        name = 'Current Time Series' if name is 'Current' else name
        return CurrentTS(name=name, units=units, timeseries=kwargs['timeseries'])

    # Gridded Wind - Wind(filename=fn, dataset=ds)
    if ('filename' in kwargs or 'dataset' in kwargs):
        if 'ice_aware' in kwargs:
            # Ice Aware Gridded Current - Current(filename=fn, dataset=ds, ice_aware=True)
            name = 'IceAwareCurrent' if name is 'Current' else name
            return IceAwareCurrent.from_netCDF(**kwargs)
        name = 'GridCurrent' if name is 'Current' else name
        return GridCurrent.from_netCDF(**kwargs)

    # Arguments do not trigger a factory route. Attempt construction using default class __init__s
    w = None
    warnings.warn('Attempting default Wind constructions')
    return _attempt_construction([CurrentTS, GridCurrent], **kwargs)


def Temperature(*args, **kwargs):
    units = kwargs['units'] if 'units' in kwargs else 'K'
    name = kwargs['name'] if 'name' in kwargs else 'WaterTemp'
    kwargs['units'] = units
    kwargs['name'] = name
    # Constant Temperature - WaterTemp(temp=t)
    if ('temp' in kwargs or 'temperature' in kwargs) or len(args) == 1:
        temp = 0
        if len(args) == 1:
            temp = args[0]
        else:
            temp = kwargs['temp'] if 'temp' in kwargs else kwargs['temperature']
        name = 'Constant Temperature' if name is 'Temperature' else name
        if isinstance(speed, Number):
            return TemperatureTS.constant_temp(name, units, temp)
        else:
            raise TypeError('temperature must be a single value. For a timeseries, use timeseries=[(t1, temp1), (t2, temp2),...]')

    # Time-varying temp - Temperature(timeseries=[(t1, temp1), (t2, temp2),...])
    if ('timeseries' in kwargs):
        name = 'Temperature Time Series' if name is 'Temperature' else name
        return TemperatureTS(name=name, units=units, timeseries=kwargs['timeseries'])

    # Gridded Temp - Temperature(filename=fn, dataset=ds)
    if ('filename' in kwargs or 'dataset' in kwargs):
        if 'ice_aware' in kwargs:
            # Ice Aware Gridded Wind - Wind(filename=fn, dataset=ds, ice_aware=True)
            name = 'IceAwareWaterTemp' if name is 'Temperature' else name
            return IceAwareWaterTemperature.from_netCDF(**kwargs)
        name = 'GridTemperature' if name is 'Temperature' else name
        return GridTemperature.from_netCDF(**kwargs)

    # Arguments do not trigger a factory route. Attempt construction using default class __init__s
    w = None
    warnings.warn('Attempting default Temperature constructions')
    return _attempt_construction([TemperatureTS, GridTemperature], **kwargs)


def _attempt_construction(types, **kwargs):
    for t in types:
        try:
            prop = t(**kwargs)
            return prop
        except Error as e:
            print('t.__name__ construction failed: {0}'.format(e))
    raise RuntimeError('Unable to build any type of environment object using the arguments provided. Please see the documentation for the usage of')

