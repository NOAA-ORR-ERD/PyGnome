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
from gnome.environment.grid import PyGrid
from gnome.environment.property import Time, PropertySchema, VectorProp, EnvProp
from gnome.environment.ts_property import TSVectorProp, TimeSeriesProp, TimeSeriesPropSchema
from gnome.environment.grid_property import GridVectorProp, GriddedProp, GridPropSchema, GridVectorPropSchema
from gnome.utilities.file_tools.data_helpers import _get_dataset


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


class S_Depth_T1(object):

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
        '''
        Returns a pair of values. The 1st value is an array of the depth indices of all the particles.
        The 2nd value is an array of the interpolation alphas for the particles between their depth
        index and depth_index+1. If both values are None, then all particles are on the surface layer.
        '''
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


class VelocityTS(TSVectorProp):

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
        :param units='m/s': units for speed, as a string, i.e. "knots", "m/s",
                           "cm/s", etc.

        .. note:: 
            The time for a constant wind timeseries is irrelevant. This
            function simply sets it to datetime.now() accurate to hours.
        """
        direction = direction * -1 - 90
        u = speed * np.cos(direction * np.pi / 180)
        v = speed * np.sin(direction * np.pi / 180)
        u = TimeSeriesProp.constant('u', units, u)
        v = TimeSeriesProp.constant('v', units, v)
        return super(VelocityTS, cls).constant(name, units, variables=[u, v])

    @property
    def timeseries(self):
        x = self.variables[0].data
        y = self.variables[1].data
        return map(lambda t, x, y: (t, (x, y)), self._time, x, y)

#     def serialize(self, json_='webapi'):
#         dict_ = serializable.Serializable.serialize(self, json_=json_)
#         # The following code is to cover the needs of webapi
#         if json_ == 'webapi':
#             dict_.pop('timeseries')
#             dict_.pop('units')
#             x = np.asanyarray(self.variables[0].data)
#             y = np.asanyarray(self.variables[1].data)
#             direction = -(np.arctan2(y, x) * 180 / np.pi + 90)
#             magnitude = np.sqrt(x ** 2 + y ** 2)
#             ts = (unicode(tx.isoformat()) for tx in self._time)
#             dict_['timeseries'] = map(lambda t, x, y: (t, (x, y)), ts, magnitude, direction)
#             dict_['units'] = (unicode(self.variables[0].units), u'degrees')
#             dict_['varnames'] = [u'magnitude', u'direction', dict_['varnames'][0], dict_['varnames'][1]]
#         return dict_

#     @classmethod
#     def deserialize(cls, json_):
#         if json_ == 'webapi':
#             dict_ = super(VelocityTS, cls).deserialize(json_)
#     
#             ts, data = zip(*dict_.pop('timeseries'))
#             ts = np.array(ts)
#             data = np.array(data).T
#             units = dict_['units']
#             if len(units) > 1 and units[1] == 'degrees':
#                 u_data, v_data = data
#                 v_data = ((-v_data - 90) * np.pi / 180)
#                 u_t = u_data * np.cos(v_data)
#                 v_data = u_data * np.sin(v_data)
#                 u_data = u_t
#                 data = np.array((u_data, v_data))
#                 dict_['varnames'] = dict_['varnames'][2:]
#     
#             units = units[0]
#             dict_['units'] = units
#             dict_['time'] = ts
#             dict_['data'] = data
#             return dict_
#         else:
#             return super(VelocityTS, cls).deserialize(json_)
# 
#     @classmethod
#     def new_from_dict(cls, dict_):
#         varnames = dict_['varnames']
#         vs = []
#         for i, varname in enumerate(varnames):
#             vs.append(TimeSeriesProp(name=varname,
#                                      units=dict_['units'],
#                                      time=dict_['time'],
#                                      data=dict_['data'][i]))
#         dict_.pop('data')
#         dict_['variables'] = vs
#         return super(VelocityTS, cls).new_from_dict(dict_)


class VelocityGrid(GridVectorProp):

    def __init__(self, **kwargs):
        if 'variables' in kwargs:
            variables = kwargs['variables']
            if len(variables) == 2:
                variables.append(TimeSeriesProp(name='constant w', data=[0.0], time=Time.constant_time(), units='m/s'))
            kwargs['variables'] = variables
        self.angle = None
        df = None
        if kwargs.get('dataset', None) is not None:
            df = kwargs['dataset']
        elif kwargs.get('grid_file', None) is not None:
            df = _get_dataset(kwargs['grid_file'])
        if df is not None and 'angle' in df.variables.keys():
            # Unrotated ROMS Grid!
            self.angle = GriddedProp(name='angle', units='radians', time=None, grid=self.grid, data=df['angle'])
        super(VelocityGrid, self).__init__(**kwargs)

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


class IceConcentration(GriddedProp, Environment):

    default_names = ['ice_fraction', ]

    def __init__(self, *args, **kwargs):
        super(IceConcentration, self).__init__(*args, **kwargs)

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

    def at(self, points, time, units=None, extrapolate=False, **kwargs):
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

    def at(self, points, time, units=None, extrapolate=False, **kwargs):
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
        value[points[:, 2] > 0.0] = 0  # no wind underwater!
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


class IceAwarePropSchema(GridVectorPropSchema):
    ice_conc_var = GridPropSchema(missing=drop)


class IceAwareCurrentSchema(IceAwarePropSchema):
    ice_var = GridVectorPropSchema(missing=drop)


class IceAwareCurrent(GridCurrent):

    _ref_as = 'current'
    _req_refs = [IceConcentration, IceVelocity]

    _schema = IceAwareCurrentSchema
    _state = copy.deepcopy(GridCurrent._state)

    _state.add_field([serializable.Field('ice_var', save=True, update=True, save_reference=True),
                      serializable.Field('ice_conc_var', save=True, update=True, save_reference=True)])

    def __init__(self,
                 ice_var=None,
                 ice_conc_var=None,
                 *args,
                 **kwargs):
        self.ice_var = ice_var
        self.ice_conc_var = ice_conc_var
        super(IceAwareCurrent, self).__init__(*args, **kwargs)

    def _attach_refs(self, _dict):
        self.ice_conc_var = _dict[IceConcentration] if self.ice_conc_var is None else self.ice_conc_var
        self.ice_var = _dict[IceVelocity] if self.ice_var is None else self.ice_var

    @classmethod
    @GridCurrent._get_shared_vars()
    def from_netCDF(cls,
                    ice_conc_var=None,
                    ice_var=None,
                    **kwargs):
        if ice_conc_var is None:
            ice_conc_var = IceConcentration.from_netCDF(**kwargs)
        if ice_var is None:
            ice_var = IceVelocity.from_netCDF(**kwargs)
        return super(IceAwareCurrent, cls).from_netCDF(ice_conc_var=ice_conc_var,
                                                       ice_var=ice_var,
                                                       **kwargs)

    def at(self, points, time, units=None, extrapolate=False):
        interp = self.ice_conc_var.at(points, time, extrapolate=extrapolate).copy()
        interp_mask = np.logical_and(interp >= 0.2, interp < 0.8)
        if len(interp > 0.2):
            ice_mask = interp >= 0.8

            water_v = super(IceAwareCurrent, self).at(points, time, units, extrapolate)
            ice_v = self.ice_var.at(points, time, units, extrapolate).copy()
            interp = (interp * 10) / 6 - 0.2

            vels = water_v.copy()
            vels[ice_mask] = ice_v[ice_mask]
            diff_v = ice_v
            diff_v -= water_v
            vels[interp_mask] += diff_v[interp_mask] * interp[interp_mask][:, np.newaxis]
            return vels
        else:
            return super(IceAwareCurrent, self).at(points, time, units, extrapolate)


class IceAwareWind(GridWind):

    _ref_as = 'wind'
    _req_refs = [IceConcentration, IceVelocity]

    _schema = IceAwarePropSchema
    _state = copy.deepcopy(GridWind._state)

    _state.add_field([serializable.Field('ice_conc_var', save=True, update=True, save_reference=True)])

    def __init__(self,
                 ice_conc_var=None,
                 *args,
                 **kwargs):
        self.ice_conc_var = ice_conc_var
        super(IceAwareWind, self).__init__(*args, **kwargs)

    @classmethod
    @GridCurrent._get_shared_vars()
    def from_netCDF(cls,
                    ice_conc_var=None,
                    ice_var=None,
                    **kwargs):
        if ice_conc_var is None:
            ice_conc_var = IceConcentration.from_netCDF(**kwargs)
        if ice_var is None:
            ice_var = IceVelocity.from_netCDF(**kwargs)
        return super(IceAwareWind, cls).from_netCDF(ice_conc_var=ice_conc_var,
                                                    ice_var=ice_var,
                                                    **kwargs)

    def at(self, points, time, units=None, extrapolate=False):
        interp = self.ice_conc_var.at(points, time, extrapolate=extrapolate)
        interp_mask = np.logical_and(interp >= 0.2, interp < 0.8)
        if len(interp >= 0.2) != 0:
            ice_mask = interp >= 0.8

            wind_v = super(IceAwareWind, self).at(points, time, units, extrapolate)
            interp = (interp * 10) / 6 - 0.2

            vels = wind_v.copy()
            vels[ice_mask] = 0
            vels[interp_mask] = vels[interp_mask] * (1 - interp[interp_mask][:, np.newaxis])  # scale winds from 100-0% depending on ice coverage
            return vels
        else:
            return super(IceAwareWind, self).at(points, time, units, extrapolate)

def load_all_from_netCDF(filename=None,
                         grid_topology=None,
                         name=None,
                         time=None,
                         grid=None,
                         depth=None,
                         dataset=None,
                         data_file=None,
                         grid_file=None,
                         **kwargs):
    pass