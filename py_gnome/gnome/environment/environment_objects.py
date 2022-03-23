
import copy
from datetime import datetime

import netCDF4 as nc4
import numpy as np

from colander import drop
from gnome.persist import (Boolean,
                           SchemaNode,
                           ObjTypeSchema,
                           FilenameSchema,
                           )


import gridded
from gnome.gnomeobject import combine_signatures
import unit_conversion as uc

from .environment import Environment
from .timeseries_objects_base import TimeseriesData, TimeseriesVector

from .gridded_objects_base import (Time,
                                   Variable,
                                   VectorVariable,
                                   VariableSchema,
                                   VectorVariableSchema,
                                   LocalDateTime,
                                   )
from gnome.persist.validators import convertible_to_seconds

from .gridcur import init_from_gridcur, GridCurReadError


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
                    raise ValueError('Need data_file or dataset '
                                     'containing sigma equation terms')

            ds = gridded.utilities.get_dataset(data_file)

        self.bathymetry = bathymetry
        self.terms = terms

        if len(terms) == 0:
            for s in S_Depth_T1.default_terms:
                for term in s:
                    self.terms[term] = ds[term][:]

    @classmethod
    def from_netCDF(cls, **kwargs):
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
        Returns a pair of values.

        - The 1st value is an array of the depth indices of all the
          particles.

        - The 2nd value is an array of the interpolation alphas for the
          particles between their depth index and depth_index + 1.

        - If both values are None, then all particles are on the
          surface layer.
        '''
        underwater = points[:, 2] > 0.0

        if len(np.where(underwater)[0]) == 0:
            return None, None

        indices = -np.ones((len(points)), dtype=np.int64)
        alphas = -np.ones((len(points)), dtype=np.float64)
        depths = self.bathymetry.at(points,
                                    datetime.now(),
                                    _hash=_hash)[underwater]
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
            raise ValueError('Cannot get depth interpolation alphas '
                             'for data shape specified; '
                             'does not fit r or w depth axis')

        blev_depths = ulev_depths = None

        for ulev in range(0, num_levels):
            ulev_depths = ldgb(depths, ulev)

            within_layer = np.where(np.logical_and(ulev_depths < pts[:, 2],
                                                   und_ind == -1))[0]

            und_ind[within_layer] = ulev

            if ulev == 0:
                und_alph[within_layer] = -2
            else:
                a = ((pts[:, 2].take(within_layer) -
                      blev_depths.take(within_layer)) /
                     (ulev_depths.take(within_layer) -
                      blev_depths.take(within_layer)))
                und_alph[within_layer] = a
            blev_depths = ulev_depths

        indices[underwater] = und_ind
        alphas[underwater] = und_alph

        return indices, alphas


@combine_signatures
class VelocityTS(TimeseriesVector):

    _gnome_unit = 'm/s'

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

        u = TimeseriesData.constant('u', units, u)
        v = TimeseriesData.constant('v', units, v)

        return super(VelocityTS, cls).constant(name, units, variables=[u, v])

    @property
    def timeseries(self):
        x = self.variables[0].data
        y = self.variables[1].data
        return map(lambda t, x, y: (t, (x, y)), self._time, x, y)


class VelocityGrid(VectorVariable):

    _gnome_unit = 'm/s'
    comp_order = ['u', 'v', 'w']

    @combine_signatures
    def __init__(self, angle=None, **kwargs):
        """
            :param angle: scalar field of cell rotation angles (for rotated/distorted grids)
        """

        if 'variables' in kwargs:
            variables = kwargs['variables']
            if len(variables) == 2:
                variables.append(TimeseriesData(name='constant w',
                                                data=[0.0],
                                                time=Time.constant_time(),
                                                units='m/s'))

            kwargs['variables'] = variables

        if angle is None:
            df = None

            if kwargs.get('dataset', None) is not None:
                df = kwargs['dataset']
            elif kwargs.get('grid_file', None) is not None:
                df = gridded.utilities.get_dataset(kwargs['grid_file'])

            if df is not None and 'angle' in df.variables.keys():
                # Unrotated ROMS Grid!
                self.angle = Variable(name='angle',
                                      units='radians',
                                      time=Time.constant_time(),
                                      grid=kwargs['grid'],
                                      data=df['angle'])
            else:
                self.angle = None
        else:
            self.angle = angle

        super(VelocityGrid, self).__init__(**kwargs)


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
        return super(WindTS, self).constant(name=name, speed=speed,
                                            direction=direction, units=units)


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
    def constant_current(cls,
                         name='Constant Current',
                         speed=None,
                         direction=None,
                         units='m/s'):
        """
        :param speed: speed of current
        :param direction: direction -- degrees True, direction current is going
                          (degrees True)
        :param unit='m/s': units for speed, as a string, i.e. "knots", "m/s",
                           "cm/s", etc.

        """
        return cls.constant(name=name, speed=speed, direction=direction,
                            units=units)


class TemperatureTS(TimeseriesData, Environment):

    _gnome_unit = 'K'

    def __init__(self, name=None, units='K',
                 time=None, data=None,
                 **kwargs):
        if 'timeseries' in kwargs:
            ts = kwargs['timeseries']

            time = map(lambda e: e[0], ts)
            data = np.array(map(lambda e: e[1], ts))

        TimeseriesData.__init__(self, name, units, time, data=data)

    @classmethod
    def constant_temperature(cls,
                             name='Constant Temperature',
                             temperature=None,
                             units='K'):
        return cls.constant(name=name, data=temperature, units=units)


class GridTemperature(Variable, Environment):
    default_names = ['water_t', 'temp']

    cf_names = ['sea_water_temperature', 'sea_surface_temperature']

    _gnome_unit = 'K'
    _default_unit_type = 'Temperature'


class SalinityTS(TimeseriesData, Environment):
    _gnome_unit = 'ppt'

    @classmethod
    def constant_salinity(cls,
                          name='Constant Salinity',
                          salinity=None,
                          units='ppt'):
        return cls.constant(name=name, data=salinity, units=units)


class GridSalinity(Variable, Environment):
    default_names = ['salt']

    cf_names = ['sea_water_salinity', 'sea_surface_salinity']
    _gnome_unit = 'ppt'


class WaterDensityTS(TimeseriesData, Environment):
    _gnome_unit = 'kg/m^3'

    def __init__(self,
                 name=None,
                 units='kg/m^3',
                 temperature=None,
                 salinity=None):
        if (temperature is None or
                salinity is None or
                not isinstance(temperature, TemperatureTS) or
                not isinstance(salinity, SalinityTS)):
            raise ValueError('Must provide temperature and salinity '
                             'time series Environment objects')

        if len(temperature.time.time) > len(salinity.time.time):
            density_times = temperature.time
        else:
            density_times = salinity.time

        dummy_pt = np.array([[0, 0], ])

        import gsw
        from gnome import constants

        data = [gsw.rho(salinity.at(dummy_pt, t),
                        temperature.at(dummy_pt, t, units='C'),
                        constants.atmos_pressure * 0.0001)
                for t in density_times.time]

        TimeseriesData.__init__(self, name, units, time=density_times,
                                data=data)


class GridSediment(Variable, Environment):
    _gnome_unit = 'ppt'
    default_names = ['sand_06']


class IceConcentration(Variable, Environment):
    _ref_as = ['ice_concentration', 'ice_aware']
    default_names = ['ice_fraction', 'aice' ]
    cf_names = ['sea_ice_area_fraction']
    _gnome_unit = 'fraction'

    def __init__(self, *args, **kwargs):
        super(IceConcentration, self).__init__(*args, **kwargs)


class Bathymetry(Variable):
    _gnome_unit = 'm'
    default_names = ['h']
    cf_names = ['depth']


class GridCurrent(VelocityGrid, Environment):
    """
    GridCurrent is VelocityGrid that adds specific stuff for currents:

    - Information about how to find currents in netCDF file
    - Ability to apply an angle adjustment of grid-aligned currents
    - overloading the memorization to memoize the angle-adjusted current.
    - add a get_data_vectors() provides  magnitude, direction -- used to
      draw the currents in a GUI

    loading code for netcdf files
    for gridded currents, and an interpolation (`.at`), function that provides caching
    """

    _ref_as = 'current'
    _gnome_unit = 'm/s'
    default_names = {'u': ['u', 'U', 'water_u', 'curr_ucmp', 'u_surface', 'u_sur'],
                     'v': ['v', 'V', 'water_v', 'curr_vcmp', 'v_surface', 'v_sur'],
                     'w': ['w', 'W']}
    cf_names = {'u': ['eastward_sea_water_velocity',
                      'surface_eastward_sea_water_velocity'],
                'v': ['northward_sea_water_velocity',
                      'surface_northward_sea_water_velocity'],
                'w': ['upward_sea_water_velocity']}

    def at(self, points, time, *args, **kwargs):
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
            res = self._get_memoed(points, time,
                                   self._result_memo, _hash=_hash)
            if res is not None:
                return res

        extrapolate = self.extrapolation_is_allowed

        value = super(GridCurrent, self).at(points, time,
                                            extrapolate=extrapolate,
                                            **kwargs)

        if self.angle is not None:
            angs = (self.angle.at(points, time, extrapolate=extrapolate,
                                  **kwargs)
                    .reshape(-1))

            if 'degree' in self.angle.units:
                angs = angs * np.pi/180.

            x = value[:, 0] * np.cos(angs) - value[:, 1] * np.sin(angs)
            y = value[:, 0] * np.sin(angs) + value[:, 1] * np.cos(angs)

            value[:, 0] = x
            value[:, 1] = y

        value[:, 2][points[:, 2] == 0.0] = 0

        if mem:
            self._memoize_result(points, time, value,
                                 self._result_memo, _hash=_hash)

        return value

    def get_data_vectors(self):
        '''
        return array of shape (2, time_slices, len_linearized_data)
        first is magnitude, second is direction
        '''

        if(hasattr(self, 'angle') and self.angle):

            raw_uv = super(GridCurrent, self).get_data_vectors()
            lin_u = raw_uv[0,:,:]
            lin_v = raw_uv[1,:,:]

            raw_ang = self.angle.data[:]
            angle_padding_slice = self.grid.get_padding_slices(self.grid.center_padding)
            raw_ang = raw_ang[angle_padding_slice]

            if 'degree' in self.angle.units:
                raw_ang = raw_ang * np.pi/180.

            ctr_mask = gridded.utilities.gen_celltree_mask_from_center_mask(self.grid.center_mask, angle_padding_slice)
            ang = raw_ang.reshape(-1)
            ang = np.ma.MaskedArray(ang, mask = ctr_mask.reshape(-1))
            ang = ang.compressed()

            x = lin_u[:] * np.cos(ang) - lin_v[:] * np.sin(ang)
            y = lin_u[:] * np.sin(ang) + lin_v[:] * np.cos(ang)
            r = np.concatenate((x[None,:], y[None,:]))
            return np.ascontiguousarray(r.astype(np.float32)) # r.compressed().astype(np.float32)
            # return np.ascontiguousarray(r.filled(0), np.float32)

        else:
            return super(GridCurrent, self).get_data_vectors()


class GridWind(VelocityGrid, Environment):
    """
    Gridded winds -- usually from netcdf files from meteorological models.

    This will most often be initialized from netcdf files as:

    wind = GridWind.from_netCDF(filename="a/path/to/a/netcdf_file.nc")

    filename can be:

    * An already open netCDF4 Dataset

    * A single path to a netcdf file

    * A list of paths to a set of netcdf files with timeseries
    """
    _ref_as = 'wind'
    _gnome_unit = 'm/s'
    default_names = {'u': ['air_u', 'Air_U', 'air_ucmp', 'wind_u'],
                     'v': ['air_v', 'Air_V', 'air_vcmp', 'wind_v']}

    cf_names = {'u': ['eastward_wind', 'eastward wind'],
                'v': ['northward_wind', 'northward wind']}

    def __init__(self,
                 wet_dry_mask=None,
                 *args, **kwargs):
        super(GridWind, self).__init__(*args, **kwargs)


        if wet_dry_mask is not None:
            if self.grid.infer_location(wet_dry_mask) != 'center':
                raise ValueError('Wet/Dry mask does not correspond to '
                                 'grid cell centers')

        self.wet_dry_mask = wet_dry_mask

    def at(self, points, time,
           coord_sys='uv', _auto_align=True, **kwargs):
        '''
        Find the value of the property at positions P at time T

        :param points: Coordinates to be queried (P)
        :type points: Nx2 array of double

        :param time: The time at which to query these points (T)
        :type time: datetime.datetime object

        :param depth: Specifies the depth level of the variable
        :type depth: integer

        :param units: units the values will be returned in (or converted to)
        :type units: string such as ('m/s', 'knots', etc)

        :param extrapolate: if True, extrapolation will be supported
        :type extrapolate: boolean (True or False)

        :param coord_sys: String describing the coordinate system to be used.
        :type coord_sys: string, one of ('uv','u','v','r-theta','r','theta')

        :return: returns a Nx2 array of interpolated values
        :rtype: double
        '''
        pts = gridded.utilities._reorganize_spatial_data(points)
        value = None
        has_depth = pts.shape[1] > 2

        mem = kwargs['memoize'] if 'memoize' in kwargs else True
        _hash = kwargs['_hash'] if '_hash' in kwargs else None

        if _hash is None:
            _hash = self._get_hash(pts, time)
            if '_hash' not in kwargs:
                kwargs['_hash'] = _hash

        if mem:
            res = self._get_memoed(pts, time,
                                   self._result_memo, _hash=_hash)
            if res is not None:
                value = res
                if _auto_align:
                    value = (gridded.utilities
                             ._align_results_to_spatial_data(value, points))
                return self.transform_result(value, coord_sys)

        if value is None:
            extrapolate = self.extrapolation_is_allowed

            value = super(GridWind, self).at(pts, time,
                                             extrapolate=extrapolate,
                                             _auto_align=False, **kwargs)

            if has_depth:
                value[pts[:, 2] > 0.0] = 0  # no wind underwater!

            if self.angle is not None:
                angs = (self.angle.at(pts, time,
                                      extrapolate=extrapolate,
                                      _auto_align=False,
                                      **kwargs)
                        .reshape(-1))

                x = value[:, 0] * np.cos(angs) - value[:, 1] * np.sin(angs)
                y = value[:, 0] * np.sin(angs) + value[:, 1] * np.cos(angs)

                value[:, 0] = x
                value[:, 1] = y

        if _auto_align:
            value = gridded.utilities._align_results_to_spatial_data(value,
                                                                     points)

        if mem:
            self._memoize_result(pts, time, value, self._result_memo,
                                 _hash=_hash)

        return self.transform_result(value, coord_sys)

    def transform_result(self, value, coord_sys):
        #internally all results are computed and memoized in 'uv'
        #this function transforms those to the alternates before returning
        rv = value
        if coord_sys == 'u':
            rv = value[:, 0][:, None]
        elif coord_sys == 'v':
            rv = value[:, 1][:, None]
        elif coord_sys in ('r-theta', 'r', 'theta'):
            _mag = np.sqrt(value[:, 0] ** 2 + value[:, 1] ** 2)
            _dir = np.arctan2(value[:, 1], value[:, 0]) * 180. / np.pi

            if coord_sys == 'r':
                rv = _mag[:, None]
            elif coord_sys == 'theta':
                rv = _dir[:, None]
            else:
                rv = np.column_stack((_mag, _dir))
        return rv


    def get_start_time(self):
        return self.time.min_time

    def get_end_time(self):
        return self.time.max_time


class LandMask(Variable):

    '''
    This class depends on gridded features not yet finalized so it likely non-functional
    '''

    def __init__(self, *args, **kwargs):
        data = kwargs.pop('data', None)

        if data is None or not isinstance(data, (np.ma.MaskedArray,
                                                 nc4.Variable,
                                                 np.ndarray)):
            raise ValueError('Must provide a '
                             'netCDF4 Variable, '
                             'masked numpy array, or '
                             'an explicit mask on nodes or faces')

        if isinstance(data, np.ma.MaskedArray):
            data = data.mask

        kwargs['data'] = data

    def at(self, points, time, units=None, extrapolate=False,
           _hash=None, _mem=True, **kwargs):

        if _hash is None:
            _hash = self._get_hash(points, time)

        if _mem:
            res = self._get_memoed(points, time,
                                   self._result_memo, _hash=_hash)
            if res is not None:
                return res

        # TODO: Why are these here?  idxs and time_idx not used.
        _idxs = self.grid.locate_faces(points)
        _time_idx = self.time.index_of(time)
        order = self.dimension_ordering

        if order[0] == 'time':
            value = self._time_interp(points, time, extrapolate,
                                      _mem=_mem, _hash=_hash, **kwargs)
        elif order[0] == 'depth':
            value = self._depth_interp(points, time, extrapolate,
                                       _mem=_mem, _hash=_hash, **kwargs)
        else:
            value = self._xy_interp(points, time, extrapolate,
                                    _mem=_mem, _hash=_hash, **kwargs)

        if _mem:
            self._memoize_result(points, time, value,
                                 self._result_memo, _hash=_hash)

        return value


class IceVelocity(VelocityGrid, Environment):
    _ref_as = ['ice_velocity', 'ice_aware']
    _gnome_unit = 'm/s'
    default_names = {'u': ['ice_u','uice'],
                     'v': ['ice_v','vice']}

    cf_names = {'u': ['eastward_sea_ice_velocity'],
                'v': ['northward_sea_ice_velocity']}


class IceAwarePropSchema(VectorVariableSchema):
    ice_concentration = VariableSchema(
        missing=drop,
        save=True,
        update=True,
        save_reference=True,
    )


class IceAwareCurrentSchema(IceAwarePropSchema):
    ice_velocity = VectorVariableSchema(
        missing=drop,
        save=True,
        update=True,
        save_reference=True,
    )


class IceAwareCurrent(GridCurrent):
    """
    IceAwareCurrent is a GridCurrent that modulates the usual water velocity field
    using ice velocity and concentration information.

    While under 20% ice coverage, queries will return water velocity.
    Between 20% and 80% coverage, queries will interpolate linearly between water and ice velocity
    Above 80% coverage, queries will return the ice velocity.
    """

    _ref_as = ['current', 'ice_aware']
    _req_refs = {'ice_concentration': IceConcentration,
                 'ice_velocity': IceVelocity}

    _schema = IceAwareCurrentSchema

    def __init__(self,
                 ice_velocity=None,
                 ice_concentration=None,
                 *args,
                 **kwargs):
        """
        :param ice_velocity: VectorVariable representing surface ice velocity
        :type ice_velocity: VectorVariable or compatible object
        :param ice_concentration: Variable representing surface ice concentration
        :type ice_concentration: Variable or compatible object
        """

        self.ice_velocity = ice_velocity
        self.ice_concentration = ice_concentration

        super(IceAwareCurrent, self).__init__(*args, **kwargs)

    @classmethod
    def from_netCDF(cls,
                    *args,
                    **kwargs):
        var = cls.__new__(cls)
        var.init_from_netCDF(*args, **kwargs)
        return var

    @GridCurrent._get_shared_vars()
    def init_from_netCDF(self,
                         ice_file=None,
                         ice_concentration=None,
                         ice_velocity=None,
                         *args,
                         **kwargs):
        temp_fn = None
        if ice_file is not None:
            temp_fn = kwargs['filename']
            kwargs['filename'] = ice_file
        if ice_concentration is None:
            ice_concentration = IceConcentration.from_netCDF(**kwargs)

        if ice_velocity is None:
            ice_velocity = IceVelocity.from_netCDF(**kwargs)

        if temp_fn is not None:
            kwargs['filename'] = temp_fn

        super(IceAwareCurrent, self).init_from_netCDF(
            ice_concentration=ice_concentration,
            ice_velocity=ice_velocity,
            **kwargs
        )

    def at(self, points, time, *args, **kwargs):
        extrapolate = self.extrapolation_is_allowed
        cctn = (self.ice_concentration.at(points, time,
                                            extrapolate=extrapolate, **kwargs)
                  .copy())

        water_v = super(IceAwareCurrent, self).at(points,
                                                  time,
                                                  *args,
                                                  **kwargs)

        if np.any(cctn >= 0.2):
            ice_mask = cctn >= 0.8
            water_mask = cctn < 0.2
            interp_mask = np.logical_and(cctn >= 0.2, cctn < 0.8)

            ice_vel_factor = cctn.copy()
            ice_vel_factor[ice_mask] = 1
            ice_vel_factor[water_mask] = 0
            ice_vel_factor[interp_mask] = ((ice_vel_factor[interp_mask] - 0.2) * 10) / 6

            vels = water_v.copy()
            ice_v = self.ice_velocity.at(points, time, extrapolate=extrapolate, *args, **kwargs).copy()

            #deals with the >0.8 concentration case
            vels[:] = vels[:] + (ice_v - water_v) * ice_vel_factor[:,None]

            return vels
        else:
            return water_v


class IceAwareWind(GridWind):

    _ref_as = ['wind', 'ice_aware']
    _req_refs = {'ice_concentration': IceConcentration}

    _schema = IceAwarePropSchema

    def __init__(self,
                 ice_concentration=None,
                 *args,
                 **kwargs):
        self.ice_concentration = ice_concentration

        super(IceAwareWind, self).__init__(*args, **kwargs)

    @classmethod
    @GridWind._get_shared_vars()
    def from_netCDF(cls,
                    ice_concentration=None,
                    ice_velocity=None,
                    **kwargs):
        if ice_concentration is None:
            ice_concentration = IceConcentration.from_netCDF(**kwargs)

        if ice_velocity is None:
            ice_velocity = IceVelocity.from_netCDF(**kwargs)

        return (super(IceAwareWind, cls)
                .from_netCDF(ice_concentration=ice_concentration,
                             ice_velocity=ice_velocity,
                             **kwargs))

    def at(self, points, time, *args, **kwargs):
        extrapolate = self.extrapolation_is_allowed

        cctn = self.ice_concentration.at(points, time, extrapolate=extrapolate, *args, **kwargs)
        wind_v = super(IceAwareWind, self).at(points, time, *args, **kwargs)

        if np.any(cctn >= 0.2):
            ice_mask = cctn >= 0.8
            water_mask = cctn < 0.2
            interp_mask = np.logical_and(cctn >= 0.2, cctn < 0.8)

            ice_vel_factor = cctn.copy()
            ice_vel_factor[ice_mask] = 1
            ice_vel_factor[water_mask] = 0
            ice_vel_factor[interp_mask] = ((ice_vel_factor[interp_mask] - 0.2) * 10) / 6

            vels = wind_v.copy()
            vels[ice_mask] = 0

            # scale winds from 100-0% depending on ice coverage
            # 100% wind up to 0.2 coverage, 0% wind at >0.8 coverage
            vels[:] = vels[:] * (1 - ice_vel_factor[:,None])

            return vels
        else:
            return wind_v



class FileGridCurrentSchema(ObjTypeSchema):
    filename = FilenameSchema(
        isdatafile=True, test_equal=False, update=False
    )
    extrapolation_is_allowed = SchemaNode(Boolean())

    data_start = SchemaNode(LocalDateTime(), read_only=True,
                            validator=convertible_to_seconds)
    data_stop = SchemaNode(LocalDateTime(), read_only=True,
                           validator=convertible_to_seconds)



class FileGridCurrent(GridCurrent):
    """
    class that presents an interface for GridCurrent loaded from
    files of various formats

    Done as a class to provide a Schema for the persistence system

    And to provide a simple interface to making a current from a file.
    """
    _schema = FileGridCurrentSchema

    def __init__(self, filename=None, extrapolation_is_allowed=False, *args, **kwargs):

        #FileGridCurrent('filename.nc')
        #FileGridCurrent('filename.nc', extrapolation_is_allowed=true)
        #FileGridCurrent(filename='filename.nc', extrapolation_is_allowed=true)

        if len(args) == 0 and len(kwargs) == 0:

            # determine what file format this is
            if filename is None:
                raise TypeError("FileGridCurrent requires a filename")
            filename = str(filename)  # just in case it's a Path object

            if filename.endswith(".nc"):  # should be a netCDF file
                try:
                    GridCurrent.init_from_netCDF(self,
                                                 filename=filename,
                                                 extrapolation_is_allowed=extrapolation_is_allowed,
                                                 **kwargs)
                except Exception as ex:
                    raise ValueError(f"Could not read: {filename}") from ex


            else:  # maybe it's a gridcur file -- that's the only other option
                FileGridCurrent.init_from_gridcur(self,
                                                  filename,
                                                  extrapolation_is_allowed,
                                                  **kwargs)
        else:
            super(FileGridCurrent, self).__init__(
                extrapolation_is_allowed=extrapolation_is_allowed,
                *args,
                **kwargs
            )
            self.filename = filename

    def init_from_gridcur(self, filename, extrapolation_is_allowed, **kwargs):
        '''
        Wrapper for external initalize function
        '''
        try:
            init_from_gridcur(self, filename, extrapolation_is_allowed, **kwargs)
        except GridCurReadError as ex:
            raise ValueError(f"{filename} is not a valid gridcur file") from ex

    @classmethod
    def new_from_dict(cls, serial_dict):
        return cls(filename=serial_dict.get('filename'),
                   extrapolation_is_allowed=serial_dict.get('extrapolation_is_allowed')  # noqa
                   )


