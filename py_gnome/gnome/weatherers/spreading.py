'''
objects used to model the spreading of oil
Include the Langmuir process here as well
'''
import copy

import numpy as np
from repoze.lru import lru_cache
from colander import SchemaNode, Float, drop

from gnome.utilities.serializable import Serializable, Field
from gnome.environment import constant_wind, WindSchema, WaterSchema
from gnome.constants import gravity
from gnome import constants
from .core import Weatherer
from gnome.exceptions import GnomeRuntimeError

from .core import WeathererSchema


class FayGravityViscousSchema(WeathererSchema):
    thickness_limit = SchemaNode(Float(), missing=drop)


class FayGravityViscous(Weatherer, Serializable):
    '''
    Model the FayGravityViscous spreading of the oil. This assumes all LEs
    released together spread as a blob. The blob can be partitioned into 'N'
    LEs and the assumption is that the thickness and initial volume of the
    blob applies to all LEs in it. As such, instead of computing area, lets
    compute thickness - whether 1 or 10 LEs is used to model the blob, the
    thickness remains the same.
    '''
    _state = copy.deepcopy(Weatherer._state)
    _state += Field('water', save=True, update=True, save_reference=True)
    _schema = FayGravityViscousSchema

    # object used to model spreading of oil and area computation
    _ref_as = 'spreading'

    def __init__(self, water=None, **kwargs):
        '''
        initialize object - invoke super, add required data_arrays.
        '''
        super(FayGravityViscous, self).__init__(**kwargs)
        self.spreading_const = (1.53, 1.21)

        # need water temp to get initial viscosity of oil so thickness_limit
        # can be set
        self.water = water
        self.array_types.update({'fay_area', 'area', 'spill_num',
                                 'bulk_init_volume', 'age', 'density'})
        # relative_buoyancy - use density at release time. For now
        # temperature is fixed so just compute once and store. When temperature
        # varies over time, may want to do something different
        self._init_relative_buoyancy = None
        self.thickness_limit = None

    @lru_cache(4)
    def _gravity_spreading_t0(self,
                              water_viscosity,
                              relative_buoyancy,
                              blob_init_vol):
        '''
        time for the initial transient phase of spreading to complete. This
        depends on blob volume, but is on the order of minutes. Cache upto 4
        inputs - don't expect 4 or more spills in one scenario.
        '''
        # time to reach a0
        t0 = ((self.spreading_const[1] / self.spreading_const[0]) ** 4.0 *
              (blob_init_vol / (water_viscosity * constants.gravity *
                                relative_buoyancy)) ** (1. / 3)
              )
        return t0

    def _time_to_reach_max_area(self,
                                water_viscosity,
                                rel_buoy,
                                blob_init_vol):
        '''
        just a convenience function to compute the time to reach max area
        All inputs are scalars
        '''
        max_area = blob_init_vol / self.thickness_limit
        time = (max_area / (np.pi * self.spreading_const[1] ** 2) *
                (np.sqrt(water_viscosity) /
                 (blob_init_vol ** 2 * constants.gravity * rel_buoy)) ** (1./3)
                ) ** 2

        return time

    def init_area(self,
                  water_viscosity,
                  relative_buoyancy,
                  blob_init_vol):
        '''
        This takes scalars inputs since water_viscosity, init_volume and
        relative_buoyancy for a bunch of LEs released together will be the same
        It

        :param water_viscosity: viscosity of water
        :type water_viscosity: float
        :param init_volume: total initial volume of all LEs released together
        :type init_volume: float
        :param relative_buoyancy: relative buoyancy of oil wrt water:
            (rho_water - rho_oil)/rho_water where rho defines density
        :type relative_buoyancy: float

        Equation for gravity spreading:
        ::
            A0 = np.pi*(k2**4/k1**2)*((V0**5*g*dbuoy)/(nu_h2o**2))**(1./6.)
        '''
        a0 = (np.pi *
              (self.spreading_const[1] ** 4 / self.spreading_const[0] ** 2) *
              (((blob_init_vol) ** 5 * constants.gravity * relative_buoyancy) /
               (water_viscosity ** 2)) ** (1. / 6.))

        # highly unlikely to reach max_area, min_thickness during
        # initialization, but it is possible so add a check for it and log
        # an error/warning? for it
        max_area = blob_init_vol / self.thickness_limit
        a0 = min(a0, max_area)
        if a0 == max_area:
            msg = "max_area is achieved during init_area()"
            self.logger.warning(msg)

        return a0

    def _update_blob_area(self, water_viscosity, relative_buoyancy,
                          blob_init_volume, age):
        area = (np.pi *
                self.spreading_const[1] ** 2 *
                (blob_init_volume ** 2 *
                 constants.gravity *
                 relative_buoyancy /
                 np.sqrt(water_viscosity)) ** (1. / 3.) *
                np.sqrt(age))

        return area

    def update_area(self,
                    water_viscosity,
                    relative_buoyancy,
                    blob_init_volume,
                    area,
                    age):
        '''
        update area array in place, also return area array
        each blob is defined by its age. This updates the area of each blob,
        as such, use the mean relative_buoyancy for each blob. Still check
        and ensure relative buoyancy is > 0 for all LEs

        :param water_viscosity: viscosity of water
        :type water_viscosity: float
        :param relative_buoyancy: relative buoyancy of oil wrt water at release
            time. This does not change over time.
        :type relative_buoyancy: float
        :param blob_init_volume: numpy array of floats containing initial
            release volume of blob. This is the same for all LEs released
            together.
        :type blob_init_volume: numpy array
        :param area: numpy array of floats containing area of each LE. Assume
            The LEs with same age belong to the same blob. Sum these up to
            get the area of the blob to compare it to max_area (or min
            thickness). Keep updating blob area till max_area is achieved.
            Equally divide updated_blob_area into the number of LEs used to
            model the blob.
        :type area: numpy array
        :param age: numpy array the same size as area and blob_init_volume.
            This is the age of each LE. The LEs with the same age belong to
            the same blob. Age is in seconds.
        :type age: numpy array of int32
        :param at_max_area: np.bool array. If a blob reaches max_area beyond
            which it will not spread, toggle the LEs associated with that blob
            to True. Max spreading is based on min thickness based on initial
            viscosity of oil. This is used by Langmuir since the process acts
            on particles after spreading completes.
        :type at_max_area: numpy array of bools

        :returns: (updated 'area' array, updated 'at_max_area' array).
            It also changes the input 'area' array and the 'at_max_area' bool
            array inplace. However, the input arrays could be copies so best
            to also return the updates.
        '''
        if np.any(age == 0):
            msg = "use init_area for age == 0"
            raise ValueError(msg)

        # update area for each blob of LEs
        for b_age in np.unique(age):
            # within each age blob_init_volume should also be the same
            m_age = b_age == age
            t0 = self._gravity_spreading_t0(water_viscosity,
                                            relative_buoyancy,
                                            blob_init_volume[m_age][0])

            if b_age <= t0:
                '''
                only update initial area, A_0, if age is past the transient
                phase. Expect this to be the case since t0 is on the order of
                minutes; but do a check incase we want to experiment with
                smaller timesteps.
                '''
                continue

            # now update area of old LEs - only update till max area is reached
            max_area = blob_init_volume[m_age][0] / self.thickness_limit
            if area[m_age].sum() < max_area:
                # update area
                blob_area = self._update_blob_area(water_viscosity,
                                                   relative_buoyancy,
                                                   blob_init_volume[m_age][0],
                                                   age[m_age][0])

                if blob_area >= max_area:
                    area[m_age] = max_area / m_age.sum()
                else:
                    area[m_age] = blob_area / m_age.sum()

                self.logger.debug('{0}\tarea after update: {1}'
                                  .format(self._pid, blob_area))

        return area

    def _get_thickness_limit(self, vo):
        '''
        return the spreading thickness limit based on viscosity
        todo: documented in langmiur docs
            1. vo >= 1e-4;           limit = 1e-4 m
            2. 1e-4 > vo >= 1e-6;    limit = 1e-5 + 0.9091*(vo - 1e-6) m
            3. 1e-6 > vo;            limit = 1e-5 m
        '''
        if vo >= 1e-4:
            thickness_limit = 1e-4
        elif 1e-4 > vo and vo >= 1e-6:
            thickness_limit = 1e-5 + 0.9091 * (vo - 1e-6)
        elif vo < 1e-6:
            thickness_limit = 1e-5

        return thickness_limit

    def _set_thickness_limit(self, vo):
        '''
        sets internal thickness_limit variable
        '''
        self.thickness_limit = self._get_thickness_limit(vo)

    def prepare_for_model_run(self, sc):
        '''
        Assumes only one type of substance is spilled
        '''
        subs = sc.get_substances(False)

        if len(subs) > 0:
            vo = subs[0].get_viscosity(self.water.get('temperature'))
            # set thickness_limit
            self._set_thickness_limit(vo)

        # reset _init_relative_buoyancy for every run
        # make it None so no stale data
        self._init_relative_buoyancy = None

    def _set_init_relative_buoyancy(self, substance):
        '''
        set the initial relative buoyancy of oil wrt water
        use temperature of water to get oil density
        if relative_buoyancy < 0 raises a GnomeRuntimeError - particles will
        sink.
        '''
        rho_h2o = self.water.get('density')
        rho_oil = substance.get_density(self.water.get('temperature'))

        # maybe weathering_data should catch error below?
        # todo: write and raise appropriate exception
        if np.any(rho_h2o < rho_oil):
            msg = ("Found particles with relative_buoyancy < 0. "
                   "Oil is a sinker")
            raise GnomeRuntimeError(msg)

        self._init_relative_buoyancy = (rho_h2o - rho_oil) / rho_h2o

    def initialize_data(self, sc, num_released):
        '''
        initialize  'bulk_init_volume', 'area', 'fay_area' and 'area'
        Currently, carrying both 'fay_area' and 'area', but should drop
        'fay_area' eventually. 'area' gets initialized and updated the same
        as 'fay_area'; however, Langmuir updates 'area'.

        If on is False, then arrays should not be included - dont' initialize
        '''
        if not self.on:
            return

        # do this once incase there are any unit conversions, it only needs to
        # happen once - for efficiency
        water_kvis = self.water.get('kinematic_viscosity',
                                    'square meter per second')

        for substance, data in sc.itersubstancedata(self.array_types):
            if len(data['fay_area']) == 0:
                # no particles released yet
                continue

            if self._init_relative_buoyancy is None:
                self._set_init_relative_buoyancy(substance)

            mask = data['fay_area'] == 0

            for s_num in np.unique(data['spill_num'][mask]):
                s_mask = np.logical_and(mask, data['spill_num'] == s_num)

                # do the sum only once for efficiency
                num = s_mask.sum()

                data['bulk_init_volume'][s_mask] = (data['mass'][s_mask][0] /
                                                    data['density'][s_mask][0]
                                                    ) * num

                init_blob_area = \
                    self.init_area(water_kvis,
                                   self._init_relative_buoyancy,
                                   data['bulk_init_volume'][s_mask][0])

                data['fay_area'][s_mask] = init_blob_area / num
                data['area'][s_mask] = init_blob_area / num

        sc.update_from_fatedataview()

    def weather_elements(self, sc, time_step, model_time):
        '''
        Update 'area', 'fay_area' for previously released particles
        The updated 'area', 'fay_area' is associated with age of particles at:
            model_time + time_step
        '''
        if not self.active:
            return

        water_kvis = self.water.get('kinematic_viscosity',
                                    'square meter per second')
        for _, data in sc.itersubstancedata(self.array_types):
            if len(data['fay_area']) == 0:
                continue

            for s_num in np.unique(data['spill_num']):
                s_mask = data['spill_num'] == s_num
                data['fay_area'][s_mask] = \
                    self.update_area(water_kvis,
                                     self._init_relative_buoyancy,
                                     data['bulk_init_volume'][s_mask],
                                     data['fay_area'][s_mask],
                                     data['age'][s_mask] + time_step)

                data['area'][s_mask] = data['fay_area'][s_mask]

        sc.update_from_fatedataview()

    def serialize(self, json_="webapi"):
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()

        if json_ == 'webapi':
            if self.water is not None:
                schema.add(WaterSchema(name="water"))

        serial = schema.serialize(toserial)

        return serial

    @classmethod
    def deserialize(cls, json_):
        schema = cls._schema(name=cls.__name__)
        if 'water' in json_:
            schema.add(WaterSchema(name="water"))

        _to_dict = schema.deserialize(json_)

        return _to_dict


class ConstantArea(Weatherer, Serializable):
    '''
    Used for testing and diagnostics
    - must be manually hooked up
    '''
    _ref_as = 'spreading'

    def __init__(self, area, **kwargs):
        self.area = area
        super(ConstantArea, self).__init__(**kwargs)

        self.array_types.update({'fay_area'})

    def initialize_data(self, sc):
        '''
        If on is False, then arrays should not be included - dont' initialize
        '''
        if not self.on:
            return

        for _, data in sc.itersubstancedata(self.array_types):
            if len(data['fay_area']) == 0:
                continue

            mask = data['fay_area'] == 0
            data['fay_area'][mask] = self.area
            data['area'][mask] = self.area

        sc.update_from_fatedataview()

    def weather_elements(self, sc, time_step, model_time):
        '''
        return the area array as it was entered since that contains area per
        LE if there is more than one LE. Kept the interface the same as
        FayGravityViscous since WeatheringData will call it the same way.
        '''
        for _, data in sc.itersubstancedata(self.array_types):
            if len(data['fay_area']) == 0:
                continue

            data['fay_area'] = self.area

        sc.update_from_fatedataview()


class Langmuir(Weatherer, Serializable):
    '''
    Easiest to define this as a weathering process that updates 'area' array
    '''
    _schema = WeathererSchema

    _state = copy.deepcopy(Weatherer._state)
    _state += [Field('wind', update=True, save=True, save_reference=True),
               Field('water', update=True, save=True, save_reference=True)]

    def __init__(self,
                 water=None,
                 wind=None,
                 **kwargs):
        '''
        initialize wind to (0, 0) if it is None
        '''
        super(Langmuir, self).__init__(**kwargs)
        self.array_types.update(('area', 'frac_coverage'))

        if wind is None:
            self.wind = constant_wind(0, 0)
        else:
            self.wind = wind

        # need water object to find relative buoyancy
        self.water = water

    def _get_frac_coverage(self, model_time, rel_buoy, thickness):
        '''
        return fractional coverage for a blob of oil with inputs;
        relative_buoyancy, and thickness

        Assumes the thickness is the minimum oil thickness associated with
        max area achievable by Fay Spreading

        Frac coverage bounds are constants. If computed frac_coverge is outside
        the bounds of (0.1, or 1.0), then limit it to:
            0.1 <= frac_cov <= 1.0
        '''
        v_max = self.wind.get_value(model_time)[0] * 0.005
        cr_k = (v_max ** 2 *
                4 *
                np.pi ** 2 /
                (thickness * rel_buoy * gravity)) ** (1. / 3.)
        frac_cov = 1. / cr_k

        frac_cov[frac_cov < 0.1] = 0.1
        frac_cov[frac_cov > 1.0] = 1.0

        return frac_cov

    def _wind_speed_bound(self, rel_buoy, thickness):
        '''
        return min/max wind speed for given rel_buoy, thickness such that
        Langmuir effect is within bounds:
            0.1 <= frac_coverage <= 1.0
        '''
        v_min = np.sqrt(1.0 * thickness * rel_buoy * gravity /
                        (4 * np.pi ** 2)) / 0.005
        v_max = np.sqrt((1. / 0.1) ** 3 * thickness * rel_buoy * gravity /
                        (4 * np.pi ** 2)) / 0.005

        return (v_min, v_max)

    def weather_elements(self, sc, time_step, model_time):
        '''
        set the 'area' array based on the Langmuir process
        This only applies to particles marked for weathering on the surface:
        ie fate_status is surface_weather
        '''
        if not self.active or sc.num_released == 0:
            return

        rho_h2o = self.water.get('density', 'kg/m^3')
        for _, data in sc.itersubstancedata(self.array_types):
            for s_num in np.unique(data['spill_num']):
                s_mask = data['spill_num'] == s_num
                # thickness for blob of oil released together - need per spill
                # Use the 'bulk_init_volume' and the 'fay_area' of the
                # blob of oil. Each LE used to model the blob will have the
                # same thickness. In order to get the 'fay_area' for the blob
                # of oil released at same time, from same spill, sum
                # the 'fay_area' array for elements that belong to same oil
                # blob.
                thickness = (data['bulk_init_volume'][s_mask][0] /
                             data['fay_area'][s_mask].sum())

                # assume only one type of oil is modeled so thickness_limit is
                # already set and constant for all
                rel_buoy = (rho_h2o - data['density'][s_mask]) / rho_h2o
                data['frac_coverage'][s_mask] = \
                    self._get_frac_coverage(model_time, rel_buoy, thickness)

            # update 'area'
            data['area'][:] = data['fay_area'] * data['frac_coverage']

        sc.update_from_fatedataview()

    def serialize(self, json_='webapi'):
        """
        Since 'wind' property is saved as a reference when used in save file
        and 'save' option, need to add appropriate node to WindMover schema
        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema(name=self.__class__.__name__)
        if json_ == 'webapi':
            # add wind schema
            schema.add(WindSchema(name='wind'))

            if self.water is not None:
                schema.add(WaterSchema(name='water'))

        serial = schema.serialize(toserial)

        return serial

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for wind object
        """
        schema = cls._schema(name=cls.__name__)
        if 'wind' in json_:
            schema.add(WindSchema(name='wind'))

        if 'water' in json_:
            schema.add(WaterSchema(name='water'))

        _to_dict = schema.deserialize(json_)

        return _to_dict
