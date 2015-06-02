'''
objects used to model the spreading of oil
Include the Langmuir process here as well
'''
import copy

import numpy as np
from repoze.lru import lru_cache

from gnome.utilities.serializable import Serializable, Field
from gnome.environment import constant_wind, WindSchema, WaterSchema
from gnome.constants import gravity
from gnome import AddLogger, constants
from .core import Weatherer

from gnome.persist.base_schema import ObjType


class SpreadingThicknessLimit(object):
    '''
    a mixin containing only one method which sets the thickness_limit based
    on viscosity. FayGravityViscous and Langmuir both need access to
    thickness_limit so make it a mixin so each derived class will get this
    attribute and the attribute is set the same way for both.
    '''
    thickness_limit = None

    def set_thickness_limit(self, vo):
        '''
        set the spreading thickness limit based on viscosity
        todo: documented in langmiur docs
            1. vo >= 1e-4;           limit = 1e-4 m
            2. 1e-4 > vo >= 1e-6;    limit = 1e-5 + 0.9091*(vo - 1e-6) m
            3. 1e-6 > vo;            limit = 1e-5 m
        '''
        if vo >= 1e-4:
            self.thickness_limit = 1e-4
        elif 1e-4 > vo and vo >= 1e-6:
            self.thickness_limit = 1e-5 + 0.9091 * (vo - 1e-6)
        elif vo < 1e-6:
            self.thickness_limit = 1e-5


class FayGravityViscous(SpreadingThicknessLimit, AddLogger):
    '''
    Model the FayGravityViscous spreading of the oil. This assumes all LEs
    released together spread as a blob. The blob can be partitioned into 'N'
    LEs and the assumption is that the thickness and initial volume of the
    blob applies to all LEs in it. As such, instead of computing area, lets
    compute thickness - whether 1 or 10 LEs is used to model the blob, the
    thickness remains the same.
    '''
    def __init__(self):
        self.spreading_const = (1.53, 1.21)

    @lru_cache(4)
    def _gravity_spreading_t0(self,
                              water_viscosity,
                              relative_bouyancy,
                              blob_init_vol):
        '''
        time for the initial transient phase of spreading to complete. This
        depends on blob volume, but is on the order of minutes. Cache upto 4
        inputs - don't expect 4 or more spills in one scenario.
        '''
        # time to reach a0
        t0 = ((self.spreading_const[1]/self.spreading_const[0]) ** 4.0 *
              (blob_init_vol/(water_viscosity * constants.gravity *
                              relative_bouyancy))**(1./3))
        return t0

    def _time_to_reach_max_area(self,
                                water_viscosity,
                                rel_buoy,
                                blob_init_vol):
        '''
        just a convenience function to compute the time to reach max area
        All inputs are scalars
        '''
        max_area = blob_init_vol/self.thickness_limit
        time = (max_area/(np.pi * self.spreading_const[1] ** 2) *
                (np.sqrt(water_viscosity) /
                 (blob_init_vol**2 * constants.gravity * rel_buoy))**(1./3)
                )**2
        return time

    def init_area(self,
                  water_viscosity,
                  relative_bouyancy,
                  blob_init_vol):
        '''
        This takes scalars inputs since water_viscosity, init_volume and
        relative_bouyancy for a bunch of LEs released together will be the same
        It

        :param water_viscosity: viscosity of water
        :type water_viscosity: float
        :param init_volume: total initial volume of all LEs released together
        :type init_volume: float
        :param relative_bouyancy: relative bouyance of oil wrt water:
            (rho_water - rho_oil)/rho_water where rho defines density
        :type relative_bouyancy: float

        Equation for gravity spreading:
        ::
            A0 = np.pi*(k2**4/k1**2)*((V0**5*g*dbuoy)/(nu_h2o**2))**(1./6.)
        '''
        self._check_relative_bouyancy(relative_bouyancy)
        a0 = (np.pi*(self.spreading_const[1]**4/self.spreading_const[0]**2)
              * (((blob_init_vol)**5*constants.gravity*relative_bouyancy) /
                 (water_viscosity**2))**(1./6.))

        # highly unlikely to reach max_area, min_thickness during
        # initialization, but it is possible so add a check for it and log
        # an error/warning? for it
        max_area = blob_init_vol/self.thickness_limit
        a0 = min(a0, max_area)
        if a0 == max_area:
            msg = "max_area is achieved during init_area() ..blah! fix"
            self.logger.error(msg)

        return a0

    def _check_relative_bouyancy(self, rel_bouy):
        '''
        For now just raise an error if any relative_bouyancy is < 0. These
        particles will sink, ask how we want to deal with them. They should
        be removed or we should only look at floating particles when computing
        area?
        '''
        if np.any(rel_bouy < 0):
            msg = ("Found particles with relative_bouyancy < 0. "
                   "Oil is a sinker")
            self.logger.error(msg)

    def _update_blob_area(self, water_viscosity, relative_bouyancy,
                          blob_init_volume, age):
        area = (np.pi * self.spreading_const[1]**2 *
                (blob_init_volume**2 * constants.gravity * relative_bouyancy /
                 np.sqrt(water_viscosity)) ** (1./3) * np.sqrt(age))

        return area

    def update_area(self,
                    water_viscosity,
                    relative_bouyancy,
                    blob_init_volume,
                    area,
                    age,
                    at_max_area):
        '''
        update area array in place, also return area array
        each blob is defined by its age. This updates the area of each blob,
        as such, use the mean relative_bouyancy for each blob. Still check
        and ensure relative bouyancy is > 0 for all LEs

        :param water_viscosity: viscosity of water
        :type water_viscosity: float
        :param relative_bouyancy: relative bouyancy of oil wrt water at release
            time. This does not change over time.
        :type relative_bouyancy: float
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

        self._check_relative_bouyancy(relative_bouyancy)

        # update area for each blob of LEs
        for b_age in np.unique(age):
            # within each age blob_init_volume should also be the same
            m_age = b_age == age
            t0 = self._gravity_spreading_t0(water_viscosity,
                                            relative_bouyancy,
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
            max_area = blob_init_volume[m_age][0]/self.thickness_limit
            if area[m_age].sum() < max_area:
                # update area
                blob_area = \
                    self._update_blob_area(water_viscosity,
                                           relative_bouyancy,
                                           blob_init_volume[m_age][0],
                                           age[m_age][0])
                if blob_area >= max_area:
                    area[m_age] = max_area/m_age.sum()
                    at_max_area[m_age] = True
                else:
                    area[m_age] = blob_area/m_age.sum()

                self.logger.debug(self._pid +
                                  "\tarea after update: {0}".format(blob_area))
            else:
                # area is at max_area - ensure at_max_area is set correctly
                # at_max_area should always be correctly set
                if not np.any(at_max_area[m_age]):
                    at_max_area[m_age] = True

        return (area, at_max_area)


class ConstantArea(SpreadingThicknessLimit, AddLogger):
    '''
    Used for testing and diagnostics
    - must be manually hooked up
    '''
    def __init__(self, area):
        self.area = area

    def init_area(self, *args):
        return self.area

    def update_area(self,
                    water_viscosity=None,
                    relative_bouyancy=None,
                    blob_init_volume=None,
                    area=None,
                    age=None,
                    at_max_area=None):
        '''
        return the area array as it was entered since that contains area per
        LE if there is more than one LE. Kept the interface the same as
        FayGravityViscous since WeatheringData will call it the same way.
        '''
        if at_max_area is None:
            at_max_area = np.asarray(blob_init_volume, np.uint8)
        at_max_area[:] = True

        if area is None:
            return (self.area, at_max_area)
        else:
            return (area, at_max_area)

    def set_thickness_limit(self, vo):
        '''
        just use constant area so not setting any thickness limit
        '''
        pass


class Langmuir(Weatherer, Serializable, SpreadingThicknessLimit):
    '''
    Easiest to define this as a weathering process that updates 'area' array
    '''
    _schema = ObjType

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
        self.array_types.update(('area', 'at_max_area'))

        if wind is None:
            self.wind = constant_wind(0, 0)
        else:
            self.wind = wind

        # need water object to find relative buoyancy
        self.water = water

    def prepare_for_model_run(self, sc):
        '''
        set thickness limit based on viscosity at release time
        '''
        subs = sc.get_substances(False)

        # initialize the thickness_limit for FayGravityViscous based on
        # viscosity of oil - assume only one type of substance for all spills
        # make sure we have spills with valid substance
        if len(subs) > 0:
            vo = subs[0].get_viscosity(self.water.get('temperature'))
            self.set_thickness_limit(vo)

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
        cr_k = \
            (v_max**2 * 4 * np.pi**2/(thickness * rel_buoy * gravity))**(1./3)
        frac_cov = 1./cr_k

        # if rel_buoy is an array, then frac_cov will be an array
        if not isinstance(frac_cov, np.ndarray):
            frac_cov = np.asarray([frac_cov], np.float64)

        frac_cov[frac_cov < 0.1] = 0.1
        frac_cov[frac_cov > 1.0] = 1.0

        if isinstance(cr_k, np.ndarray):
            return frac_cov
        else:
            # must be a scalar
            return frac_cov[0]

    def _wind_speed_bound(self, rel_buoy, thickness):
        '''
        return min/max wind speed for given rel_buoy, thickness such that
        Langmuir effect is within bounds:
            0.1 <= frac_coverage <= 1.0
        '''
        v_min = np.sqrt(1.0 * thickness * rel_buoy * gravity /
                        (4 * np.pi**2))/0.005
        v_max = np.sqrt((1./0.1)**3 * thickness * rel_buoy * gravity /
                        (4 * np.pi**2))/0.005
        return (v_min, v_max)

    def weather_elements(self, sc, time_step, model_time):
        '''
        set the 'area' array based on the Langmuir process
        '''
        if not self.active or sc.num_released == 0:
            return

        rho_h2o = self.water.get('density', 'kg/m^3')
        for substance, data in sc.itersubstancedata(self.array_types,
                                                    fate='all'):
            mask = data['at_max_area'] == True

            if np.any(mask):
                # assume only one type of oil is modeled so thickness_limit is
                # already set and constant for all
                rel_buoy = (rho_h2o - data['density'][mask])/rho_h2o
                data['area'][mask] *= self._get_frac_coverage(model_time,
                                                              rel_buoy,
                                                              self.thickness_limit)
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
