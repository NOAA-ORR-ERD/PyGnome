'''
objects used to model the spreading of oil
Include the Langmuir process here as well
'''

import numpy as np
import os
from functools import lru_cache

from colander import SchemaNode, Float, drop

from gnome.array_types import gat

from gnome.environment import WindSchema, WaterSchema
from gnome.constants import gravity
from gnome.constants import water_kinematic_viscosity as water_kvis
from gnome import constants
from .core import Weatherer
from gnome.exceptions import GnomeRuntimeError

from .core import WeathererSchema
from gnome.persist.base_schema import GeneralGnomeObjectSchema
from gnome.environment.gridded_objects_base import VectorVariableSchema
from gnome.ops import default_constants

PI = np.pi
PISQUARED = np.pi ** 2

class FayGravityViscousSchema(WeathererSchema):
    thickness_limit = SchemaNode(Float(), missing=drop, save=True, update=True)
    water = WaterSchema(save=True, update=True)


class FayGravityViscous(Weatherer):
    '''
    Model the FayGravityViscous spreading of the oil. For instantaneous release,
    this assumes all LEs released together spread as a blob following Fay (1971).
    The blob can be partitioned into 'N' LEs and the assumption is that the thickness
    and initial volume of the blob applies to all LEs in it. For continuous release,
    the spreading algorithm is similar to Dodge et al., (1983), where blob volume is
    considered as the cumulative volume of oil varying with time during the release period
    '''
    _schema = FayGravityViscousSchema

    # object used to model spreading of oil and area computation
    _ref_as = 'spreading'
    _req_refs = ['water']

    def __init__(self, water=None, thickness_limit=None, **kwargs):
        '''
        initialize object - invoke super, add required data_arrays.
        '''
        super(FayGravityViscous, self).__init__(**kwargs)
        self.spreading_const = (1.53, 1.21, 1.45)

        # need water temp to get initial viscosity of oil so thickness_limit
        # can be set
        # fixme: can use nominal viscosity!
        self.water = water
        self.array_types.update({'fay_area': gat('fay_area'),
                                 'area': gat('area'),
                                 'bulk_init_volume': gat('bulk_init_volume'),
                                 'age': gat('age'),
                                 'density': gat('density'),
                                 'frac_coverage': gat('frac_coverage'),
                                 'spill_num': gat('spill_num'),
                                 'max_area_le': gat('max_area_le'),
                                 'release_rate': gat('release_rate'),
                                 'vol_frac_le_st': gat('vol_frac_le_st')})
        # relative_buoyancy - use density at release time. For now
        # temperature is fixed so just compute once and store. When temperature
        # varies over time, may want to do something different
        self._init_relative_buoyancy = None
        self.thickness_limit = thickness_limit
        # self.is_first_step = True
        self.use_langmuir_correction = True
    @staticmethod
    @lru_cache(10)
    def _gravity_spreading_t0(water_viscosity,
                              relative_buoyancy,
                              blob_init_vol,
                              spreading_const):
        '''
        time for the initial transient phase of spreading to complete. This
        depends on blob volume, but is on the order of minutes. Cache up to 10
        inputs - don't expect 10 or more spills in one scenario.
        '''
        # time to reach a0
        t0 = ((spreading_const[1] / spreading_const[0]) ** 4.0 *
              (blob_init_vol / (water_viscosity * constants.gravity *
                                relative_buoyancy)) ** (1.0 / 3.0)
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
# correct k_nu, Spreading Law coefficient -- Eq.(6.14), 11/23/2021
#       time = (max_area / (PI * self.spreading_const[1] ** 2) *
        time = (max_area / (PI * self.spreading_const[2] ** 2) *
                (np.sqrt(water_viscosity) /
                 (blob_init_vol ** 2 * constants.gravity * rel_buoy)) ** (1. / 3)
                ) ** 2

        return time

    def init_area(self,
                  water_viscosity,
                  relative_buoyancy,
                  blob_init_vol):
        '''
        This takes scalars inputs since water_viscosity, init_volume and
        relative_buoyancy for a bunch of LEs released together will be the same


        :param water_viscosity: viscosity of water
        :type water_viscosity: float
        :param blob_init_volume: total initial volume of all LEs released together
        :type blob_init_volume: float
        :param relative_buoyancy: relative buoyancy of oil wrt water:
            (rho_water - rho_oil)/rho_water where rho is the density
        :type relative_buoyancy: float

        Equation for gravity spreading:

        ::

            A0 = PI*(k2**4/k1**2)*((V0**5*g*dbuoy)/(nu_h2o**2))**(1./6.)
        '''
# import release duration here to control init_oil_volume for continue release 12-06-2021
        a0 = (PI *
             (self.spreading_const[1] ** 4 / self.spreading_const[0] ** 2) *
             (((blob_init_vol) ** 5 * gravity * relative_buoyancy) /
             (water_viscosity ** 2)) ** (1. / 6.))
# import release duration here to control init_oil_volume for continue release 12-06-2021
        # highly unlikely to reach max_area, min_thickness during
        # initialization, but it is possible so add a check for it and log
        # an error/warning? for it
        max_area = blob_init_vol / self.thickness_limit
        a0 = min(a0, max_area)
        if a0 == max_area:
            msg = "max_area is achieved during init_area()"
            self.logger.warning(msg)

        return a0

    # def _update_blob_area(self, water_viscosity, relative_buoyancy,
                          # blob_init_volume, age):
        # area = (PI *
                # self.spreading_const[2] ** 2 *
                # (blob_init_volume ** 2 *
                 # constants.gravity *
                 # relative_buoyancy /
                 # np.sqrt(water_viscosity)) ** (1. / 3.) *
                # np.sqrt(age))

        # return area

    # def update_area(self,
    #                 water_viscosity,
    #                 relative_buoyancy,
    #                 blob_init_volume,
    #                 area,
    #                 age):
    #     '''
    #     update area array in place, also return area array
    #     each blob is defined by its age. This updates the area of each blob,
    #     as such, use the mean relative_buoyancy for each blob. Still check
    #     and ensure relative buoyancy is > 0 for all LEs

    #     :param water_viscosity: viscosity of water
    #     :type water_viscosity: float
    #     :param relative_buoyancy: relative buoyancy of oil wrt water at release
    #         time. This does not change over time.
    #     :type relative_buoyancy: float
    #     :param blob_init_volume: numpy array of floats containing initial
    #         release volume of blob. This is the same for all LEs released
    #         together.
    #     :type blob_init_volume: numpy array
    #     :param area: numpy array of floats containing area of each LE. Assume
    #         The LEs with same age belong to the same blob. Sum these up to
    #         get the area of the blob to compare it to max_area (or min
    #         thickness). Keep updating blob area till max_area is achieved.
    #         Equally divide updated_blob_area into the number of LEs used to
    #         model the blob.
    #     :type area: numpy array
    #     :param age: numpy array the same size as area and blob_init_volume.
    #         This is the age of each LE. The LEs with the same age belong to
    #         the same blob. Age is in seconds.
    #     :type age: numpy array of int32
    #     :param at_max_area: bool array. If a blob reaches max_area beyond
    #         which it will not spread, toggle the LEs associated with that blob
    #         to True. Max spreading is based on min thickness based on initial
    #         viscosity of oil. This is used by Langmuir since the process acts
    #         on particles after spreading completes.
    #     :type at_max_area: numpy array of bools

    #     :returns: (updated 'area' array, updated 'at_max_area' array).
    #         It also changes the input 'area' array and the 'at_max_area' bool
    #         array inplace. However, the input arrays could be copies so best
    #         to also return the updates.
    #     '''
    #     if np.any(age == 0):
    #         msg = "use init_area for age == 0"
    #         raise ValueError(msg)

    #     # update area for each blob of LEs
    #     for b_age in np.unique(age):
    #         # within each age blob_init_volume should also be the same
    #         m_age = b_age == age
    #         t0 = self._gravity_spreading_t0(water_viscosity,
    #                                         relative_buoyancy,
    #                                         blob_init_volume[m_age][0],
    #                                         self.spreading_const)

    #         if b_age <= t0:
    #             '''
    #             only update initial area, A_0, if age is past the transient
    #             phase. Expect this to be the case since t0 is on the order of
    #             minutes; but do a check in case we want to experiment with
    #             smaller timesteps.
    #             '''
    #             continue

    #         # now update area of old LEs - only update till max area is reached
    #         max_area = blob_init_volume[m_age][0] / self.thickness_limit
    #         if area[m_age].sum() < max_area:
    #             # update area
    #             blob_area = self._update_blob_area(water_viscosity,
    #                                                relative_buoyancy,
    #                                                blob_init_volume[m_age][0],
    #                                                age[m_age][0])

    #             if blob_area >= max_area:
    #                 area[m_age] = max_area / m_age.sum()
    #             else:
    #                 area[m_age] = blob_area / m_age.sum()

    #             self.logger.debug('{0}\tarea after update: {1}'
    #                               .format(self._pid, blob_area))

    #     return area

#     def update_area2(self,
#                      water_viscosity,
#                      relative_buoyancy,
#                      blob_init_volume,
#                      area,
#                      time_step,
#                      age):
#         '''
#         update area array in place, also return area array
#         each blob is defined by its age. This updates the area of each blob,
#         as such, use the mean relative_buoyancy for each blob. Still check
#         and ensure relative buoyancy is > 0 for all LEs

#         :param water_viscosity: viscosity of water
#         :type water_viscosity: float
#         :param relative_buoyancy: relative buoyancy of oil wrt water at release
#             time. This does not change over time.
#         :type relative_buoyancy: float
#         :param blob_init_volume: numpy array of floats containing initial
#             release volume of blob. This is the same for all LEs released
#             together.
#         :type blob_init_volume: numpy array
#         :param area: numpy array of floats containing area of each LE. Assume
#             The LEs with same age belong to the same blob. Sum these up to
#             get the area of the blob to compare it to max_area (or min
#             thickness). Keep updating blob area till max_area is achieved.
#             Equally divide updated_blob_area into the number of LEs used to
#             model the blob.
#         :type area: numpy array
#         :param age: numpy array the same size as area and blob_init_volume.
#             This is the age of each LE. The LEs with the same age belong to
#             the same blob. Age is in seconds.
#         :type age: numpy array of int32
#         :param at_max_area: bool array. If a blob reaches max_area beyond
#             which it will not spread, toggle the LEs associated with that blob
#             to True. Max spreading is based on min thickness based on initial
#             viscosity of oil. This is used by Langmuir since the process acts
#             on particles after spreading completes.
#         :type at_max_area: numpy array of bools

#         :returns: (updated 'area' array, updated 'at_max_area' array).
#             It also changes the input 'area' array and the 'at_max_area' bool
#             array inplace. However, the input arrays could be copies so best
#             to also return the updates.
#         '''
#         if np.any(age == 0):
#             msg = "use init_area for age == 0"
#             raise ValueError(msg)

#         # update area for each blob of LEs
#         for b_age in np.unique(age):
#             # within each age blob_init_volume should also be the same
#             m_age = b_age == age

#             t0 = self._gravity_spreading_t0(water_viscosity,
#                                             relative_buoyancy,
#                                             blob_init_volume[m_age][0],
#                                             self.spreading_const)

#             if b_age <= t0:
#                 '''
#                 only update initial area, A_0, if age is past the transient
#                 phase. Expect this to be the case since t0 is on the order of
#                 minutes; but do a check in case we want to experiment with
#                 smaller timesteps.
#                 '''
#                 continue

#             # now update area of old LEs - only update till max area is reached
#             max_area = blob_init_volume[m_age][0] / self.thickness_limit
#             if area[m_age].sum() < max_area:

#                 C = (PI *
# # correct k_nu, Spreading Law coefficient -- Eq.(6.14), 11/23/2021
# #                    self.spreading_const[1] ** 2 *
#                      self.spreading_const[2] ** 2 *
#                     (blob_init_volume[m_age][0] ** 2 *
#                     constants.gravity *
#                     relative_buoyancy /
#                     np.sqrt(water_viscosity)) ** (1. / 3.))

#                 #blob_area_fgv = area[m_age].sum() + .5 * (C**2 / area[m_age].sum()) * time_step	# make sure area > 0
#                 blob_area_fgv = .5 * (C**2 / area[m_age].sum()) * time_step	# make sure area > 0

#                 K = 4 * PI * 2 * .033

#                 #blob_area_diffusion = area[m_age].sum() + ((7 / 6) * K * (area[m_age].sum() / K) ** (1 / 7)) * time_step
#                 blob_area_diffusion = ((7. / 6.) * K * (area[m_age].sum() / K) ** (1. / 7.)) * time_step

#                 #blob_area = blob_area_fgv + blob_area_diffusion
#                 blob_area = area[m_age].sum() + blob_area_fgv + blob_area_diffusion

#                 if blob_area >= max_area:
#                     area[m_age] = max_area / m_age.sum()
#                 else:
#                     area[m_age] = blob_area / m_age.sum()

#                 self.logger.debug('{0}\tarea after update: {1}'
#                                   .format(self._pid, blob_area))

#         return area

#     def update_area33(self,
#                       water_viscosity,
#                       relative_buoyancy,
#                       blob_init_vol,
#                       area,
#                       max_area_le,
#                       time_step,
#                       vol_frac_le_st,
#                       age):
#         '''
#         update area array in place, also return area array
#         each blob is defined by its age. This updates the area of each blob,
#         as such, use the mean relative_buoyancy for each blob. Still check
#         and ensure relative buoyancy is > 0 for all LEs

#         :param water_viscosity: viscosity of water
#         :type water_viscosity: float
#         :param relative_buoyancy: relative buoyancy of oil wrt water at release
#             time. This does not change over time.
#         :type relative_buoyancy: float
#         :param blob_init_volume: numpy array of floats containing initial
#             release volume of blob. This is the same for all LEs released
#             together.
#         :type blob_init_volume: numpy array
#         :param area: numpy array of floats containing area of each LE. Assume
#             The LEs with same age belong to the same blob. Sum these up to
#             get the area of the blob to compare it to max_area (or min
#             thickness). Keep updating blob area till max_area is achieved.
#             Equally divide updated_blob_area into the number of LEs used to
#             model the blob.
#         :type area: numpy array
#         :param age: numpy array the same size as area and blob_init_volume.
#             This is the age of each LE. The LEs with the same age belong to
#             the same blob. Age is in seconds.
#         :type age: numpy array of int32
#         :param at_max_area: bool array. If a blob reaches max_area beyond
#             which it will not spread, toggle the LEs associated with that blob
#             to True. Max spreading is based on min thickness based on initial
#             viscosity of oil. This is used by Langmuir since the process acts
#             on particles after spreading completes.
#         :type at_max_area: numpy array of bools

#         :returns: (updated 'area' array, updated 'at_max_area' array).
#             It also changes the input 'area' array and the 'at_max_area' bool
#             array inplace. However, the input arrays could be copies so best
#             to also return the updates.
#         '''

#         if np.any(age == 0):
#             msg = "use init_area for age == 0"
#             raise ValueError(msg)

# # update area for each blob of LEs
# #       for b_age in np.unique(age):
# # numpy arrays -- faster
#         for index, le_area in np.ndenumerate(area):
#             # within each age blob_init_volume should also be the same
# #           m_age = b_age == age

#             t0 = self._gravity_spreading_t0(water_viscosity,
#                                            relative_buoyancy,
# #                                          blob_init_volume[m_age][0],
#                                            blob_init_vol,
#                                            self.spreading_const)

# #            if b_age <= t0:
#             if age[index] <= t0:
#                 '''
#                 only update initial area, A_0, if age is past the transient
#                 phase. Expect this to be the case since t0 is on the order of
#                 minutes; but do a check in case we want to experiment with
#                 smaller timesteps.
#                 '''
#                 continue

#             # now update area of old LEs - only update till max area is reached
#             if le_area < max_area_le[index]:
#                 C = (PI *
# # correct k_nu, Spreading Law coefficient -- Eq.(6.14), 11/23/2021
# #                    self.spreading_const[1] ** 2 *
#                      self.spreading_const[2] ** 2 *
# #                   (blob_init_volume[m_age][0] ** 2 *
#                     (blob_init_vol ** 2 *
#                     constants.gravity *
#                     relative_buoyancy /
#                     np.sqrt(water_viscosity)) ** (1. / 3.))

#                 K = 4 * PI * 2 * .033

#                 blob_area_fgv = .5 * (C**2 / (le_area / vol_frac_le_st)) * time_step	# make sure area > 0

#                 blob_area_diffusion = ((7. / 6.) * K * ((le_area / vol_frac_le_st) / K) ** (1. / 7.)) * time_step
#                 new_le_area = le_area + vol_frac_le_st * (blob_area_fgv + blob_area_diffusion)


#                 if new_le_area >= max_area_le[index]:
#                     area[index] = max_area_le[index]
#                 else:
#                     area[index] = new_le_area
#                 self.logger.debug('{0}\tarea after update: {1}'
#                                   .format(self._pid, new_le_area))

#         return area

    def update_area(self,
                     water_viscosity,
                     relative_buoyancy,
                     blob_init_vol,
                     area,
                     max_area_le,
                     time_step,
                     vol_frac_le_st,
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
            together. Note that for continuous release, the blob_init_vol
            will be updated as cumulative volume within the release duration.
        :type blob_init_volume: numpy array
        :param area: numpy array of floats containing area of each LE. Assume
            The LEs with same age belong to the same blob. Sum these up to
            get the area of the blob to compare it to max_area (or min
            thickness). Keep updating blob area till max_area is achieved.
            Equally divide updated_blob_area into the number of LEs used to
            model the blob.
        :type area: numpy array
        :param max_area_le: bool array. If a LE area reaches max_area_le beyond
            which it will not spread, toggle the LEs associated with that LE
            to True. Max spreading is based on min thickness based on initial
            viscosity of oil. This is used by Langmuir since the process acts
            on particles after spreading completes.
        :type max_area_le: numpy array of bools
        :param time_step: time step of simulation, which is in seconds.
        :type max_area_le: float
        :param vol_frac_le_st: numpy array the same size as area.
            This is the volume fraction of each LE. It is used to convert the
            computation into element-based.
        :type vol_frac_le_st: numpy array of int32
        :param age: numpy array the same size as area and blob_init_volume.
            This is the age of each LE. The LEs with the same age belong to
            the same blob. Age is in seconds.
        :type age: numpy array of int32

        :returns: (updated 'area' array, updated 'at_max_area' array).
            It also changes the input 'area' array and the 'at_max_area' bool
            array inplace. However, the input arrays could be copies so best
            to also return the updates.
        '''

        if np.any(age == 0):
            msg = "use init_area for age == 0"
            raise ValueError(msg)

        t0 = self._gravity_spreading_t0(water_viscosity,
                                           relative_buoyancy,
                                           blob_init_vol[0],
                                           self.spreading_const)
        # once the area computed from previous ts is larger than the max_area_le at current ts area needs to remain
        #mask = np.logical_and(age > t0, area * (1.0 + 1.e-14) < max_area_le)
        mask = np.logical_and(age > t0, area < max_area_le)
        s_mask = np.logical_and(mask, area > 0.0)
        if len(area[s_mask]) > 0:

            C = (PI *
                 self.spreading_const[2] ** 2 *
                 (blob_init_vol[s_mask] ** 2 *
                 constants.gravity *
                 relative_buoyancy /
                 np.sqrt(water_viscosity)) ** (1. / 3.))

            K = 4 * PI * 2 * .033

            blob_area_fgv = .5 * (C**2 / (area[s_mask] / vol_frac_le_st[s_mask])) * time_step	# make sure area > 0

            blob_area_diffusion = ((7. / 6.) * K * ((area[s_mask] / vol_frac_le_st[s_mask]) / K) ** (1. / 7.)) * time_step

            new_le_area = area[s_mask] + vol_frac_le_st[s_mask] * (blob_area_fgv + blob_area_diffusion)

            area[s_mask] = np.minimum(new_le_area, max_area_le[s_mask])

        return area

    @staticmethod
    def get_thickness_limit(vo):
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
        self.thickness_limit = self.get_thickness_limit(vo)

    def prepare_for_model_run(self, sc):
        '''
        Assumes only one type of substance is spilled

        That's now TRUE!
        '''
        subs = sc.get_substances(False)
        if len(subs) > 0 and subs[0].is_weatherable:
           if self.water is None:
              vo = subs[0].kvis_at_temp(default_constants.default_water_temperature)
           else:
              vo = subs[0].kvis_at_temp(self.water.get('temperature'))
            # set thickness_limit
           self._set_thickness_limit(vo)

        # reset _init_relative_buoyancy for every run
        # make it None so no stale data
        self._init_relative_buoyancy = None

        #self.is_first_step = True

    def _set_init_relative_buoyancy(self, substance):
        '''
        set the initial relative buoyancy of oil wrt water
        use temperature of water to get oil density
        if relative_buoyancy < 0 raises a GnomeRuntimeError - particles will
        sink.
        '''
        if self.water is None:
           rho_h2o = default_constants.default_water_density
           rho_oil = substance.standard_density #density_at_temp(default_constants.default_water_temperature)
        else:
           rho_h2o = self.water.get('density')
           rho_oil = substance.standard_density #density_at_temp(self.water.get('temperature'))

        # maybe weathering_data should catch error below?
        # todo: write and raise appropriate exception
        if np.any(rho_h2o < rho_oil):
            msg = ("Found particles with relative_buoyancy < 0. "
                   "Oil is a sinker")
            raise GnomeRuntimeError(msg)
# add a check for zero relative buoyancy, if it's the case, assign the smallest value allowed
        self._init_relative_buoyancy = (rho_h2o - rho_oil) / rho_h2o

    def initialize_data(self, sc, num_released):
        '''
        initialize  'relative_buoyancy'. Note that initialization of spreading area for LEs
        is done in release object.

        If on is False, then arrays should not be included - dont' initialize
        '''

        # do this once in case there are any unit conversions, it only needs to
        # happen once - for efficiency


        if not self.on or not sc.substance.is_weatherable:
            return

        for substance, data in sc.itersubstancedata(self.array_types):

            if self._init_relative_buoyancy is None:
                self._set_init_relative_buoyancy(substance)

            if len(data['fay_area']) == 0:
                # no particles released yet
                continue
        sc.update_from_fatedataview()

    def weather_elements(self, sc, time_step, model_time):
        '''
        Update 'area', 'fay_area' for previously released particles
        The updated 'area', 'fay_area' is associated with age of particles at:

            model_time + time_step
        '''
        if not self.active or not sc.substance.is_weatherable:
            return

        for substance, data in sc.itersubstancedata(self.array_types):

            if len(data['fay_area']) == 0:
                continue

            for s_num in np.unique(data['spill_num']):
                s_mask = data['spill_num'] == s_num
# change maximum area here
                data['max_area_le'][s_mask] = (data['init_mass'][s_mask] / data['density'][s_mask]) / self.thickness_limit
# change maximum area here
#                if count != 0:
#                   tmp = int(r_time_scale * sc.total_num_release[s_num] / sc.release_duration[s_num])
#                   s_mask[-1*count*tmp:] = False
# import release duration here to control init_oil_volume for continue release 12-14-2021
                data['fay_area'][s_mask] = \
                    self.update_area(water_kvis,
                                     self._init_relative_buoyancy,
                                     data['bulk_init_volume'][s_mask],
                                     data['fay_area'][s_mask],
                                     data['max_area_le'][s_mask],
                                     time_step,
                                     data['vol_frac_le_st'][s_mask],
                                     data['age'][s_mask] + time_step)
# import release duration here to control init_oil_volume for continue release 12-14-2021
    #                     self.update_area2(water_kvis,
    #                                      self._init_relative_buoyancy,
    #                                      data['bulk_init_volume'][s_mask],
    #                                      data['fay_area'][s_mask],
    #                                      time_step,
    #                                      data['age'][s_mask] + time_step)

    #                     self.update_area(water_kvis,
    #                                      self._init_relative_buoyancy,
    #                                      data['bulk_init_volume'][s_mask],
    #                                      data['fay_area'][s_mask],
    #                                      data['age'][s_mask] + time_step)

                data['area'][s_mask] = data['fay_area'][s_mask]

        sc.update_from_fatedataview()


class ConstantArea(Weatherer):
    '''
    Used for testing and diagnostics
    - must be manually hooked up
    '''
    _ref_as = 'spreading'

    def __init__(self, area, **kwargs):
        self.area = area
        super(ConstantArea, self).__init__(**kwargs)

        self.array_types.update({'area': gat('area'),
                                 'fay_area': gat('fay_area')})

    def initialize_data(self, sc, num_released):
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

            data['fay_area'][:] = self.area
            data['area'][:] = self.area

        sc.update_from_fatedataview()


class LangmuirSchema(WeathererSchema):
    wind = GeneralGnomeObjectSchema(
        acceptable_schemas=[WindSchema, VectorVariableSchema],
        save=True, update=True, save_reference=True
    )
    water = WaterSchema(
        save=True, update=True, save_reference=True
    )


class Langmuir(Weatherer):
    '''
    Easiest to define this as a weathering process that updates 'area' array
    '''
    _schema = LangmuirSchema
    _ref_as = 'langmuir'
    _req_refs = ['water', 'wind']

    def __init__(self,
                 water=None,
                 wind=None,
                 **kwargs):
        '''
        initialize wind to (0, 0) if it is None
        '''
        self.wind = wind

        # need water object to find relative buoyancy
        self.water = water

        super(Langmuir, self).__init__(**kwargs)
        self.array_types.update({'fay_area': gat('fay_area'),
                                 'area': gat('area'),
                                 'bulk_init_volume': gat('bulk_init_volume'),
                                 'age': gat('age'),
                                 'positions': gat('positions'),
                                 'spill_num': gat('spill_num'),
                                 'frac_coverage': gat('frac_coverage'),
                                 'density': gat('density')})


    def _get_frac_coverage(self, points, model_time, rel_buoy, thickness):
        '''
        return fractional coverage for a blob of oil with inputs;
        relative_buoyancy, and thickness

        Assumes the thickness is the minimum oil thickness associated with
        max area achievable by Fay Spreading

        Frac coverage bounds are constants. If computed frac_coverge is outside
        the bounds of (0.1, or 1.0), then limit it to:

            0.1 <= frac_cov <= 1.0
        '''
        # fixme: sometimes get v_max of zero
        #        probably shouldn't
        # explore v to be dependent of particle locations
        v_max = np.max(self.get_wind_speed(points, model_time) * .005)

        # typo in equation, 4 should be in denominator
        # cr_k = (v_max ** 2 *
        #         4 *
        #         PI ** 2 /
        #         (thickness * rel_buoy * gravity)) ** (1. / 3.)
        # cr_k[np.isnan(cr_k)] = 10.  # if density becomes equal to water density
        # cr_k[cr_k == 0] = 1.
        # frac_cov = 1. / cr_k

        # refactored to compute more directly:
        # fixme: we are getting warnings when rel_buoy <= 0.0
        #        that is, when the density of the oil is >= water
        #        this gets caught in the next line, but the warnings
        #        are kind of annoying -- can we catch this sooner?
        #        and is this doing the right thing for a "sinking" oil?
        #old_settings = np.seterr(divide='ignore')
        with np.errstate(divide='ignore'):
            frac_cov = (v_max ** 2 *
#                        4 *
                        PISQUARED /
                        (4 * thickness * rel_buoy * gravity)) ** (-0.3333333333333333)
        # due to oil density > water density
        # np.seterr(**old_settings)
        # with np.errstate(invalid='raise') :

        frac_cov[np.isnan(frac_cov)] = 0.1

        # clip takes care of inf
        np.clip(frac_cov, 0.1, 1.0, out=frac_cov)

        return frac_cov

    def _wind_speed_bound(self, rel_buoy, thickness):
        '''
        return min/max wind speed for given rel_buoy, thickness such that
        Langmuir effect is within bounds:

            0.1 <= frac_coverage <= 1.0
        '''
        v_min = np.sqrt(4.0 * thickness * rel_buoy * gravity /
                        (PISQUARED)) / 0.005
        v_max = np.sqrt((1. / 0.1) ** 3 * 4 * thickness * rel_buoy * gravity /
                        (PISQUARED)) / 0.005

        return (v_min, v_max)

    def weather_elements(self, sc, time_step, model_time):
        '''
        set the 'area' array based on the Langmuir process
        This only applies to particles marked for weathering on the surface:
        ie fate_status is surface_weather
        '''
        if not self.active or sc.num_released == 0:
            return

        #return
        rho_h2o = self.water.get('density', 'kg/m^3')
        for _, data in sc.itersubstancedata(self.array_types):
            #if len(data['area']) == 0:
            if len(data['fay_area']) == 0:
                continue

            points = data['positions']

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
                    self._get_frac_coverage(points, model_time, rel_buoy, thickness)

            # update 'area'
            data['area'][:] = data['fay_area'] * data['frac_coverage']

        sc.update_from_fatedataview()

