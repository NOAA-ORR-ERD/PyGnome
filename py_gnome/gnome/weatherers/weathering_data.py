'''
This module was originally intended to hold classes that initialize weathering
data arrays that are not set by any weathering process. It was also meant to
update the intrinsic properties of the LEs, hence the name 'intrinsic.py'
However, it sets and updates weathering data arrays including intrinsic data
like 'viscosity', 'density' and other data. Call the class WeatheringData()
which is defined in a gnome model if there are weatherers defined.

For now just define a FayGravityInertial class here
It is only used by WeatheringData to update the 'area' and related arrays
'''
import copy

import numpy as np
from repoze.lru import lru_cache

from gnome.basic_types import oil_status, fate
from gnome.utilities import Serializable
from .spreading import FayGravityViscous
from .core import Weatherer, WeathererSchema


class WeatheringData(Weatherer, Serializable):
    '''
    Serves to initialize weathering data arrays. Also updates data arrays
    like density, viscosity
    Doesn't have an id like other gnome objects. It isn't exposed to
    application since Model will automatically instantiate if there
    are any Weathering objects defined

    Use this to manage data_arrays associated with weathering that are not
    initialized anywhere else. This is inplace of defining initializers for
    every single array, let WeatheringData set/initialize/update these arrays.
    '''
    _state = copy.deepcopy(Weatherer._state)
    _schema = WeathererSchema

    def __init__(self,
                 water):
        '''
        XXX
        '''
        self.water = water
        self.array_types = {'fate_status', 'positions', 'status_codes',
                            'density', 'viscosity', 'mass_components', 'mass',
                            'init_mass', 'frac_water', 'frac_lost', 'age'}

        # following used to update viscosity
        self.visc_curvfit_param = 1.5e3     # units are sec^0.5 / m
        self.visc_f_ref = 0.84

    def prepare_for_model_run(self, sc):
        '''
        1. initialize standard keys:
           avg_density, floating, amount_released, avg_viscosity to 0.0
        2. set init_density for all ElementType objects in each Spill
        3. set spreading thickness limit based on viscosity of oil at
           water temperature which is constant for now.
        '''
        # nothing released yet - set everything to 0.0
        for key in ('avg_density', 'floating', 'amount_released',
                    'avg_viscosity', 'beached'):
            sc.weathering_data[key] = 0.0

    def update(self, num_new_released, sc):
        '''
        Uses 'substance' properties together with 'water' properties to update
        'density', 'bulk_init_volume', etc
        The 'bulk_init_volume' is not updated at each step; however, it depends on
        the 'density' which must be set/updated first and this depends on
        water object. So it was easiest to initialize the 'bulk_init_volume' for
        newly released particles here.
        '''
        if len(sc) > 0:
            self._update_weathering_dataarrays(sc)
            self._update_aggregated_data(num_new_released, sc)

    def _update_aggregated_data(self, new_LEs, sc):
        '''
        intrinsic LE properties not set by any weatherer so let SpillContainer
        set these - will user be able to use select weatherers? Currently,
        evaporation defines 'density' data array
        '''
        # update avg_density from density array
        # wasted cycles at present since all values in density for given
        # timestep should be the same, but that will likely change
        # Any optimization in doing the following?:
        #   (sc['mass'] * sc['density']).sum()/sc['mass'].sum()
        # todo: move weighted average to utilities
        # also added a check for 'mass' == 0, edge case
        if len(sc.substances) > 1:
            self.logger.warning(self._pid + "current code isn't valid for "
                                "multiple weathering substances")
        elif len(sc.substances) == 0:
            # should not happen with the Web API. Just log a warning for now
            self.logger.warning(self._pid + "weathering is on but found no"
                                "weatherable substances.")
        else:
            # avg_density, avg_viscosity applies to elements that are on the
            # surface and being weathered
            data = sc.substancefatedata(sc.substances[0],
                                        {'mass', 'density', 'viscosity'})
            if data['mass'].sum() > 0.0:
                sc.weathering_data['avg_density'] = \
                    np.sum(data['mass']/data['mass'].sum() * data['density'])
                sc.weathering_data['avg_viscosity'] = \
                    np.sum(data['mass']/data['mass'].sum() * data['viscosity'])
            else:
                self.logger.info(self._pid + "sum of 'mass' array went to 0.0")

        # floating includes LEs marked to be skimmed + burned + dispersed
        # todo: remove fate_status and add 'surface' to status_codes. LEs
        # marked to be skimmed, burned, dispersed will also be marked as
        # 'surface' so following can get cleaned up.
        sc.weathering_data['floating'] = \
            (sc['mass'][sc['fate_status'] == fate.surface_weather].sum() +
             sc['mass'][sc['fate_status'] & fate.skim == fate.skim].sum() +
             sc['mass'][sc['fate_status'] & fate.burn == fate.burn].sum() +
             sc['mass'][sc['fate_status'] & fate.disperse == fate.disperse].sum())

        sc.weathering_data['beached'] = sc['mass'][sc['status_codes'] ==
                                                   oil_status.on_land].sum()

        # add 'non_weathering' key if any mass is released for nonweathering
        # particles.
        nonweather = sc['mass'][sc['fate_status'] == fate.non_weather].sum()
        sc.weathering_data['non_weathering'] = nonweather

        if new_LEs > 0:
            amount_released = np.sum(sc['mass'][-new_LEs:])
            if 'amount_released' in sc.weathering_data:
                sc.weathering_data['amount_released'] += amount_released
            else:
                sc.weathering_data['amount_released'] = amount_released

    def _update_weathering_dataarrays(self, sc):
        '''
        - initialize 'density', 'viscosity', and other optional arrays for
        newly released particles.
        - update intrinsic properties like 'density', 'viscosity' and optional
        arrays for previously released particles
        '''

        for substance, data in sc.itersubstancedata(self.array_types,
                                                    fate='all'):
            'update properties only if elements are released'
            if len(data['density']) == 0:
                continue

            # could also use 'age' but better to use an uninitialized var since
            # we might end up changing 'age' to something other than 0
            new_LEs_mask = data['density'] == 0
            if sum(new_LEs_mask) > 0:
                self._init_new_particles(new_LEs_mask, data, substance)
            if sum(~new_LEs_mask) > 0:
                self._update_old_particles(~new_LEs_mask,
                                           data,
                                           substance,
                                           sc.current_time_stamp)

        sc.update_from_fatedataview(fate='all')

    def update_fate_status(self, sc):
        '''
        Update fate status after model invokes move_elements()
        - elements will beach or refloat
        - then Model will update fate_status of elements that beached/refloated

        Model calls this and input is spill container, not a view of the data
        '''
        # for old particles, update fate_status
        # particles in_water or off_maps continue to weather
        # only particles on_land stop weathering
        non_w_mask = sc['status_codes'] == oil_status.on_land
        sc['fate_status'][non_w_mask] = fate.non_weather

        # update old particles that may now have refloated
        # only want to do this for particles with a valid substance - if
        # substance is None, they do not weather
        # also get all data for a substance since we are modifying the
        # fate_status - lets not use it to filter data
        for substance, data in sc.itersubstancedata(self.array_types,
                                                    fate='all'):
            mask = data['fate_status'] & fate.non_weather == fate.non_weather
            self._init_fate_status(mask, data)

        sc.update_from_fatedataview(fate='all')

    def _init_new_particles(self, mask, data, substance):
        '''
        initialize new particles released together in a given timestep

        :param mask: mask gives only the new LEs in data arrays
        :type mask: numpy bool array
        :param data: dict containing numpy arrays
        :param substance: OilProps object defining the substance spilled
        '''
        water_temp = self.water.get('temperature', 'K')
        density = substance.get_density(water_temp)
        if density > self.water.get('density'):
            msg = ("{0} will sink at given water temperature: {1} {2}. "
                   "Set density to water density".
                   format(substance.name,
                          self.water.get('temperature',
                                         self.water.units['temperature']),
                          self.water.units['temperature']))
            self.logger.error(msg)
            data['density'][mask] = self.water.get('density')
        else:
            data['density'][mask] = density

        if self._init_relative_buoyancy is None:
            self._init_relative_buoyancy = \
                self._get_relative_buoyancy(data['density'][mask][0])

        # initialize mass_components -
        # sub-select mass_components array by substance.num_components.
        # Currently, the physics for modeling multiple spills with different
        # substances is not being correctly done in the same model. However,
        # let's put some basic code in place so the data arrays can infact
        # contain two substances and the code does not raise exceptions. The
        # mass_components are zero padded for substance which has fewer
        # psuedocomponents. Subselecting mass_components array by
        # [mask, :substance.num_components] ensures numpy operations work
        data['mass_components'][mask, :substance.num_components] = \
            (np.asarray(substance.mass_fraction, dtype=np.float64) *
             (data['mass'][mask].reshape(len(data['mass'][mask]), -1)))

        data['init_mass'][mask] = data['mass'][mask]

        if substance.get_viscosity(water_temp) is not None:
            'make sure we do not add NaN values'
            data['viscosity'][mask] = \
                substance.get_viscosity(water_temp)

        # initialize the fate_status array based on positions and status_codes
        self._init_fate_status(mask, data)

    def _init_fate_status(self, update_LEs_mask, data):
        '''
        initialize fate_status for newly released LEs or refloated LEs
        For refloated LEs, the mask should apply to non_weather LEs.
        Currently, the 'status_codes' is separate from 'fate_status' and we
        don't want to reset the 'fate_status' of LEs that have been marked
        as 'skim' or 'burn' or 'disperse'. This should only apply for newly
        released LEs (currently marked as non_weather since that's the default)
        and for refloated LEs which should also have been marked as non_weather
        when they beached.
        '''
        surf_mask = \
            np.logical_and(update_LEs_mask,
                           np.logical_and(data['positions'][:, 2] == 0,
                                          data['status_codes'] ==
                                          oil_status.in_water))
        subs_mask = \
            np.logical_and(update_LEs_mask,
                           np.logical_and(data['positions'][:, 2] > 0,
                                          data['status_codes'] ==
                                          oil_status.in_water))

        # set status for new_LEs correctly
        data['fate_status'][surf_mask] = fate.surface_weather
        data['fate_status'][subs_mask] = fate.subsurf_weather

    @lru_cache(1)
    def _get_kv1_weathering_visc_update(self, v0):
        '''
        kv1 is constant.
        It defining the exponential change in viscosity as it weathers due to
        the fraction lost to evaporation/dissolution:
            v(t) = v' * exp(kv1 * f_lost_evap_diss)

        kv1 = sqrt(v0) * 1500
        if kv1 < 1, then return 1
        if kv1 > 10, then return 10

        Since this is fixed for an oil, it only needs to be computed once. Use
        lru_cache on this function to cache the result for a given initial
        viscosity: v0
        '''
        # find kv1
        kv1 = np.sqrt(v0) * self.visc_curvfit_param
        if kv1 < 1:
            kv1 = 1

        if kv1 > 10:
            kv1 = 10

        return kv1

    @lru_cache(1)
    def _get_k_rho_weathering_dens_update(self, substance):
        '''
        use lru_cache on substance. substance is an OilProps object, if this
        object stays the same, then return the cached value for k_rho
        This depends on initial mass fractions, initial density and fixed
        component densities
        '''
        # update density/viscosity/relative_bouyance/area for previously
        # released elements
        rho0 = substance.get_density(self.water.get('temperature', 'K'))

        # dimensionless constant
        k_rho = (rho0 /
                 (substance.component_density * substance.mass_fraction).sum())

        return k_rho

    def _update_old_particles(self, mask, data, substance, model_time):
        '''
        update density, area
        '''
        k_rho = self._get_k_rho_weathering_dens_update(substance)

        water_rho = self.water.get('density')

        # must update intrinsic properties per spill. Same substance but
        # multiple spills - update intrinsic for each spill.
        for s_num in np.unique(data['spill_num'][mask]):
            s_mask = np.logical_and(mask,
                                    data['spill_num'] == s_num)
            # sub-select mass_components array by substance.num_components.
            # Currently, the physics for modeling multiple spills with different
            # substances is not being correctly done in the same model. However,
            # let's put some basic code in place so the data arrays can infact
            # contain two substances and the code does not raise exceptions. The
            # mass_components are zero padded for substance which has fewer
            # psuedocomponents. Subselecting mass_components array by
            # [mask, :substance.num_components] ensures numpy operations work
            mass_frac = \
                (data['mass_components'][s_mask, :substance.num_components] /
                 data['mass'][s_mask].reshape(np.sum(s_mask), -1))
            # check if density becomes > water, set it equal to water in this
            # case - 'density' is for the oil-water emulsion
            oil_rho = k_rho*(substance.component_density * mass_frac).sum(1)

            # oil/water emulsion density
            new_rho = (data['frac_water'][s_mask] * water_rho +
                       (1 - data['frac_water'][s_mask]) * oil_rho)
            if np.any(new_rho > self.water.density):
                new_rho[new_rho > self.water.density] = self.water.density
                self.logger.info(self._pid + "during update, density is larger"
                                 " than water density - set to water density")

            data['density'][s_mask] = new_rho

            # following implementation results in an extra array called
            # fw_d_fref but is easy to read
            v0 = substance.get_viscosity(self.water.get('temperature', 'K'))
            if v0 is not None:
                kv1 = self._get_kv1_weathering_visc_update(v0)
                fw_d_fref = data['frac_water'][s_mask]/self.visc_f_ref
                data['viscosity'][s_mask] = \
                    (v0 * np.exp(kv1 *
                                 data['frac_lost'][s_mask]) *
                     (1 + (fw_d_fref/(1.187 - fw_d_fref)))**2.49)
