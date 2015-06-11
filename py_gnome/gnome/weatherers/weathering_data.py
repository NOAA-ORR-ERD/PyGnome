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
from gnome.utilities.serializable import Serializable, Field
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
    # UI does not need to manipulate - if make_default_refs is True as is the
    # default, it'll automatically get the default Water object
    _state += Field('water', save=True, update=False, save_reference=True)

    _schema = WeathererSchema

    def __init__(self, water, **kwargs):
        '''
        initialize object.

        :param water: requires a water object
        :type water: gnome.environment.Water

        Options arguments kwargs: these get passed to base class via super
        '''
        super(WeatheringData, self).__init__(**kwargs)
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
            sc.mass_balance[key] = 0.0

    def initialize_data(self, sc, num_released):
        '''
        If on is False, then arrays should not be included - dont' initialize

        1. initialize all weathering data arrays
        2. update aggregated data in sc.mass_balance dict
        '''
        if not self.on:
            return

        for substance, data in sc.itersubstancedata(self.array_types,
                                                    fate='all'):
            'update properties only if elements are released'
            if len(data['density']) == 0:
                continue

            # could also use 'age' but better to use an uninitialized var since
            # we might end up changing 'age' to something other than 0
            # model should only call initialize_data if new particles were
            # released
            new_LEs_mask = data['density'] == 0
            if np.any(new_LEs_mask):
                self._init_new_particles(new_LEs_mask, data, substance)

        sc.update_from_fatedataview(fate='all')

        # also initialize/update aggregated data
        self._aggregated_data(sc, num_released)

    def weather_elements(self, sc, time_step, model_time):
        '''
        Update intrinsic property data arrays: density, viscosity.
        In a model step, this is the last thing that happens. All the
        weatherers update 'mass_components' so mass_fraction will have changed
        at the end of the timestep. Update the density and viscosity
        accordingly.
        '''
        if not self.active:
            return

        water_rho = self.water.get('density')

        for substance, data in sc.itersubstancedata(self.array_types,
                                                    fate='all'):
            'update properties only if elements are released'
            if len(data['density']) == 0:
                continue

            k_rho = self._get_k_rho_weathering_dens_update(substance)

            # sub-select mass_components array by substance.num_components.
            # Currently, physics for modeling multiple spills with different
            # substances is not correctly done in the same model. However,
            # let's put some basic code in place so the data arrays can infact
            # contain two substances and the code does not raise exceptions.
            # mass_components are zero padded for substance which has fewer
            # psuedocomponents. Subselecting mass_components array by
            # [mask, :substance.num_components] ensures numpy operations work
            mass_frac = \
                (data['mass_components'][:, :substance.num_components] /
                 data['mass'].reshape(len(data['mass']), -1))
            # check if density becomes > water, set it equal to water in this
            # case - 'density' is for the oil-water emulsion
            oil_rho = k_rho*(substance.component_density * mass_frac).sum(1)

            # oil/water emulsion density
            new_rho = (data['frac_water'] * water_rho +
                       (1 - data['frac_water']) * oil_rho)
            if np.any(new_rho > self.water.density):
                new_rho[new_rho > self.water.density] = self.water.density
                self.logger.info(self._pid + "during update, density is larger"
                                 " than water density - set to water density")

            data['density'] = new_rho

            # following implementation results in an extra array called
            # fw_d_fref but is easy to read
            v0 = substance.get_viscosity(self.water.get('temperature', 'K'))
            if v0 is not None:
                kv1 = self._get_kv1_weathering_visc_update(v0)
                fw_d_fref = data['frac_water']/self.visc_f_ref
                data['viscosity'] = \
                    (v0 * np.exp(kv1 * data['frac_lost']) *
                     (1 + (fw_d_fref/(1.187 - fw_d_fref)))**2.49)

        sc.update_from_fatedataview(fate='all')

        # also initialize/update aggregated data
        self._aggregated_data(sc, 0)

    def _aggregated_data(self, sc, new_LEs):
        '''
        aggregated properties that are not set by any other weatherer are
        set here. The following keys in sc.mass_balance are set here:
            'avg_density',
            'avg_viscosity',
            'floating',
            'amount_released',
        todo: amount_released and beached can probably get set by
            SpillContainer. The trajectory only case will probably also care
            about amount 'beached'.
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
                sc.mass_balance['avg_density'] = \
                    np.sum(data['mass']/data['mass'].sum() * data['density'])
                sc.mass_balance['avg_viscosity'] = \
                    np.sum(data['mass']/data['mass'].sum() * data['viscosity'])
            else:
                self.logger.info(self._pid + "sum of 'mass' array went to 0.0")

        # floating includes LEs marked to be skimmed + burned + dispersed
        # todo: remove fate_status and add 'surface' to status_codes. LEs
        # marked to be skimmed, burned, dispersed will also be marked as
        # 'surface' so following can get cleaned up.
        sc.mass_balance['floating'] = \
            (sc['mass'][sc['fate_status'] == fate.surface_weather].sum() +
             sc['mass'][sc['fate_status'] & fate.skim == fate.skim].sum() +
             sc['mass'][sc['fate_status'] & fate.burn == fate.burn].sum() +
             sc['mass'][sc['fate_status'] & fate.disperse == fate.disperse].sum())

        #sc.mass_balance['beached'] = sc['mass'][sc['status_codes'] ==
        #                                           oil_status.on_land].sum()

        # add 'non_weathering' key if any mass is released for nonweathering
        # particles.
        nonweather = sc['mass'][sc['fate_status'] == fate.non_weather].sum()
        sc.mass_balance['non_weathering'] = nonweather

        if new_LEs > 0:
            amount_released = np.sum(sc['mass'][-new_LEs:])
            if 'amount_released' in sc.mass_balance:
                sc.mass_balance['amount_released'] += amount_released
            else:
                sc.mass_balance['amount_released'] = amount_released

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

    def serialize(self, json_='webapi'):
        '''
        No need to return serialized version of this for WebAPI
        User does not manipuate it. It automatically uses default Water object
        if make_default_refs is True
        '''
        if json_ == 'webapi':
            return

        # for save files - call super
        return super(WeatheringData, self).serialize(json_)

    @classmethod
    def deserialize(cls, json_):
        '''
        do not expect to get this object from 'webapi' since it isn't being
        serialized for 'webapi'. User does not manipulate it.
        If we wish to display to user, this can be updated - just need to add
        a serialized 'wind' object.
        '''
        if json_['json_'] == 'save':
            return super(cls, WeatheringData).deserialize(json_)
