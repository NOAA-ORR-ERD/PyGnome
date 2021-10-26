'''
This module was originally intended to hold classes that ze_dze weathering
data arrays that are not set by any weathering process. It was also meant to
update the intrinsic properties of the LEs, hence the name 'intrinsic.py'
However, it sets and updates weathering data arrays including intrinsic data
like 'viscosity', 'density' and other data. Call the class WeatheringData()
which is defined in a gnome model if there are weatherers defined.

For now just define a FayGravityInertial class here
It is only used by WeatheringData to update the 'area' and related arrays
'''

import numpy as np

from functools import lru_cache

from gnome.basic_types import oil_status, fate
from gnome.array_types import gat

from .core import Weatherer, WeathererSchema
from gnome.environment.water import WaterSchema


class WeatheringDataSchema(WeathererSchema):
    water = WaterSchema(
        save=True, update=True, save_reference=True
    )


class WeatheringData(Weatherer):
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

    _schema = WeatheringDataSchema
    _ref_as = 'weathering_data'
    _req_refs = ['water']

    def __init__(self,
                 water=None,
                 **kwargs):
        '''
        initialize WeatheringData object.

        :param water: requires a water object
        :type water: gnome.environment.Water

        Options arguments kwargs: these get passed to base class via super
        '''
        super(WeatheringData, self).__init__(**kwargs)

        self.water = water
        self.array_types = {'fate_status': gat('fate_status'),
                            'positions': gat('positions'),
                            'status_codes': gat('status_codes'),
                            'density': gat('density'),
                            'viscosity': gat('viscosity'),
                            'mass_components': gat('mass_components'),
                            'mass': gat('mass'),
                            'oil_density': gat('oil_density'),
                            'oil_viscosity': gat('oil_viscosity'),
                            'init_mass': gat('init_mass'),
                            'frac_water': gat('frac_water'),
                            'frac_lost': gat('frac_lost'),	# change to frac_dissolved
                            'frac_evap': gat('frac_evap'),
                            'age': gat('age')}

        # following used to update viscosity
        self.visc_curvfit_param = 1.5e3     # units are sec^0.5 / m
        self.visc_f_ref = 0.84

    def prepare_for_model_run(self, sc):
        """
        Initialize mass balance:
           avg_density, floating, amount_released, avg_viscosity to 0.0        '''
        """
        for key in ('avg_density', 'floating', 'amount_released', 'non_weathering',
                    'avg_viscosity'):
            sc.mass_balance[key] = 0.0

    def initialize_data(self, sc, num_released):
        '''
        If on is False, then arrays should not be included
        - don't initialize

        1. initialize all weathering data arrays
          - density
          - viscosity
          - others??

        2. update aggregated data in sc.mass_balance dict
        '''
        substance = sc.substance
        if substance.is_weatherable:
            water = self.water

            # Only the new elements need to be initialized
            sl = slice(-num_released, None, 1)
            water_temp = water.get('temperature', 'K')
            density = substance.density_at_temp(water_temp)
            if density > water.get('density'):
                msg = ("{0} will sink at given water temperature: {1} {2}. "
                       "Setting density to water density"
                       .format(self.name,
                               water.get('temperature',
                                         self.water.units['temperature']),
                               water.units['temperature']))
                self.logger.error(msg)

                sc['density'][sl] = water.get('density')
            else:
                sc['density'][sl] = density

            sc['viscosity'][sl] = substance.kvis_at_temp(water_temp)
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

        for substance, data in sc.itersubstancedata(self.array_types, fate_status='all'):

            if not substance.is_weatherable or len(data['density']) == 0:
                self._aggregated_data(sc, 0)
                continue

            k_rho = self._get_k_rho_weathering_dens_update(substance,
                                                           self.water.get('temperature', 'K'))

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
                self.logger.info('{0} during update, density is larger '
                                 'than water density - set to water density'
                                 .format(self._pid))

            data['density'] = new_rho
            data['oil_density'] = oil_rho

            # following implementation results in an extra array called
            # fw_d_fref but is easy to read
            v0 = substance.kvis_at_temp(self.water.get('temperature', 'K'))

            if v0 is not None:
                kv1 = self._get_kv1_weathering_visc_update(v0, self.visc_curvfit_param)
                fw_d_fref = data['frac_water'] / self.visc_f_ref

                data['viscosity'] = (v0 * np.exp(kv1 * data['frac_evap']) * (1 + (fw_d_fref / (1.187 - fw_d_fref))) ** 2.49 )
                data['oil_viscosity'] = (v0 * np.exp(kv1 * data['frac_evap']))

            #sc.data_arrays['fate_status'][:] = np.choose(np.isclose(sc.data_arrays['mass'], 0), [sc.data_arrays['fate_status'], fate.non_weather])
            #zeros = np.isclose(sc.data_arrays['mass'], 0)
            #sc.data_arrays['mass'][zeros] = 0
            #sc.data_arrays['mass_components'][zeros] = 0

        sc.update_from_fatedataview(fate_status='all')

            # also initialize/update aggregated data
        self._aggregated_data(sc, 0)

    def _aggregated_data(self, data, new_LEs):
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
        # avg_density, avg_viscosity applies to elements that are on the
        # surface and being weathered

        if data['mass'].sum() > 0.0:
            data.mass_balance['avg_density'] = \
                np.sum(data['mass']/data['mass'].sum() * data['density'])
            data.mass_balance['avg_viscosity'] = \
                np.sum(data['mass']/data['mass'].sum() * data['viscosity'])
        else:
            self.logger.info("{0} sum of 'mass' array went to 0.0"
                             .format(self._pid))

        # floating includes LEs marked to be skimmed + burned + dispersed
        # todo: remove fate_status and add 'surface' to status_codes. LEs
        # marked to be skimmed, burned, dispersed will also be marked as
        # 'surface' so following can get cleaned up.
        # fixme: refactor to build up the mask, and then apply it.
        # sc.mass_balance['floating'] = \
        #     (sc['mass'][sc['fate_status'] == fate.surface_weather].sum() +
        #      sc['mass'][sc['fate_status'] == fate.non_weather].sum() -
        #      sc['mass'][sc['status_codes'] == oil_status.on_land].sum() -
        #      sc['mass'][sc['status_codes'] == oil_status.to_be_removed].sum() +
        #      sc['mass'][sc['fate_status'] & fate.skim == fate.skim].sum() +
        #      sc['mass'][sc['fate_status'] & fate.burn == fate.burn].sum() +
        #      sc['mass'][sc['fate_status'] & fate.disperse == fate.disperse].sum())

        on_surface = ((data['status_codes'] == oil_status.in_water) &
                      (data['positions'][:,2] == 0.0))

        data.mass_balance['floating'] = data['mass'][on_surface].sum()

        # add 'non_weathering' key if any mass is released for nonweathering
        # particles.
        data.mass_balance['non_weathering'] = data['mass'][data['fate_status'] == fate.non_weather].sum()

        if new_LEs > 0:
            amount_released = np.sum(data['mass'][-new_LEs:])

            if 'amount_released' in data.mass_balance:
                data.mass_balance['amount_released'] += amount_released
            else:
                data.mass_balance['amount_released'] = amount_released

    @staticmethod
    @lru_cache(1)
    def _get_kv1_weathering_visc_update(v0, visc_curvfit_param):
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
        kv1 = np.sqrt(v0) * visc_curvfit_param

        if kv1 < 1:
            kv1 = 1
        elif kv1 > 10:
            kv1 = 10

        return kv1

    @staticmethod
    @lru_cache(1)
    def _get_k_rho_weathering_dens_update(substance, temp_in_k):
        '''
        use lru_cache on substance. substance is expected to be a GnomeOil,
        if this object stays the same, then return the cached value for k_rho
        This depends on initial mass fractions, initial density and fixed
        component densities
        '''
        # update density/viscosity/relative_buoyancy/area for previously
        # released elements
        rho0 = substance.density_at_temp(temp_in_k)

        # dimensionless constant
        k_rho = (rho0 /
                 (substance.component_density * substance.mass_fraction).sum())

        return k_rho
