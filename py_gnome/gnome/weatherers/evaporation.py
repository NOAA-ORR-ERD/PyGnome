'''
model evaporation process
'''
import numpy as np

from gnome.array_types import (mass_components,
                               density,
                               thickness,
                               mol,
                               evap_decay_constant)
from gnome.utilities.serializable import Serializable

from gnome.weatherers import Weatherer
from gnome.environment import constants, constant_wind


class Evaporation(Weatherer):
    def __init__(self,
                 water_props,
                 wind=None,
                 **kwargs):
        '''
        :param conditions: gnome.environment.Conditions object which contains
            things like water temperature
        :param wind: wind object for obtaining speed at specified time
        :type wind: Wind API, specifically must have get_value(time) method
        '''
        self.water = water_props

        if wind is None:
            wind = constant_wind(0, 0)

        self.wind = wind

        super(Evaporation, self).__init__(**kwargs)
        self.array_types.update({'mass_components': mass_components,
                                 'density': density,
                                 'thickness': thickness,
                                 'mol': mol,
                                 'evap_decay_constant': evap_decay_constant})
        self._decay = 0.0   # initialize to no decay

    def prepare_for_model_step(self, sc, time_step, model_time):
        '''
        Set/update arrays used by evaporation module for this timestep:

            - 'mol': total number of moles
            - 'evap_decay_rate': exponential decay factor for evaporation
            - 'thickness': still need a way to compute LE thickness
            - 'density': update this from the OilProps

        .. note::
        The restricting term due to liquid diffusion limitations on the more
        volatile hydrocarbons, slowing their access to the air-oil interface is
        modeled as (1.0 - water_fraction_of_emulsion), per Robert Jones
        If this is used by any other process, then move it to Spill object. Or
        if we want to model it differently, then refactor it out of here
        '''

        # for now temp is fixed so compute vapor_pressure once
        super(Evaporation, self).prepare_for_model_step(sc,
                                                        time_step,
                                                        model_time)
        K = self._mass_transport_coeff(model_time)
        water_temp = self.water.get('temperature', 'K')

        for spill in sc.spills:
            f_diff = (1.0 - spill.frac_water)
            mask = sc.get_spill_mask(spill)
            mw = spill.get('substance').molecular_weight
            vp = spill.get('substance').vapor_pressure(water_temp)
            sc['thickness'][mask] = self._compute_le_thickness()
            sc['density'][mask] = \
                spill.get('substance').get_density(temp=water_temp)
            sc['mol'][mask] = \
                np.sum(sc['mass_components'][mask, :]/mw, 1)
            le_area = \
                (sc['mass'][mask]/sc['density'][mask]) / sc['thickness'][mask]
            le_area = le_area.reshape(-1, 1)

            d_numer = (le_area * K * vp * spill.frac_coverage * f_diff)
            d_denom = (constants['gas_constant'] * water_temp *
                       sc['mol'][mask]).reshape(-1, 1)
            d_denom = np.repeat(d_denom, d_numer.shape[1], axis=1)
            sc['evap_decay_constant'][mask, :] = -d_numer/d_denom
            if np.any(sc['evap_decay_constant'][mask, :] > 0.0):
                raise ValueError("Error in Evaporation routine. One of the "
                                 "exponential decay constant is positive")

    #def _round_decay_const(self, decay_const, time_step):

    def _compute_le_thickness(self):
        '''
        some function to compute LE thickness
        '''
        return 1e-3

    def _mass_transport_coeff(self, model_time):
        '''
        Is wind a function of only model_time? How about time_step?
        at present yes since wind only contains timeseries data

            K = c * U ** 0.78 if U <= 10 m/s
            K = 0.06 * c * U ** 2 if U > 10 m/s

        If K is expressed in m/sec, then Buchanan and Hurford set c = 0.0025
        U is wind_speed 10m above the surface
        '''
        wind_speed = self.wind.get_value(model_time)[0]
        c_evap = 0.0025     # if wind_speed in m/s
        if wind_speed <= 10.0:
            return c_evap * wind_speed ** 0.78
        else:
            return 0.06 * c_evap * wind_speed ** 2

    def weather_elements(self, sc, time_step, model_time):
        '''
        weather elements over time_step
        '''
        if not self.active:
            return sc['mass_components']

        mass_remain = \
            self._exp_decay(sc['mass_components'],
                            sc['evap_decay_constant'],
                            time_step)
        return mass_remain
