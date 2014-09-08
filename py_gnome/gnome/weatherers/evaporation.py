'''
model evaporation process
'''
from gnome.array_types import (mass_components,
                               density,
                               thickness,
                               mol,
                               evap_decay_constant)
from gnome.utilities.serializable import Serializable

from gnome.movers.movers import Process
from gnome.environment import constants, water


class Evaporation(Process, Serializable):
    def __init__(self,
                 wind,
                 frac_water=0.0,
                 frac_area=1.0,
                 **kwargs):
        '''
        :param frac_water: fractional water content in the emulsion
        :param frac_coverage: fraction of area covered by oil
        '''
        self.wind = wind
        self.frac_water = frac_water
        self.frac_area = frac_area
        super(Evaporation, self).__init__(**kwargs)
        self.array_types.update({'mass_components': mass_components,
                                 'density': density,
                                 'thickness': thickness,
                                 'mol': mol,
                                 'evap_decay_constant': eval_decay_constant})
        self.vapor_pressure = None
        self._decay = 0.0   # initialize to no decay

    def prepare_for_model_step(self, sc, time_step, model_time):
        '''
        Set arrays used by evaporation module for this timestep:

            - set vapor_pressure attribute for saturates/aromatics
              if it is None, so this only happens in first step
            - total number of moles
            - exponential decay factor for modeling evaporation
        '''

        # for now temp is fixed so compute vapor_pressure once
        super(Evaporation, self).prepare_for_model_step(sc,
                                                        time_step,
                                                        model_time)
        K = self._mass_transport_coeff(model_time)
        f_diff = 1.0 - self.frac_water
        for spill in sc.spills:
            self._set_vapor_pressure(spill)
            mask = sc.get_spill_mask(spill)
            mw = spill.get('substance').molecular_weight
            sc['thickness'][mask] = self._compute_le_thickness()
            sc['density'][mask] = spill.get('substance').density
            sc['mol'][mask] = \
                np.sum(sc['mass_components'][mask, :len(mw)]/mw, 1)
            le_area = \
                (sc['mass'][mask]/sc['density'][mask]) / sc['thickness'][mask]

            self._decay = (le_area * K * sc.get('vapor_pressure') *
                sc['mass_components'][mask, :])/(constants['gas_constant'] *
                water['temperature'] * sc['mol'][mask]) *
                self.frac_area * (1 - self.frac_water)

    def _compute_le_thickness(self):
        '''
        some function to compute LE thickness
        '''
        return 1.

    def _mass_transport_coeff(self, model_time):
        '''
        Is wind a function of only model_time? How about time_step?
        at present yes since wind only contains timeseries data

            K = c * U ** 0.78 if U <= 10 m/s
            K = 0.06 * c * U ** 2 if U > 10 m/s

        If K is expressed in m/sec, then Buchanan and Hurford set c = 0.0025
        U is wind_speed 10m above the surface
        '''
        (wind_speed, ) = self.wind.get_value(model_time)
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
                            sc['evap_decay_rate'],
                            time_step)
        return mass_remain
