'''
model evaporation process
'''
from math import exp, log

from gnome.array_types import mass_components, density
from gnome.utilities.serializable import Serializable

from gnome.movers.movers import Process
from gnome.environment import water, atmos, constants


def vapor_pressure(bp):
    '''
    water_temp and boiling point units are Kelvin
    returns the vapor_pressure in SI units (Pascals)
    '''
    D_Zb = 0.97
    R_cal = 1.987  # calories

    D_S = 8.75 + 1.987 * log(bp)
    C_2i = 0.19 * bp - 18

    var = 1. / (bp - C_2i) - 1. / (water['temperature'] - C_2i)
    ln_Pi_Po = D_S * (bp - C_2i) ** 2 / (D_Zb * R_cal * bp) * var
    Pi = exp(ln_Pi_Po) * atmos['pressure']

    return Pi


class Evaporation(Process, Serializable):
    def __init__(self,
                 f_water_content,
                 wind,
                 water_props,
                 **kwargs):
        '''
        :param f_water_content: fractional water content in the emulsion
        '''
        self.frac_water_content = f_water_content
        self.wind = wind
        super(Evaporation, self).__init__(**kwargs)
        self.array_types.update({'mass_components': mass_components,
                                 'density': density})
        self.vapor_pressure = None

    def prepare_for_model_step(self, sc, time_step, model_time):
        '''
        Set vapor_pressure attribute for saturates/aromatics
        compute the exponential decay factor for this timestep here
        '''

        # for now temp is fixed so compute vapor_pressure once
        super(Evaporation, self).prepare_for_model_step(sc,
                                                        time_step,
                                                        model_time)
