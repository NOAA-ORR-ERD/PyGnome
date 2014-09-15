'''
OilProps class which serves as a wrapper around
gnome.db.oil_library.models.Oil class

It also contains a dict containing a small number of sample_oils if user does
not wish to query database to use an _r_oil.

It contains a function that takes a string for 'oil_name' and returns and Oil
object used to initialize and OilProps object

Not sure at present if this needs to be serializable?
'''

from itertools import chain
from collections import namedtuple
import math

from hazpy import unit_conversion
uc = unit_conversion


MassComponent = namedtuple('MassComponent',
                           ''' fraction,
                               halflife,
                           ''')


def boiling_point(num_pc, api):
    '''
    return an array of boiling points for each psuedo-components
    Assume 5 pseudo-components
        T_o = 457.16 - 3.3447 * API
        dT/df = 1356.7 - 247.36*ln(API)
        for i = range(num_pc):
            boiling_point = T_o + dT_dF * (i + 0.5)/num_pc

    boiling point units of Kelvin
    '''
    T_o = 457.16 - 3.3447 * api
    dT_dF = 1356.7 - 247.36 * math.log(api)

    # bp = [T_o + dT_dF * ((ix + 0.5)/num_pc) for ix in range(num_pc)]
    bp = [float('nan')] * num_pc
    for ix in range(num_pc):
        # since ix is 0 based indexing, we get ((ix + 1) - 0.5) = (ix + 0.5)
        bp[ix] = T_o + dT_dF * ((ix + 0.5) / num_pc)

    return bp


def vapor_pressure_ratio(bp, water_temp, P_atmos=101325.0):
    '''
    water_temp and boiling point units are Kelvin
    returns the ratio: vapor_pressure/atmospheric_pressure
    '''
    D_Zb = 0.97
    R_cal = 1.987  # calories

    D_S = 8.75 + 1.987 * math.log(bp)
    C_2i = 0.19 * bp - 18

    var = 1. / (bp - C_2i) - 1. / (water_temp - C_2i)
    ln_Pi_Po = D_S * (bp - C_2i) ** 2 / (D_Zb * R_cal * bp) * var
    Pi_atmos = math.exp(ln_Pi_Po)

    return Pi_atmos


def mw_saturate(bp):
    '''
    return the molecular weight of the pseudocomponents (mw_i) given the
    boiling points. It returns the mw_i for saturates and aromatic components
    '''
    mw_s = (0.04132 - 1.985e-4 * bp + (9.494e-7 * bp ** 2))

    return mw_s


class OilProps(object):
    '''
    Class which:
    - Contains an oil object.
    - Provides more sophisticated oil properties than the basic oil database
      object.
    - These properties are the result of calculations made upon the
      basic oil database properties.
    - Generally speaking, we will try to adhere to the ASTM standards and use
      the SI measurement system when determining units for our values.
      The SI base units are consistent with the MKS system.

    Specifically, OilProps has a few categories of properties:
    Density:
    - returns a scalar as opposed to a list of Densities.

    Viscosity:

    '''

    def __init__(self, oil_, temperature=311.15):
        '''
        If oil_ is amongst self._sample_oils dict, then use the properties
        defined here. If not, then query the Oil database to check if oil_
        exists and get the properties from DB.

        :param oil_: Oil object that maps to entity in OilLib database
        :type oil_: Oil object
        :param water_temp: The temperature in 'K'.  Per ASTM, the default is
                           38 degrees Celcius (311.15 degrees Kelvin)
        '''
        self._r_oil = oil_
        self.num_pc = 5     # probably determine this from data
        #======================================================================
        # self.mass_components = [0.] * self.num_pc
        # self.mass_components[0] = 1.
        # self.hl = [float('inf')] * self.num_pc
        # self.hl[0] = 15.*60
        #======================================================================
        self._temperature = temperature

        self.boiling_point = boiling_point(self.num_pc, self.get('api'))
        self.vapor_pressure_ratio = []
        self.mw_saturates = []
        for bp in self.boiling_point:
            self.vapor_pressure_ratio.append(
                vapor_pressure_ratio(bp, self.temperature))
            self.mw_saturates.append(mw_saturate(bp))

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'oil_={0._r_oil!r}, water_temp={0.water_temp}'
                ')'.format(self))

    density = property(lambda self: self.get_density())
    temperature = property(lambda self: self.get_temperature())
    name = property(lambda self: self._r_oil.name,
                    lambda self, val: setattr(self._r_oil, 'name', val))
    api = property(lambda self: self.get('api'))

    def get(self, prop):
        'get raw oil props'
        return getattr(self._r_oil, prop)

    def get_temperature(self, units='K'):
        return uc.convert('Temperature', 'K', units, self._temperature)

    def set_temperature(self, value, units):
        temp = uc.convert('Temperature', units, 'K', value)
        self._temperature = temp
        # update dependencies
        self.vapor_pressure = []
        for ix, bp in enumerate(self.boiling_point):
            self.vapor_pressure_ratio.append(
                vapor_pressure_ratio(bp, self._temperature))

    @property
    def mass_components(self):
        '''
           Gets the mass components of our _r_oil
           - Set 'mass_components' array based on mass fractions
             (distillation cuts?) that are found in the _r_oil library
           - Set 'half-lives' array based on ???
           TODO: Right now this is just a stub that returns a hardcoded value
                 for testing purposes.
                 - Try to query our distillation cuts and see if they are
                   usable.
                 - Figure out where we will get the half-lives data.
        '''
        mc = (1., 0., 0., 0., 0.)
        hl = ((15. * 60), float('inf'),
              float('inf'), float('inf'),
              float('inf'))
        return [MassComponent(*n) for n in zip(mc, hl)]

    def get_density(self, units='kg/m^3'):
        '''
        :param units: optional input if output units should be something other
                      than kg/m^3
        :return: scalar Density.  Default units: (kg/m^3)
        '''

        if self.api is None:
            raise ValueError("Oil with name '{0}' does not contain 'api'"
                             " property.".format(self._r_oil.name))

        # since Oil object can have various densities depending on temperature,
        # lets return API in correct units
        return uc.convert('Density', 'API degree', units, self.api)

    def get_viscosity(self, units='m^2/s'):
        '''
        :param units: optional input if output units should be something other
                      than kg/m^3
        :return: scalar Density.  Default units: (kg/m^3)

        - The kinematic viscosity (nu) is the ratio of the dynamic viscosity
          (mu) to the density of the fluid (rho). (nu = mu / rho)  It is
          measured in (m^2/s)
        - 1 m^2/s = 1000000 centistokes (cSt)
        '''

        if self.api is None:
            raise ValueError("Oil with name '{0}' does not contain 'api'"
                             " property.".format(self._r_oil.name))

        # since Oil object can have various densities depending on temperature,
        # lets return API in correct units
        return uc.convert('Density', 'API degree', units, self.api)
