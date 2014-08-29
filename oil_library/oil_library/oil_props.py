'''
OilProps class which serves as a wrapper around
gnome.db.oil_library.models.Oil class

It also contains a dict containing a small number of sample_oils if user does
not wish to query database to use an _r_oil.

It contains a function that takes a string for 'oil_name' and returns and Oil
object used to initialize and OilProps object

Not sure at present if this needs to be serializable?
'''
from math import exp, log

from collections import namedtuple

from hazpy import unit_conversion
uc = unit_conversion

from .models import KVis

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
    dT_dF = 1356.7 - 247.36 * log(api)

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

    D_S = 8.75 + 1.987 * log(bp)
    C_2i = 0.19 * bp - 18

    var = 1. / (bp - C_2i) - 1. / (water_temp - C_2i)
    ln_Pi_Po = D_S * (bp - C_2i) ** 2 / (D_Zb * R_cal * bp) * var
    Pi_atmos = exp(ln_Pi_Po)

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
                'oil_={0._r_oil!r}, water_temp={0.temperature}'
                ')'.format(self))

    density = property(lambda self: self.get_density())
    viscosity = property(lambda self: self.get_viscosity())
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

    @property
    def viscosities(self):
        '''
            Get a list of all kinematic viscosities associated with this
            oil object.  The list is compiled from the stored kinematic
            and dynamic viscosities associated with the oil record.
            The viscosity fields contain:
              - kinematic viscosity in m^2/sec
              - reference temperature in degrees kelvin
              - weathering ???
            Viscosity entries are ordered by (weathering, temperature)
            If we are using dynamic viscosities, we calculate the
            kinematic viscosity from the density that is closest
            to the respective reference temperature
        '''
        # first we get the kinematic viscosities if they exist
        ret = []
        if self._r_oil.kvis:
            ret = [(k.meters_squared_per_sec,
                    k.ref_temp,
                    (0.0 if k.weathering == None else k.weathering))
                    for k in self._r_oil.kvis]

        if self._r_oil.dvis:
            # If we have any DVis records, we need to get the
            # dynamic viscosities, convert to kinematic, and
            # add them if possible.
            # We have dvis at a certain (temperature, weathering).
            # We need to get density at the same weathering and
            # the closest temperature in order to calculate the kinematic.
            # There are lots of oil entries where the dvis do not have
            # matching densities for (temp, weathering)
            densities = [(d.kg_per_m_cubed,
                          d.ref_temp,
                          (0.0 if d.weathering == None else d.weathering))
                         for d in self._r_oil.densities]

            for v, t, w in [(d.kg_per_msec, d.ref_temp, d.weathering)
                            for d in self._r_oil.dvis]:
                if w == None:
                    w = 0.0

                # if we already have a KVis at the same
                # (temperature, weathering), we do not need
                # another one
                if len([vv for vv in ret
                        if vv[1] == t and vv[2] == w]) > 0:
                    continue

                # grab the densities with matching weathering
                dlist = [(d[0], abs(t - d[1]))
                         for d in densities
                         if d[2] == w]

                if len(dlist) == 0:
                    continue

                # grab the density with the closest temperature
                density = sorted(dlist, key=lambda x: x[1])[0][0]

                # kvis = dvis/density
                ret.append(((v / density), t, w))

        ret.sort(key=lambda x: (x[2], x[1]))
        kwargs = ['(m^2/s)', 'Ref Temp (K)', 'Weathering']

        # caution: although we will have a list of real
        #          KVis objects, they are not persisted
        #          in the database.
        ret = [(KVis(**dict(zip(kwargs, v)))) for v in ret]
        return ret

    def get_viscosity(self, units='m^2/s'):
        '''
        :param units: optional input if output units should be something other
                      than m^2/s
        :return: Kinematic Viscosity at current temperature.
                 Default units: (m^2/s)

        The Oil object has a list of kinematic viscosities at empirically
        measured temperatures.  We need to use the ones closest to our
        current temperature and calculate our viscosity from it.
        '''
        if self.viscosities:
            # Get the one that most closely matches our current temperature
            visc = sorted([(v, abs(v.ref_temp - self.temperature))
                            for v in self.viscosities],
                           key=lambda v: v[1])[0][0]
            v_ref = visc.meters_squared_per_sec
            t_ref = visc.ref_temp
            k_v2 = 5000.0

            #print 'temperature =', self.temperature
            #print '(v_ref, t_ref, k_v2) =', (v_ref, t_ref, k_v2)
            if (self.temperature - t_ref) == 0:
                v_0 = v_ref
            else:
                v_0 = v_ref * exp(k_v2 / self.temperature - k_v2 / t_ref)

            return uc.convert('Kinematic Viscosity', 'm^2/s', units, v_0)
        else:
            return None
