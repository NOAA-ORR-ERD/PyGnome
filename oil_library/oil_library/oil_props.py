'''
OilProps class which serves as a wrapper around
gnome.db.oil_library.models.Oil class

It also contains a dict containing a small number of sample_oils if user does
not wish to query database to use an _r_oil.

It contains a function that takes a string for 'oil_name' and returns and Oil
object used to initialize and OilProps object

Not sure at present if this needs to be serializable?
'''
from math import exp, log   # for scalars, python is faster

from hazpy import unit_conversion
uc = unit_conversion

from .models import KVis
from .utilities import get_density


def boiling_point(max_cuts, api):
    '''
    return an array of boiling points for each psuedo-components
    Assume max_cuts * 2 components containing [saturate, aromatic]
    Output boiling point in this form:

      components: [s_0, a_0, s_1, a_1, ..., s_n, a_n]
      index, i:   [0, 1, 2, .., max_cuts-1]

    where s_i is boiling point corresponding with i-th saturate component
    similarly a_i is boiling point corresponding with i-th aromatic component
    Hence, max_cuts * 2 components. Boiling point is computed assuming a linear
    relation as follows:

        T_o = 457.16 - 3.3447 * API
        dT/df = 1356.7 - 247.36*ln(API)

    The linear fit is done for evenly spaced intervals and BP is in ascending
    order

        if i % 2 == 0:    # for saturates, i is even
            bp[i] = T_o + dT/df * (i + 1)/(max_cuts * 2)
            so i = [0, 2, 4, .., max_cuts-2]

    The boiling point for i-th component's saturate == aromatic bp:

        if i % 2 == 1:    # aromatic, i is odd
            bp[i] = T_o + dT/df * i/(max_cuts * 2)

    boiling point units of Kelvin
    Boiling point of saturate and aromatic i-th mass component is equal.
    '''
    T_o = 457.16 - 3.3447 * api
    dT_dF = 1356.7 - 247.36 * log(api)

    bp = [float('nan')] * (max_cuts * 2)
    array_size = (max_cuts * 2)
    for ix in range(0, max_cuts * 2, 2):
        bp[ix] = dT_dF*(ix + 1)/array_size + T_o
        bp[ix + 1] = bp[ix]

    return bp


def molecular_weight(bp, component):
    '''
    return the molecular weight of the pseudocomponents (mw_i) given the
    boiling points. It returns the mw_i for saturates and aromatic components
    '''
    if component == 'saturate':
        mw = (0.04132 - 1.985e-4 * bp + (9.494e-7 * bp ** 2))
    elif component == 'aromatic':
        mw = (0.04132 - 1.985e-4 * bp + (9.494e-7 * bp ** 2))

    return mw


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

    def __init__(self, oil_):
        '''
        Extends the raw Oil object to include properties required by
        weathering processes. If oil_ is not pulled from database or user may
        wish to use simple half life weatherer, in this case, there is no need
        to carry around more than one psuedo-component. Let user set max_cuts
        if desired, but only during initialization.

        :param oil_: Oil object that maps to entity in OilLib database
        :type oil_: Oil object
        '''
        self._r_oil = oil_

        # Default format for mass components:
        # mass_fraction =
        # [m0_s, m0_a, m1_s, m1_a, ..., m_resins, m_asphaltenes]
        #self._component_frac_bp()
        #self._component_mw()

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'oil_={0._r_oil!r}, water_temp={0.temperature}'
                ')'.format(self))

    #density = property(lambda self: self.get_density())
    #viscosity = property(lambda self: self.get_viscosity())
    name = property(lambda self: self._r_oil.oil_name,
                    lambda self, val: setattr(self._r_oil, 'oil_name', val))
    api = property(lambda self: self.get('api'))

    def get(self, prop):
        'get raw oil props'
        return getattr(self._r_oil, prop)

    def get_density(self, temp, out=None):
        '''
        return density at a temperature
        do we want to do any unit conversions here?

        :param temp: temperature in Kelvin. Could be an ndarray, list or scalar
        :type temp: scalar, list, tuple or ndarray - assumes it is in Kelvin
        '''
        return get_density(self._r_oil, temp, out)

 #=============================================================================
 # 
 #    @property
 #    def num_components(self):
 #        return len(self.mass_fraction)
 # 
 #    def _add_resins_asphalt(self, heavy_comp):
 #        'add heavier components if mass_fraction < 1.0'
 #        if heavy_comp is None or heavy_comp == 0.0:
 #            return
 # 
 #        f_remain = sum(self.mass_fraction)
 #        if f_remain < 1.0:
 #            if heavy_comp + f_remain <= 1.0:
 #                self.mass_fraction.append(heavy_comp)
 #            else:
 #                self.mass_fraction.append(1.0-f_remain)
 #            self.boiling_point.append(float('inf'))
 # 
 #    def _frac_bp_from_cuts(self):
 #        '''
 #        Need the mass_fraction to sum upto 1.0
 #        Also need to understand how to identify saturates/aromatics
 #        Should mass_fractions be determined by cuts (boiling_points) or
 #        should they be given by largest to smallest mass fraction?
 #        '''
 #        # distillation cut data available
 #        last_frac = 0.0
 #        for ix, cut in enumerate(self._r_oil.cuts):
 #            if ix < (self.max_cuts * 2):
 #                self.boiling_point.append(cut.vapor_temp_k)
 #                self.mass_fraction.append(cut.fraction - last_frac)
 #                last_frac = cut.fraction
 #        self._add_resins_asphalt(self.get('resins'))
 #        self._add_resins_asphalt(self.get('asphaltene_content'))
 # 
 #    def _frac_bp_estimated(self):
 #        'no distillation cuts data available'
 #        resins = 0.0
 #        asphalt = 0.0
 #        if self.get('resins') is not None:
 #            resins = self.get('resins')
 #        if self.get('asphaltene_content') is not None:
 #            asphalt = self.get('asphaltene_content')
 # 
 #        mass_left = 1.0 - resins - asphalt
 #        mass_per_comp = mass_left / (self.max_cuts * 2)
 #        self.mass_fraction = [mass_per_comp] * (self.max_cuts * 2)
 #        if self.api is not None:
 #            self.boiling_point = boiling_point(self.max_cuts,
 #                                               self.get('api'))
 #        else:
 #            self.boiling_point = [float('nan')] * (self.max_cuts * 2)
 # 
 #        if resins > 0.0:
 #            self.mass_fraction.append(resins)
 #            self.boiling_point.append(float('inf'))
 # 
 #        if asphalt > 0.0:
 #            self.mass_fraction.append(asphalt)
 #            self.boiling_point.append(float('inf'))
 # 
 #    def _component_frac_bp(self):
 #        '''
 #        if number of psuedo components changes, update related properties
 #        self.mass_fraction defined as:
 #        [m0_s, m0_a, m1_s, m1_a, ..., m_resins, m_asphaltenes]
 #        '''
 #        self.mass_fraction = []
 #        self.boiling_point = []
 #        if len(self._r_oil.cuts) > 0:
 #            self._frac_bp_from_cuts()
 #        else:
 #            # no distillation cut data
 #            self._frac_bp_estimated()
 # 
 #    def _component_mw(self):
 #        'estimate molecular weights of components'
 #        self.molecular_weight = [float('nan')] * self.num_components  # initialize to 'nan'
 #        for ix, bp in enumerate(self.boiling_point):
 #            if bp is 'inf' or bp is 'nan':
 #                continue
 # 
 #            if ix % 2 == 0:
 #                self.molecular_weight[ix] = molecular_weight(bp, 'saturate')
 #            else:
 #                # will define a different function for mw_aromatics
 #                self.molecular_weight[ix] = molecular_weight(bp, 'aromatic')
 #=============================================================================
