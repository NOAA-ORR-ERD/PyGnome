'''
OilProps class which serves as a wrapper around
gnome.db.oil_library.models.Oil class

It also contains a dict containing a small number of sample_oils if user does
not wish to query database to use an _r_oil.

It contains a function that takes a string for 'oil_name' and returns and Oil
object used to initialize and OilProps object

Not sure at present if this needs to be serializable?
'''

from .utilities import get_density


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
        self._mass_frac_bp_from_cuts()
        self._component_mw()

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'oil_={0._r_oil!r})'.format(self))

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

    @property
    def num_components(self):
        return len(self.mass_fraction)

    def _add_resins_asphalt(self, heavy_comp):
        '''
        add heavier components if mass_fraction < 1.0
        This just ensures mass_fraction sums to 1.0
        '''
        if heavy_comp is None or heavy_comp == 0.0:
            return

        f_remain = sum(self.mass_fraction)
        if f_remain < 1.0:
            if heavy_comp + f_remain <= 1.0:
                self.mass_fraction.append(heavy_comp)
            else:
                self.mass_fraction.append(1.0-f_remain)
            self.boiling_point.append(float('inf'))

    def _mass_frac_bp_from_cuts(self):
        '''
        Need the mass_fraction to sum upto 1.0
        self.mass_fraction defined as:

            [m0_s, m0_a, m1_s, m1_a, ..., m_resins, m_asphaltenes]

        Also need to understand how to identify saturates/aromatics
        Currently, assumes cuts are added as alternating saturate, then
        aromatic in the list of cuts
        '''
        # distillation cut data available
        self.mass_fraction = []
        self.boiling_point = []

        last_frac = 0.0
        for cut in self._r_oil.cuts:
            self.boiling_point.append(cut.vapor_temp_k)
            self.mass_fraction.append(round(cut.fraction - last_frac, 4))
            last_frac = cut.fraction
        self._add_resins_asphalt(self.get('resins'))
        self._add_resins_asphalt(self.get('asphaltene_content'))

    def _component_mw(self):
        'estimate molecular weights of components'
        self.molecular_weight = [float('nan')] * self.num_components  # initialize to 'nan'
        for ix, bp in enumerate(self.boiling_point):
            if bp is 'inf' or bp is 'nan':
                continue

            if ix % 2 == 0:
                self.molecular_weight[ix] = molecular_weight(bp, 'saturate')
            else:
                # will define a different function for mw_aromatics
                self.molecular_weight[ix] = molecular_weight(bp, 'aromatic')
