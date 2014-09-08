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

    def __init__(self, oil_, temperature=311.15, max_cuts=5):
        '''
        Extends the raw Oil object to include properties required by
        weathering processes. If oil_ is not pulled from database or user may
        wish to use simple half life weatherer, in this case, there is no need
        to carry around more than one psuedo-component. Let user set max_cuts
        if desired, but only during initialization.

        :param oil_: Oil object that maps to entity in OilLib database
        :type oil_: Oil object
        :param temperature: The temperature in 'K'.  Per ASTM, the default is
            38 degrees Celcius (311.15 degrees Kelvin)
        :param max_cuts: max number of distillation cuts. Default is 5, so
            each cut will have a 'saturate' and 'aromatic' component, making
            it 10 components. In addition, there maybe some mass in 'resins'
            and 'asphaltenes' making it 12 components
        '''
        self._r_oil = oil_
        self._temperature = temperature

        # Default mass components:
        #    5 saturates, 5 aromatics, resins, asphaltenes
        #
        # mass_fraction =
        # [m0_s, m0_a, m1_s, m1_a, ..., m_resins, m_asphaltenes]
        self.max_cuts = 5
        self._component_frac_bp()
        self._component_mw()

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'oil_={0._r_oil!r}, water_temp={0.temperature}'
                ')'.format(self))

    density = property(lambda self: self.get_density())
    viscosity = property(lambda self: self.get_viscosity())
    temperature = property(lambda self: self.get_temperature(),
                           lambda self, val: self.set_temperature(val))
    name = property(lambda self: self._r_oil.name,
                    lambda self, val: setattr(self._r_oil, 'name', val))
    api = property(lambda self: self.get('api'))

    def get(self, prop):
        'get raw oil props'
        return getattr(self._r_oil, prop)

    def get_temperature(self, units='K'):
        return uc.convert('Temperature', 'K', units, self._temperature)

    def set_temperature(self, value, units='K'):
        temp = uc.convert('Temperature', units, 'K', value)
        self._temperature = temp

    def get_density(self, units='kg/m^3'):
        '''
            We will prefer to calculate density from the empirical densities
            associated with the oil record.
            If no density values exist, estimate it from API

            :param units: optional input if output units should be something
                          other than kg/m^3
            :return: scalar Density.  Default units: (kg/m^3)
        '''

        if self._r_oil.densities:
            # calculate our density at temperature
            density_rec = sorted([(d, abs(d.ref_temp - self.temperature))
                                  for d in self._r_oil.densities],
                                 key=lambda d: d[1])[0][0]
            d_ref = density_rec.kg_per_m_cubed
            t_ref = density_rec.ref_temp
            k_p1 = 0.008

            d_0 = d_ref / (1 - k_p1 * (t_ref - self.temperature))
        elif self.api != None:
            # calculate our density from api
            d_0 = 141.5 / (131.5 + self.api) * 1000
        else:
            raise ValueError("Oil with name '{0}' does not contain 'api'"
                             " property.".format(self._r_oil.name))

        return uc.convert('Density', 'kg/m^3', units, d_0)

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
                    for k in self._r_oil.kvis
                    if k.ref_temp != None]

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
            # first get our v_max
            k_v2 = 5000.0
            pour_point = (self._r_oil.pour_point_max
                          if self._r_oil.pour_point_max != None
                          else self._r_oil.pour_point_min)
            if pour_point:
                visc = sorted([(v, abs(v.ref_temp - pour_point))
                                for v in self.viscosities
                                if v != None],
                               key=lambda v: v[1])[0][0]
                v_ref = visc.meters_squared_per_sec
                t_ref = visc.ref_temp

                v_max = v_ref * exp(k_v2 / pour_point - k_v2 / t_ref)
            else:
                v_max = None

            # now get our v_0
            visc = sorted([(v, abs(v.ref_temp - self.temperature))
                            for v in self.viscosities],
                           key=lambda v: v[1])[0][0]
            v_ref = visc.meters_squared_per_sec
            t_ref = visc.ref_temp

            if (self.temperature - t_ref) == 0:
                v_0 = v_ref
            else:
                v_0 = v_ref * exp(k_v2 / self.temperature - k_v2 / t_ref)

            if v_max:
                return uc.convert('Kinematic Viscosity', 'm^2/s', units,
                                  v_0 if v_0 <= v_max else v_max)
            else:
                return uc.convert('Kinematic Viscosity', 'm^2/s', units,
                                  v_0)
        else:
            return None

    @property
    def num_components(self):
        return len(self.mass_fraction)

    def _add_resins_asphalt(self, heavy_comp):
        'add heavier components if mass_fraction < 1.0'
        if heavy_comp is None or heavy_comp == 0.0:
            return

        f_remain = sum(self.mass_fraction)
        if f_remain < 1.0:
            if heavy_comp + f_remain <= 1.0:
                self.mass_fraction.append(heavy_comp)
            else:
                self.mass_fraction.append(1.0-f_remain)
            self.boiling_point.append(float('inf'))

    def _frac_bp_from_cuts(self):
        '''
        Need the mass_fraction to sum upto 1.0
        Also need to understand how to identify saturates/aromatics
        Should mass_fractions be determined by cuts (boiling_points) or
        should they be given by largest to smallest mass fraction?
        '''
        # distillation cut data available
        last_frac = 0.0
        for ix, cut in enumerate(self._r_oil.cuts):
            if ix < (self.max_cuts * 2):
                self.boiling_point.append(cut.vapor_temp)
                self.mass_fraction.append(cut.fraction - last_frac)
                last_frac = cut.fraction
        self._add_resins_asphalt(self.get('resins'))
        self._add_resins_asphalt(self.get('asphaltene_content'))

    def _frac_bp_estimated(self):
        'no distillation cuts data available'
        resins = 0.0
        asphalt = 0.0
        if self.get('resins') is not None:
            resins = self.get('resins')
        if self.get('asphaltene_content') is not None:
            asphalt = self.get('asphaltene_content')

        mass_left = 1.0 - resins - asphalt
        mass_per_comp = mass_left / (self.max_cuts * 2)
        self.mass_fraction = [mass_per_comp] * (self.max_cuts * 2)
        if self.api is not None:
            self.boiling_point = boiling_point(self.max_cuts,
                                               self.get('api'))
        else:
            self.boiling_point = [float('nan')] * (self.max_cuts * 2)

        if resins > 0.0:
            self.mass_fraction.append(resins)
            self.boiling_point.append(float('inf'))

        if asphalt > 0.0:
            self.mass_fraction.append(asphalt)
            self.boiling_point.append(float('inf'))

    def _component_frac_bp(self):
        '''
        if number of psuedo components changes, update related properties
        self.mass_fraction defined as:
        [m0_s, m0_a, m1_s, m1_a, ..., m_resins, m_asphaltenes]
        '''
        self.mass_fraction = []
        self.boiling_point = []
        if len(self._r_oil.cuts) > 0:
            self._frac_bp_from_cuts()
        else:
            # no distillation cut data
            self._frac_bp_estimated()

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
