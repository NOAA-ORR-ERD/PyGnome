'''
OilProps class which serves as a wrapper around
gnome.db.oil_library.models.Oil class

It also contains a dict containing a small number of sample_oils if user does
not wish to query database to use an _r_oil.

It contains a function that takes a string for 'name' and returns and Oil
object used to initialize and OilProps object

Not sure at present if this needs to be serializable?
'''
import copy
from itertools import groupby, chain, izip_longest

from repoze.lru import lru_cache
import numpy as np

import unit_conversion as uc
from .utilities import get_density, get_viscosity
from .models import Oil


# create a dtype for storing sara information in numpy array
sara_dtype = np.dtype([('type', 'S16'),
                       ('boiling_point', np.float64),
                       ('fraction', np.float64),
                       ('density', np.float64),
                       ('mol_wt', np.float64)])


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
        #     [m0_s, m0_a, m1_s, m1_a, ..., m_resins, m_asphaltenes]
        #
        # the boiling points are in ascending order
        self._init_sara()

        self._bullwinkle = None
        self._bulltime = None

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'oil_={0._r_oil!r})'.format(self))

    name = property(lambda self: self._r_oil.name,
                    lambda self, val: setattr(self._r_oil, 'name', val))
    api = property(lambda self: self.get('api'))

    def get(self, prop):
        'get raw oil props'
        val = None
        try:
            val = getattr(self._r_oil, prop)
        except AttributeError:
            try:
                val = getattr(self._r_oil.imported, prop)
            except:
                pass

        return val

    @lru_cache(2)
    def get_density(self, temp=None, out=None):
        '''
        return density at a temperature
        do we want to do any unit conversions here?
        todo: memoize function

        :param temp: temperature in Kelvin. Could be an ndarray, list or scalar
        :type temp: scalar, list, tuple or ndarray - assumes it is in Kelvin
        '''
        if temp:
            return get_density(self._r_oil, temp, out)
        else:
            return uc.convert('density', 'API', 'kg/m^3', self.api)

    @lru_cache(2)
    def get_viscosity(self, temp=288.15, out=None):
        '''
        return viscosity at a temperature, default is viscosity at 15degC
        todo: memoize function

        :param temp: temperature in Kelvin. Could be an ndarray, list or scalar
        :type temp: scalar, list, tuple or ndarray - assumes it is in Kelvin
        '''
        return get_viscosity(self._r_oil, temp, out)

    @property
    def bulltime(self):
        '''
        return bulltime (time to emulsify)
        either user set or just return a flag
        '''
        # check for user input value, otherwise set to -999 as a flag
        bulltime = -999.

        if self._bulltime is not None:
            return self._bulltime
        else:
            return bulltime

    @bulltime.setter
    def bulltime(self, value):
        """
        time to start emulsification
        """
        self._bulltime = value

    @property
    def bullwinkle(self):
        '''
        return bullwinkle (emulsion constant)
        either user set or return database value
        '''
        # check for user input value, otherwise return database value

        if self._bullwinkle is not None:
            return self._bullwinkle
        else:
            return self.get('bullwinkle_fraction')

    @bullwinkle.setter
    def bullwinkle(self, value):
        """
        emulsion constant
        """
        self._bullwinkle = value

    @property
    def num_components(self):
        '''
        number of components with mass fraction > 0.0 used to model the oil
        '''
        return len(self._sara)

    @property
    def molecular_weight(self):
        return self._component_mw()

    def _component_mw(self, sara_type=None):
        '''
        return the molecular weight of the pseudocomponents
        '''
        ret = self._sara['mol_wt']

        if sara_type is not None:
            ret = ret[np.where(self._sara['type'] == sara_type)]

        return ret

    @property
    def mass_fraction(self):
        return self._sara['fraction']

    @property
    def boiling_point(self):
        return self._sara['boiling_point']

    @property
    def component_density(self):
        return self._component_density()

    def _component_density(self, sara_type=None):
        '''
        return the density of the pseudocomponents.
        '''
        ret = self._sara['density']

        if sara_type is not None:
            ret = ret[np.where(self._sara['type'] == sara_type)]

        return ret

    @lru_cache(2)
    def vapor_pressure(self, temp, atmos_pressure=101325.0):
        '''
        water_temp and boiling point units are Kelvin
        returns the vapor_pressure in SI units (Pascals)
        todo: memoize function
        '''
        D_Zb = 0.97
        R_cal = 1.987  # calories

        D_S = 8.75 + 1.987 * np.log(self.boiling_point)
        C_2i = 0.19 * self.boiling_point - 18

        var = 1. / (self.boiling_point - C_2i) - 1. / (temp - C_2i)
        ln_Pi_Po = (D_S * (self.boiling_point - C_2i) ** 2 /
                    (D_Zb * R_cal * self.boiling_point) * var)
        Pi = np.exp(ln_Pi_Po) * atmos_pressure

        return Pi

    def tojson(self):
        '''
            For now, just convert underlying oil object tojson() method
            - An Oil object that has been queried from the database
              contains a lot of unnecessary relationships that we do not
              want to represent in our JSON output,
              So we prune them by first constructing an Oil object from the
              JSON payload of the queried Oil object.
              This creates an Oil object in memory that does not have any
              database links.
              Then we output the JSON from the unlinked object.
        '''

        return Oil.from_json(self._r_oil.tojson()).tojson()

    def _compare__dict(self, other):
        '''
        cannot just do self.__dict__ == other.__dict__ since
        '''
        for key, val in self.__dict__.iteritems():
            o_val = other.__dict__[key]
            if isinstance(val, np.ndarray):
                if np.any(val != o_val):
                    return False
            else:
                if val != o_val:
                    return False
        return True

    def __eq__(self, other):
        '''
        need to explicitly compare __dict__
        However, PyGnome initializes two OilProps object when invoked from the
        WebGnomeClient, there is an sql alchemy object embedded in _r_oil
        which maybe different. To avoid comparing the sqlalchemy object that
        is part of the raw oil record, this works as follows:

        1. check if self.__dict__ == other.__dict__
        2. if above fails, then check if the tojson() for both OilProps objects
        match. This assumes that both objects contain tojson()
        '''
        if type(self) != type(other):
            return False

        if self._compare__dict(other):
            return True

        try:
            return self.tojson() == other.tojson()
        except Exception:
            return False

    def __ne__(self, other):
        return not self == other

    def __deepcopy__(self, memo):
        '''
        The _r_oil object should not be copied - it should just be referenced
        to create the OilProps copy. The database record itself does not need
        to be a deepcopy - both OilProps objects can reference the same
        database record
        '''
        c_op = self.__class__(self._r_oil)
        if c_op != self:
            '''
            Attributes are currently derived from _r_oil object. Unless the
            user changes 'mass_fractions', 'boiling_point', 'molecular_weight'
            after initialization, the two objects should be equal
            '''
            for attr in c_op.__dict__:
                if getattr(self, attr) != getattr(c_op, attr):
                    setattr(c_op, attr,
                            copy.deepcopy(getattr(self, attr), memo))
        return c_op

    def _init_sara(self):
        '''
        initialize self._sara as a numpy array. The information is structured
        in increasing boiling points as:
            ['Saturates', boiling_point_0, mass_fraction, density, molWt]
            ['Aromatics', boiling_point_0, mass_fraction, density, molWt]
            ['Saturates', boiling_point_1, mass_fraction, density, molWt]
            ['Aromatics', boiling_point_1, mass_fraction, density, molWt]
            ...
            ['Resins', boiling_point_terminal, mass_fraction, density, molWt]
            ['Asphaltenes', boiling_point_terminal, mass_fraction, density, molWt]

        Omit components that have 0 mass fraction
        '''
        all_comp = list(chain(*[sorted(list(g), key=lambda s: s.sara_type,
                                       reverse=True)
                                for k, g
                                in groupby(sorted(self._r_oil.sara_fractions,
                                                  key=lambda s: s.ref_temp_k),
                                           lambda x: x.ref_temp_k)]
                              ))

        all_dens = list(chain(*[sorted(list(g), key=lambda s: s.sara_type,
                                       reverse=True)
                                for k, g
                                in groupby(sorted(self._r_oil.sara_densities,
                                                  key=lambda s: s.ref_temp_k),
                                           lambda x: x.ref_temp_k)]
                              ))

        all_mw = list(chain(*[sorted(list(g), key=lambda s: s.sara_type,
                                     reverse=True)
                              for k, g
                              in groupby(sorted(self._r_oil.molecular_weights,
                                                key=lambda s: s.ref_temp_k),
                                         lambda x: x.ref_temp_k)]
                            ))

        items = []
        sum_frac = 0.
        for comp, dens, mol_wt in izip_longest(all_comp, all_dens, all_mw):
            if (comp.ref_temp_k != comp.ref_temp_k or
                    comp.sara_type != comp.sara_type):
                msg = "mismatch in sara_fractions and sara_densities tables"
                raise ValueError(msg)

            if comp.fraction > 0.0:
                if hasattr(mol_wt, 'g_mol'):
                    mw = mol_wt.g_mol
                else:
                    # We currently don't have estimation methods for
                    # resin and asphaltene molecular weights, so they
                    # don't exist in the oil record.
                    if comp.sara_type == 'Resins':
                        # recommended avg. value from Bill is 800 g/mol
                        mw = 800.0
                    elif comp.sara_type == 'Asphaltenes':
                        # recommended avg. value from Bill is 1000 g/mol
                        mw = 1000.0

                items.append((comp.sara_type, comp.ref_temp_k, comp.fraction,
                              dens.density, mw))
                sum_frac += comp.fraction

        self._sara = np.asarray(items, dtype=sara_dtype)

        if not np.allclose(self._sara[:]['fraction'].sum(), 1.0):
            msg = ("mass fraction sum: {0} - sum should be approximately 1.0"
                   .format(self._sara[:]['fraction'].sum()))
            raise ValueError(msg)
