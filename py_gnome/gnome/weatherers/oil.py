"""
gnome oil object

This provides an Oil object that can be used in the GNOME weathering algorithms.

"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from backports.functools_lru_cache import lru_cache

import json
import numpy as np
from scipy.optimize import curve_fit

from colander import (SchemaNode, Int, String, Float, drop)

from gnome.persist.base_schema import ObjTypeSchema


class Density(object):

    def __init__(self, kg_m_3, ref_temp_k, weathering=0):
        self.kg_m_3 = kg_m_3
        self.ref_temp_k = ref_temp_k
        self.weathering = weathering


    def __repr__(self):
        return ('<Density({0.kg_m_3} kg/m^3 at {0.ref_temp_k}K), '
                'w={0.weathering}>'
                .format(self))


class KVis(object):

    def __init__(self, m_2_s, ref_temp_k, weathering=0):
        self.m_2_s = m_2_s
        self.ref_temp_k = ref_temp_k
        self.weathering = weathering

    def __repr__(self):
        return ('<KVis({0.m_2_s} m^2/s at {0.ref_temp_k}K, w={0.weathering})>'
                .format(self))

    def __getitem__(self, item):
        return getattr(self, item)


def density_at_temp(ref_density, ref_temp_k, temp_k, k_rho_t=0.0008):
    '''
        Source: Adios2

        If we have an oil density at a reference temperature,
        then we can estimate what its density might be at
        another temperature.

        NOTE: need a reference for the coefficient of expansion
    '''
    return ref_density / (1.0 - k_rho_t * (ref_temp_k - temp_k))


def vol_expansion_coeff(rho_0, t_0, rho_1, t_1):
    '''
        Calculate the volumetric expansion coefficient of a liquid
        based on a set of two densities and their associated temperatures.
    '''
    if t_0 == t_1:
        k_rho_t = 0.0
    else:
        k_rho_t = (rho_0 - rho_1) / (rho_0 * (t_1 - t_0))

    return k_rho_t


def kvis_at_temp(ref_kvis, ref_temp_k, temp_k, k_v2=2416.0):
    '''
        Source: Adios2

        If we have an oil kinematic viscosity at a reference temperature,
        then we can estimate what its viscosity might be at
        another temperature.

        Note: Bill's most recent viscosity document, and an analysis of the
              multi-KVis oils in our oil library suggest that a value of
              2416.0 (Abu Eishah 1999) would be a good default value for k_v2.
    '''
    return ref_kvis * np.exp(k_v2 / temp_k - k_v2 / ref_temp_k)


class OilSchema(ObjTypeSchema):
    '''schema for Oil object'''
    name = SchemaNode(
        String(), missing=drop, save=True, update=True
    )
    api = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    adios_oil_id = SchemaNode(
        Int(), missing=drop, save=True, update=True
    )
    pour_point = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    flash_point = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    solubility = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    bullwinkle_fraction = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    bullwinkle_time = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    emulsion_water_fraction_max = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    densities = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    density_ref_temps = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    density_weathering = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    kvis = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    kvis_ref_temps = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    kvis_weathering = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    mass_fraction = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    boiling_point = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    molecular_weight = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    component_density = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    sara_type = SchemaNode(
        Int(), missing=drop, save=True, update=True
    )
    num_pcs = SchemaNode(
        Int(), missing=drop, save=True, update=True
    )
    num_components = SchemaNode(
        Int(), missing=drop, save=True, update=True
    )

class Oil(object):
    """
    Oil object: provides all properties and methods required by the GNOME
    weathering algorithms.

    An oil has a number of properties and methods, and N pseudo components

    For each PC, there are a number of properties, each of those properties
    is an array with length number of PCs

    """
    _schema = OilSchema

    def __init__(self,
                 name,
                 # Physical properties
                 api,
                 pour_point,
                 solubility,  # kg/m^3
                 # emulsification properties
                 bullwinkle_fraction,
                 bullwinkle_time,
                 emulsion_water_fraction_max,
                 densities,
                 density_ref_temps,
                 density_weathering,
                 kvis,
                 kvis_ref_temps,
                 kvis_weathering,
                 # PCs:
                 mass_fraction,
                 boiling_point,
                 molecular_weight,
                 component_density,
                 sara_type,
                 flash_point=None,
                 adios_oil_id=None,
                 **kwargs
                 ):
        """
        Create an oil from pseudo component data
        """
        self.num_pcs = len(mass_fraction)
        self.num_components = len(mass_fraction)

        self.api = api
        self.adios_oil_id = adios_oil_id
        self.name = name
        self.densities = densities
        self.density_ref_temps = density_ref_temps
        self.density_weathering = density_weathering
        self.kvis = kvis
        self.kvis_ref_temps = kvis_ref_temps
        self.kvis_weathering = kvis_weathering
        self.bullwinkle_frac = bullwinkle_fraction
        self.emulsion_water_fraction_max = emulsion_water_fraction_max
        self.bull_time = bullwinkle_time
        self._pour_point = pour_point
        self._flash_point = flash_point
        self.solubility = solubility
        self._k_v2 = None

        # set the PC properties
        self._set_pc_values('mass_fraction', mass_fraction)
        self._set_pc_values('molecular_weight', molecular_weight)
        self._set_pc_values('boiling_point', boiling_point)
        self._set_pc_values('component_density', component_density)
        if len(sara_type) == self.num_pcs:
            self.sara_type = sara_type
        else:
            raise ValueError("You must have the same number of sara_type as PCs")

        self._bullwinkle = None
        self._bulltime = None
        # self.product_type = "Refined"

    # @classmethod
    # def from_json(cls, data):
    #     if type(data) in (str, str):
    #         data = json.loads(data)
    #     return cls(**num_pcs)


    def __eq__(self, other):
        '''
        need to explicitly compare __dict__
        check if self.__dict__ == other.__dict__
        '''
        if type(self) != type(other):
            return False

        d1 = self.get_dict()
        d2 = other.get_dict()

        for key, val in d1.items():
            o_val = d2[key]

            if isinstance(val, np.ndarray):
                if np.any(val != o_val):
                    return False
            else:
                if val != o_val:
                    return False

        return True


    def get_dict(self):
        """
        Returns a dictionary representation of this object. Uses the schema to
        determine which attributes are put into the dictionary. No extra
        processing is done to each attribute. They are presented as is.
        """

        data = {'name': self.name,
                       'api': self.api,
                       'adios_oil_id': self.adios_oil_id,
                       'pour_point': self._pour_point,
                       'flash_point': self._flash_point,
                       'solubility': self.solubility,
                       'bullwinkle_fraction': self.bullwinkle,
                       'bullwinkle_time': self.bulltime,
                       'densities': self.densities,
                       'density_ref_temps': self.density_ref_temps,
                       'density_weathering': self.density_weathering,
                       'kvis': self.kvis,
                       'kvis_ref_temps': self.kvis_ref_temps,
                       'kvis_weathering': self.kvis_weathering,
                       'emulsion_water_fraction_max': self.emulsion_water_fraction_max,
                       'mass_fraction': self.mass_fraction.tolist(),
                       'boiling_point': self.boiling_point.tolist(),
                       'molecular_weight': self.molecular_weight.tolist(),
                       'component_density': self.component_density.tolist(),
                       'sara_type': self.sara_type}

        return data


    def to_dict(self, json_=None):
        """
        Returns a dictionary representation of this object. Uses the schema to
        determine which attributes are put into the dictionary. No extra
        processing is done to each attribute. They are presented as is.

        The ``json_`` parameter is ignored in this base class. 'save' is passed
        in when the schema is saving the object. This allows an override of
        this function to do any custom stuff necessary to prepare for saving.
        """

        json_ = super(Oil, self).to_dict(json_=json_)

        data = {'name': self.name,
                'api': self.api,
                'adios_oil_id': self.adios_oil_id,
                # 'pour_point': self.pour_point(),
                'pour_point': self._pour_point,
                'flash_point': self._flash_point,
                'solubility': self.solubility,
                'bullwinkle_fraction': self.bullwinkle,
                'bullwinkle_time': self.bulltime,
                'densities': self.densities,
                'density_ref_temps': self.density_ref_temps,
                'density_weathering': self.density_weathering,
                'kvis': self.kvis,
                'kvis_ref_temps': self.kvis_ref_temps,
                'kvis_weathering': self.kvis_weathering,
                'emulsion_water_fraction_max': self.emulsion_water_fraction_max,
                'mass_fraction': self.mass_fraction.tolist(),
                'boiling_point': self.boiling_point.tolist(),
                'molecular_weight': self.molecular_weight.tolist(),
                'component_density': self.component_density.tolist(),
                'sara_type': self.sara_type}

        data.update(json_)
        return data


    def _set_pc_values(self, prop, values):
        """
        utility that sets a property to each pseudo component

        checks that it's the right size, and converts to an array
        """
        if len(values) != self.num_pcs:
            raise ValueError("must be the same number of {} as there "
                             "are pseudo components".format(prop))
        setattr(self, prop, np.array(values, dtype=np.float64))


    @lru_cache(2)
    def vapor_pressure(self, temp, atmos_pressure=101325.0):
        """
        the vapor pressure on the PCs at a given temperature
        water_temp and boiling point units are Kelvin

        :param temp: temperature in K

        :returns: vapor_pressure array in SI units (Pascals)

        ## Fixme: shouldn't this be in the Evaporation code?
        """
        D_Zb = 0.97
        R_cal = 1.987  # calories

        D_S = 8.75 + R_cal * np.log(self.boiling_point)
        C_2i = 0.19 * self.boiling_point - 18

        var = 1. / (self.boiling_point - C_2i) - 1. / (temp - C_2i)
        ln_Pi_Po = ((D_S * (self.boiling_point - C_2i) ** 2 /
                    (D_Zb * R_cal * self.boiling_point)) * var)
        Pi = np.exp(ln_Pi_Po) * atmos_pressure

        return Pi

    def bounding_temperatures(cls, obj_list, temperature):
        '''
            General Utility Function

            From a list of objects containing a ref_temp_k attribute,
            return the object(s) that are closest to the specified
            temperature(s)
            specifically:
            - we want the ones that immediately bound our temperature.
            - if our temperature is high and out of bounds of the temperatures
              in our obj_list, then we return a range containing only the
              highest temperature.
            - if our temperature is low and out of bounds of the temperatures
              in our obj_list, then we return a range containing only the
              lowest temperature.

            We accept only a scalar temperature or a sequence of temperatures
        '''
        temperature = np.array(temperature).reshape(-1, 1)

        if len(obj_list) <= 1:
            # range where the lowest and highest are basically the same.
            obj_list *= 2

        geq_temps = temperature >= [obj.ref_temp_k for obj in obj_list]

        high_and_oob = np.all(geq_temps, axis=1)
        low_and_oob = np.all(geq_temps ^ True, axis=1)

        rho_idxs0 = np.argmin(geq_temps, axis=1)
        rho_idxs0[rho_idxs0 > 0] -= 1
        rho_idxs0[high_and_oob] = len(obj_list) - 1

        rho_idxs1 = (rho_idxs0 + 1).clip(0, len(obj_list) - 1)
        rho_idxs1[low_and_oob] = 0

        return list(zip([obj_list[i] for i in rho_idxs0],
                   [obj_list[i] for i in rho_idxs1]))


    def get_densities(self):
        '''
            return a list of densities for the oil at a specified state
            of weathering.
            We include the API as a density if:
            - the specified weathering is 0
            - the culled list of densities does not contain a measurement
              at 15C
        '''
        densities = [d for d in self.densities]
        density_ref_temps = [d for d in self.density_ref_temps]
        weathering = [d for d in self.density_weathering]
        new_densities = []

        for x in range(0,len(densities)):
            if weathering[x] == 0:	#also check None
                new_densities.append(Density(kg_m_3=densities[x],ref_temp_k=density_ref_temps[x],weathering=0.0))

        return sorted(new_densities, key=lambda d: d.ref_temp_k)

    def density_at_temp(self, temperature=288.15):
        '''
            Get the oil density at a temperature or temperatures.

            Note: there is a catch-22 which prevents us from getting
                  the min_temp in all casees:
                  - to estimate pour point, we need viscosities
                  - if we need to convert dynamic viscosities to
                    kinematic, we need density at 15C
                  - to estimate density at temp, we need to estimate pour point
                  - ...and then we recurse
                  For this case we need to make an exception.
            Note: If we have a pour point that is higher than one or more
                  of our reference temperatures, then the lowest reference
                  temperature will become our minimum temperature.

            TODO: We are getting rid of the argument that specifies a
                  weathering amount because it is currently implemented
                  in an unusably precise manner.  Robert would like us to
                  implement a means of interpolating density using a
                  combination of (temperature, weathering).  But the algorithm
                  for this is not defined at the moment.
        '''
        shape = None
        densities = [d for d in self.get_densities()
                     if np.isclose(d.weathering, 0.0)]

        # set the minimum temperature to be the oil's pour point
        min_temp = np.min([d.ref_temp_k for d in densities] +
                          [pp for pp in self.pour_point()[:2]
                           if pp is not None])

        if hasattr(temperature, '__iter__'):
            temperature = np.clip(temperature, min_temp, 1000.0)
            shape = temperature.shape
            temperature = temperature.reshape(-1)
        else:
            temperature = min_temp if temperature < min_temp else temperature

        ref_density, ref_temp_k = self._get_reference_densities(densities,
                                                                temperature)
        k_rho_t = self._vol_expansion_coeff(densities, temperature)

        rho_t = density_at_temp(ref_density, ref_temp_k,
                                    temperature, k_rho_t)

        if len(rho_t) == 1:
            return rho_t[0]
        elif shape is not None:
            return rho_t.reshape(shape)
        else:
            return rho_t

    @property
    def standard_density(self):
        '''
            Standard density is simply the density at 15C, which is the
            default temperature for density_at_temp()
        '''
        return self.density_at_temp()

    def _get_reference_densities(self, densities, temperature):
        '''
            Given a temperature, we return the best measured density,
            and its reference temperature, to be used in calculation.

            For our purposes, it is the density closest to the given
            temperature.
        '''
        closest_densities = self.bounding_temperatures(densities, temperature)

        try:
            # sequence of ranges
            density_values = np.array([[d.kg_m_3 for d in r]
                                       for r in closest_densities])
            ref_temp_values = np.array([[d.ref_temp_k for d in r]
                                        for r in closest_densities])

            greater_than = np.all((temperature > ref_temp_values.T).T, axis=1)

            density_values[greater_than, 0] = density_values[greater_than, 1]
            ref_temp_values[greater_than, 0] = ref_temp_values[greater_than, 1]

            return density_values[:, 0], ref_temp_values[:, 0]
        except TypeError:
            # single range
            density_values = np.array([d.kg_m_3 for d in closest_densities])
            ref_temp_values = np.array([d.ref_temp_k
                                        for d in closest_densities])

            if np.all(temperature > ref_temp_values):
                return density_values[1], ref_temp_values[1]
            else:
                return density_values[0], ref_temp_values[0]

    def _vol_expansion_coeff(self, densities, temperature):
        closest_densities = self.bounding_temperatures(densities, temperature)

        temperature = np.array(temperature)
        closest_values = np.array([[(d.kg_m_3, d.ref_temp_k)
                                    for d in r]
                                   for r in closest_densities])

        args_list = [[t for d in v for t in d]
                     for v in closest_values]
        k_rho_t = np.array([vol_expansion_coeff(*args)
                            for args in args_list])

        greater_than = np.all((temperature > closest_values[:, :, 1].T).T,
                              axis=1)
        less_than = np.all((temperature < closest_values[:, :, 1].T).T,
                           axis=1)

        if self.api > 30:
            k_rho_default = 0.0009
        else:
            k_rho_default = 0.0008

        k_rho_t[greater_than | less_than] = k_rho_default

        if k_rho_t.shape[0] == 1:
            return k_rho_t[0]
        else:
            return k_rho_t


    def closest_to_temperature(cls, obj_list, temperature, num=1):
        '''
            General Utility Function

            From a list of objects containing a ref_temp_k attribute,
            return the object(s) that are closest to the specified
            temperature(s)

            We accept only a scalar temperature or a sequence of temperatures
        '''
        if hasattr(temperature, '__iter__'):
            # we like to deal with numpy arrays as opposed to simple iterables
            temperature = np.array(temperature)

        # our requested number of objs can have a range [0 ... listsize-1]
        if num >= len(obj_list):
            num = len(obj_list) - 1

        temp_diffs = np.array([abs(obj.ref_temp_k - temperature)
                               for obj in obj_list]).T

        if len(obj_list) <= 1:
            return obj_list
        else:
            # we probably don't really need this for such a short list,
            # but we use a numpy 'introselect' partial sort method for speed
            try:
                # temp_diffs for sequence of temps
                closest_idx = np.argpartition(temp_diffs, num)[:, :num]
            except IndexError:
                # temp_diffs for single temp
                closest_idx = np.argpartition(temp_diffs, num)[:num]

            try:
                # sequence of temperatures result
                closest = [sorted([obj_list[i] for i in r],
                                  key=lambda x: x.ref_temp_k)
                           for r in closest_idx]
            except TypeError:
                # single temperature result
                closest = sorted([obj_list[i] for i in closest_idx],
                                 key=lambda x: x.ref_temp_k)

            return closest


    def aggregate_kvis(self):
        kvis_list = [((k.ref_temp_k, k.weathering), (k.m_2_s, False))
                     for k in self.culled_kvis()]

        agg = dict(kvis_list)

        return list(zip(*[(KVis(m_2_s=k, ref_temp_k=t, weathering=w), e)
                     for (t, w), (k, e) in sorted(agg.items())]))


    def kvis_at_temp(self, temp_k=288.15, weathering=0.0):
        shape = None

        if hasattr(temp_k, '__iter__'):
            # we like to deal with numpy arrays as opposed to simple iterables
            temp_k = np.array(temp_k)
            shape = temp_k.shape
            temp_k = temp_k.reshape(-1)

        kvis = [d for d in self.kvis]
        kvis_ref_temps = [d for d in self.kvis_ref_temps]
        weathering = [d for d in self.kvis_weathering]
        kvis_list = []

        for x in range(0,len(kvis)):
            if weathering[x] == 0:	#also check None
                kvis_list.append(KVis(m_2_s=kvis[x],ref_temp_k=kvis_ref_temps[x],weathering=0.0))

       # agg = dict(kvis_list)

        #new_kvis_list = zip(*[(KVis(m_2_s=k, ref_temp_k=t, weathering=w), e)
        #             for (t, w), (k, e) in sorted(agg.iteritems())])
        #kvis_list = [kv for kv in self.aggregate_kvis()[0]
                     #if (kv.weathering == weathering)]
        closest_kvis = self.closest_to_temperature(kvis_list, temp_k)

        if closest_kvis is not None:
            try:
                # treat as a list
                ref_kvis, ref_temp_k = list(zip(*[(kv[0].m_2_s, kv[0].ref_temp_k)
                                             for kv in closest_kvis]))
                if len(closest_kvis) > 1:
                    ref_kvis = np.array(ref_kvis).reshape(temp_k.shape)
                    ref_temp_k = np.array(ref_temp_k).reshape(temp_k.shape)
                else:
                    ref_kvis, ref_temp_k = ref_kvis[0], ref_temp_k[0]
            except TypeError:
                # treat as a scalar
                ref_kvis, ref_temp_k = (closest_kvis[0].m_2_s,
                                        closest_kvis[0].ref_temp_k)
        else:
            return None

        if self._k_v2 is None:
            self.determine_k_v2(kvis_list)

        kvis_t = kvis_at_temp(ref_kvis, ref_temp_k, temp_k, self._k_v2)

        if shape is not None:
            return kvis_t.reshape(shape)
        else:
            return kvis_t

    def determine_k_v2(self, kvis_list=None):
      # FIXME: this should be able to be done with a simple linear fit.
        '''
            The value k_v2 is the coefficient of exponential decay used
            when calculating kinematic viscosity as a function of
            temperature.
            - If the oil contains two or more viscosity measurements, then
              we will make an attempt at determining k_v2 using a least
              squares fit.
            - Otherwise we will need to choose a reasonable average default
              value.  Bill's most recent viscosity document, and an
              analysis of the multi-KVis oils in our oil library suggest that
              a value of 2416.0 (Abu Eishah 1999) would be a good default
              value.
        '''
        self._k_v2 = 2416.0

        def exp_func(temp_k, a, k_v2):
            return a * np.exp(k_v2 / temp_k)

        if kvis_list is None:
            kvis_list = [kv for kv in self.aggregate_kvis()[0]
                         if (kv.weathering in (None, 0.0))]

        if len(kvis_list) < 2:
            return

        ref_temp_k, ref_kvis = zip(*((k.ref_temp_k, k.m_2_s)
                                     for k in kvis_list))

        for k in np.logspace(3.6, 4.5, num=8):
            # k = log range from about 5000-32000
            a_coeff = ref_kvis[0] * np.exp(-k / ref_temp_k[0])

            try:
                popt, pcov = curve_fit(exp_func, ref_temp_k, ref_kvis,
                                       p0=(a_coeff, k), maxfev=2000)

                # - we want our covariance to be a reasonably small number,
                #   but it can get into the thousands even for a good fit.
                #   So we will only check for inf values.
                # - for sample sizes < 3, the covariance is unreliable.
                if len(ref_kvis) > 2 and np.any(pcov == np.inf):
                    print('covariance too high.')
                    continue

                if popt[1] <= 1.0:
                    continue

                self._k_v2 = popt[1]
                break
            except (ValueError, RuntimeError):
                continue


    def get(self, prop):
        'get oil props'
        val = None
        try:
            val = getattr(self, prop)
        except AttributeError:
            try:
                val = getattr(self, prop)
            except Exception:
                pass

        return val

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
            if self.bull_time is not None:
                return self.bull_time
            else:
                return bulltime
            #return bulltime

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
            #return self.get('bullwinkle_fraction')
            return self.bullwinkle_fraction()

    @bullwinkle.setter
    def bullwinkle(self, value):
        """
        emulsion constant
        """
        self._bullwinkle = value

    def bullwinkle_time(self):
        if self.bull_time is not None:
            bullwinkle_time = self.bull_time
        return bullwinkle_time

    def bullwinkle_fraction(self):
        """
        require bullwinkle in the data, do not estimate
        """
        if self.bullwinkle_frac is not None:
            bullwinkle_fraction = self.bullwinkle_frac
        return bullwinkle_fraction
        #return self._adios2_bullwinkle_fraction()

    def pour_point(self):

        #return self.pour_point_min_k, self.pour_point_max_k, Estimated
        return self._pour_point, self._pour_point, False

    def flash_point(self):

        #return self.flash_point_min_k, self.flash_point_max_k, Estimated
        return self._flash_point, self._flash_point, False
