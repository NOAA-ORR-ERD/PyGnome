"""
gnome oil object

This provides an Oil object that can be used in the GNOME weathering algorithms.

"""

from backports.functools_lru_cache import lru_cache

import json
import numpy as np


class Oil(object):
    """
    Oil object: provides all properties and methods required by the GNOME
    weathering algorithms.

    An oil has a number of properties and methods, and N pseudo components

    For each PC, there are a number of properties, each of those properties
    is an array with length number of PCs

    """
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
                 kvis,
                 kvis_ref_temp,
                 # PCs:
                 mass_fractions,
                 boiling_points,
                 molecular_weights,
                 component_densities,
                 sara_types,
                 ):
        """
        Create an oil from pseudo component data
        """
        self.num_pcs = len(mass_fractions)

        # set the PC properties
        self._set_pc_values('mass_fractions', mass_fractions)
        self._set_pc_values('molecular_weights', molecular_weights)
        self._set_pc_values('boiling_points', boiling_points)
        self._set_pc_values('component_densities', component_densities)
        if len(sara_types) == self.num_pcs:
            self.sara_types = sara_types
        else:
            raise ValueError("You must have the same number of sara_types as PCs")

    @classmethod
    def from_json(cls, data):
        if type(data) in (str, unicode):
            data = json.loads(data)
        return cls(**num_pcs)


    # def to_dict(self):
    #     return

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

        D_S = 8.75 + R_cal * np.log(self.boiling_points)
        C_2i = 0.19 * self.boiling_points - 18

        var = 1. / (self.boiling_point - C_2i) - 1. / (temp - C_2i)
        ln_Pi_Po = (D_S * (self.boiling_points - C_2i) ** 2 /
                    (D_Zb * R_cal * self.boiling_points) * var)
        Pi = np.exp(ln_Pi_Po) * atmos_pressure

        return Pi


