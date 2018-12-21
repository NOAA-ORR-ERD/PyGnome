"""
gnome oil object

This provides an Oil object that can be used in the GNOME weathering algorithms.

"""

class Oil:
    """
    Oil object: provides all properties and methods required by the GNOME
    weathering algorithms.

    An oil has a number of properties and methods, and N psuedo components

    For each PC, there are a number of properties, each of those properties
    is an array with len(num_pcs)

    """
    def __init__(self, num_pcs):
        """
        create an empty oil

        :param num_pcs: number of psuedo components
        """
        self.num_pcs = num_pcs

    @property
    def boiling_points(self):
        """
        the boiling points of each pseudo component
        """
        return self._boiling_points

    @property.setter
    def boiling_points(self, bps):
        if len(bps) != self.num_pcs:
            raise ValueError


    @lru_cache(2)
    def vapor_pressure(self, temp, atmos_pressure=101325.0):
        '''
        water_temp and boiling point units are Kelvin
        returns the vapor_pressure in SI units (Pascals)

        ""
        '''
        D_Zb = 0.97
        R_cal = 1.987  # calories

        D_S = 8.75 + 1.987 * np.log(self.boiling_points)
        C_2i = 0.19 * self.boiling_points - 18

        var = 1. / (self.boiling_point - C_2i) - 1. / (temp - C_2i)
        ln_Pi_Po = (D_S * (self.boiling_points - C_2i) ** 2 /
                    (D_Zb * R_cal * self.boiling_points) * var)
        Pi = np.exp(ln_Pi_Po) * atmos_pressure

        return Pi


