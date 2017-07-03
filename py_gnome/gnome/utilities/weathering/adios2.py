
import numpy as np

from gnome import constants

g = constants.gravity  # the gravitational constant.


class Adios2(object):
    '''
        This is for any Adios 2 algorithms for which a scientific reference
        is not documented.

        Note: We should really try to look up the references for these and
              move them to an appropriately referenced class.
    '''
    @staticmethod
    def wave_height(U, fetch):
        """
        compute the wave height

        :param U: wind speed
        :type U: floating point number in m/s units

        :returns Hrms: RMS wave height in meters
        """

        # wind stress factor
        # Transition at U = 4.433049525859078 for linear scale with wind speed.
        #   4.433049525859078 is where the solutions match
        ws = np.where(U < 4.433049525859078, 0.71 * U ** 1.23, U)
#         ws = 0.71 * U ** 1.23 if U < 4.433049525859078 else U

        # (2268 * ws ** 2) is limit of fetch limited case.
        if fetch is None:
            H = 0.243 * ws * ws / g
        else:
            H = np.where(fetch < 2268 * ws ** 2,
                         0.0016 * np.sqrt(fetch / g) * ws,
                         0.243 * ws * ws / g)

        Hrms = 0.707 * H

        # arbitrary limit at 30 m -- about the largest waves recorded
        # fixme -- this really depends on water depth -- should take that
        #          into account?
        return np.clip(H, None, 30.0)

    @staticmethod
    def wind_speed_from_height(H):
        """
        Compute the wind speed to use for the whitecap fraction
        This is the reverse of wave_height()
        - Used if the wave height is specified.
        - Unlimited fetch is assumed:

        :param H: given wave height.
        """
        # U_h = 2.0286 * g * sqrt(H / g) # Bill's version
        U_h = np.sqrt(g * H / 0.243)

        if U_h < 4.433049525859078:  # check if low wind case
            U_h = (U_h / 0.71) ** 0.813008

        return U_h

    @staticmethod
    def mean_wave_period(U, wave_height, fetch):
        """
        Compute the mean wave period

        fixme: check for discontinuity at large fetch..
               Is this s bit low??? 32 m/s -> T=15.7 s
        """
        if wave_height is None:
            ws = U * 0.71 * U ** 1.23  # fixme -- linear for large windspeed?

            if fetch is None:
                T = 0.83 * ws
            else:
                T = np.where(fetch >= 2268* ws ** 2,
                             0.83 * ws,
                             0.06238 * (fetch * ws) ** 0.333333333)
        else:
            # user-specified wave height
            T = 7.508 * np.sqrt(wave_height)
        if not isinstance(T, np.array):
            raise TypeError('wave_height or period is not array')

        return T

    @staticmethod
    def dissipative_wave_energy(water_density, H):
        """
        Compute the dissipative wave energy

        units? --  should be: 1/s^3
                   i.e. energy disspiation rate per m^2
                   so water density is in there -- but something else is up
                   does the constant have units?
        """
        return 0.0034 * water_density * g * H ** 2
