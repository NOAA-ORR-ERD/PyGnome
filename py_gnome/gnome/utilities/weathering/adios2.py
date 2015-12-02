
import numpy as np

from gnome import constants

g = constants.gravity  # the gravitational constant.


class Adios2(object):
    '''
        This is for any Adios 2 algorithms for which a scientific reference
        is not documented.
    '''
    @classmethod
    def wave_height(cls, U, fetch):
        """
        compute the wave height

        :param U: wind speed
        :type U: floating point number in m/s units

        :returns Hrms: RMS wave height in meters
        """

        # wind stress factor
        # Transition at U = 4.433049525859078 for linear scale with wind speed.
        #   4.433049525859078 is where the solutions match
        ws = 0.71 * U ** 1.23 if U < 4.433049525859078 else U

        # (2268 * ws ** 2) is limit of fetch limited case.
        if (fetch is not None) and (fetch < 2268 * ws ** 2):
            H = 0.0016 * np.sqrt(fetch / g) * ws
        else:  # fetch unlimited
            H = 0.243 * ws * ws / g

        Hrms = 0.707 * H

        # arbitrary limit at 30 m -- about the largest waves recorded
        # fixme -- this really depends on water depth -- should take that
        #          into account?
        return Hrms if Hrms < 30.0 else 30.0

    @classmethod
    def wind_speed_from_height(cls, H):
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

    @classmethod
    def wave_period(cls, U, wave_height, fetch):
        """
        Compute the mean wave period
        """
        if wave_height is None:
            ws = U * 0.71 * U ** 1.23  # fixme -- linear for large windspeed?

            if (fetch is None) or (fetch >= 2268 * ws ** 2):
                # fetch unlimited
                T = 0.83 * ws
            else:
                # eq 3-34 (SPM?)
                T = 0.06238 * (fetch * ws) ** 0.3333333333
        else:
            # user-specified wave height
            T = 7.508 * np.sqrt(wave_height)

        return T
