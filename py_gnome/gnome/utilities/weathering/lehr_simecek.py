from monahan import Monahan


class LehrSimecek(object):
    '''
        This based on formulas by:
        Lehr and Simecek-Beatty
    '''
    @staticmethod
    def whitecap_fraction(U, salinity):
        """
        compute the white capping fraction

        This based on Lehr and Simecek-Beatty
        The Relation of Langmuir Circulation Processes to the Standard
        Oil Spill Spreading, Dispersion, and Transport Algorithms
        Spill Sci. and Tech. Bull, 6:247-253 (2000)
        (maybe typo -- didn't match)

        Additionally:  Ocean Waves Breaking and Marine Aerosol Fluxes
                       By Stanislaw R. Massel
        """
        Tm = Monahan.whitecap_decay_constant(salinity)

        if U < 4.0:  # m/s
            # linear fit from 0 to the 4m/s value from Ding and Farmer
            # The Lehr and Simecek-Beatty paper had a different formulation:
            #     fw = 0.025 * (U - 3.0) / Tm
            # that one produces a kink at 4 m/s and negative for U < 1
            fw = (0.0125 * U) / Tm
        else:
            # # Ding and Farmer (JPO 1994)
            # fw = (0.01*U + 0.01) / Tm

            # Ding and Farmer (JPO 1994)
            fw = (0.01 * U + 0.01) / Tm

        fw *= 0.5  # old ADIOS had a .5 factor - not sure why but we'll keep it
                   # for now

        return min(fw, 1.0)  # only with U > 200m/s!
