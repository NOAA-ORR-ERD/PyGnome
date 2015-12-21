
class LehrSimecek(object):
    '''
        This based on formulas by:
        Lehr and Simecek-Beatty
    '''
    @classmethod
    def whitecap_fraction(cls, U, salinity):
        """
        compute the white capping fraction

        This based on Lehr and Simecek-Beatty
        The Relation of Langmuir Circulation Processes to the Standard
        Oil Spill Spreading, Dispersion and Transport Algorithms
        Spill Sci. and Tech. Bull, 6:247-253 (2000)
        (maybe typo -- didn't match)

        Additionally:  Ocean Waves Breaking and Marine Aerosol Fluxes
                       By Stanislaw R. Massel
        """
        # Monahan(JPO, 1971) time constant characterizing exponential
        # whitecap decay.
        # The saltwater value for this constant is 3.85 sec while the
        # freshwater value is 2.54 sec.
        # interpolate with salinity:
        Tm = 0.03742857 * salinity + 2.54

        if U < 4.0:  # m/s
            # linear fit from 0 to the 4m/s value from Ding and Farmer
            # maybe should be a exponential / quadratic fit?
            # or zero less than 3, then a sharp increase to 4m/s?
            fw = (0.0125 * U) / Tm
        else:
            # # Ding and Farmer (JPO 1994)
            # fw = (0.01*U + 0.01) / Tm
            # old ADIOS had a .5 factor - not sure why but we'll keep it
            # for now

            # Ding and Farmer (JPO 1994)
            fw = 0.5 * (0.01 * U + 0.01) / Tm

        return fw if fw <= 1.0 else 1.0  # only with U > 200m/s!
