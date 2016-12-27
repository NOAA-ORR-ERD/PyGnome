
class Monahan(object):
    '''
        This based on formulas by:
        Monahan (JPO, 1971)
    '''
    @staticmethod
    def whitecap_decay_constant(salinity):
        """
            Monahan(JPO, 1971) time constant characterizing exponential
            whitecap decay.

            :param salinity: the salinity of the water
            :type salinity: integer or float in PSU


            The saltwater value for this constant is 3.85 sec while the
            freshwater value is 2.54 sec.

            This interpolates between those values by salinity
        """
        return 0.03742857 * salinity + 2.54
