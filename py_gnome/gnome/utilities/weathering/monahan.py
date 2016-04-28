
class Monahan(object):
    '''
        This based on formulas by:
        Monahan (JPO, 1971)
    '''
    @classmethod
    def whitecap_decay_constant(cls, salinity):
        """
            Monahan(JPO, 1971) time constant characterizing exponential
            whitecap decay.

            The saltwater value for this constant is 3.85 sec while the
            freshwater value is 2.54 sec.

            We will interpolate with salinity
        """
        return 0.03742857 * salinity + 2.54
