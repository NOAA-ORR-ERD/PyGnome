import numpy as np


class Overstreet(object):
    '''
        This based on formulas by Roy Overstreet, one of the founding members
        of ERD.
    '''
    @classmethod
    def roys_constant(cls, emulsion_kvis):
        """
            Roy's Constant (C_disp):
                This constant represents the reduction in dispersion
                as the oil viscosity increases.

            :param emulsion_kvis: the emulsion kinematic viscosity (m^2/s)
        """
        return 2400.0 * np.exp(-73.682 * np.sqrt(emulsion_kvis))
