
import numpy as np


class Riazi(object):
    '''
        Using Riazi (2005), aromatic properties are approximated
        from distillation cut temperature (degrees Kelvin)
    '''
    @classmethod
    def mol_wt(cls, temp):
        '''
           returns molecular weight in kg/kmole
        '''
        return 350.0 * (6.98 - np.log(1070.0 - temp)) ** (3.0 / 2.0)

    @classmethod
    def density(cls, temp):
        '''
           returns density in kg/m^3
        '''
        return 100.0 * temp ** (1.0 / 3.0)

    @classmethod
    def molar_volume(cls, temp):
        '''
            returns molar_volume in m^3/kmole
        '''
        return cls.mol_wt(temp) / cls.density(temp)
