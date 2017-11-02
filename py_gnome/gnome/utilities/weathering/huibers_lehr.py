
import numpy as np


# class Toluene(object):
#     '''
#         The measured values of the known aromatic, toluene
#     '''
#     mol_wt = 92.1
#     density = 866.0
#     k_ow = 1000.0
# 

class HuibersLehr(object):
    '''
        Using Huibers & Katrisky for solubility.
        Using EPA report (2012), and tweaking by Bill so that results
        better match measured values, to estimate the correlation between
        a specific aromatic hydrocarbon's density and molecular weight
        with its partition coefficient.

            rho_arom = density of aromatic
            mol_wt = molecular weight
            S_w = solubility

            S_w = Huibers(rho_arom, mol_wt)
            k_ow = 5.45 * s**(-.89)  (EPA)
            k_ow = 10 * s**(-.95)  (Lehr)

    '''

    @classmethod
    def partition_coeff(cls, mol_wt, density):
        '''
            :param mol_wt: Molecular weight in kg/kmole
            :param density: Density in kg/m^3
        '''

        S_w = 18.6 * 10 ** (-38 * mol_wt / density)
        K_ow = 10 * (10*S_w) ** (-.89)

        return K_ow