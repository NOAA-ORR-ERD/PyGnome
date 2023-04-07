
import numpy as np


# class Toluene(object):
#     '''
#         The measured values of the known aromatic, toluene
#     '''
#     mol_wt = 92.1
#     density = 866.0
#     k_ow = 1000.0
#

class BanerjeeHuibers(object):
    '''
        The combination of correlations by Huibers and Katritzky (1998)
        and Banerjee et al (1980) to estimate the correlation between
        a specific aromatic hydrocarbon's density and molecular weight
        with its partition coefficient.

            rho_arom = density of aromatic

            mol_wt = molecular weight

            s = solubility

            s = Huibers(rho_arom, mol_wt)

            k_ow = Banerjee(s)

                 = Banerjee(Huibers(rho_arom, mol_wt))

    '''

    @classmethod
    def partition_coeff(cls, mol_wt, density):
        '''
            :param mol_wt: Molecular weight in kg/kmole
            :param density: Density in kg/m^3
        '''
        # In Banerjee MV is in mL/mole so we need to convert
        return 1.8 * np.exp(0.059 * 1000. * mol_wt / density)