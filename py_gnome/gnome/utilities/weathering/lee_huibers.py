
import numpy as np


class Toluene(object):
    '''
    The measured values of the known aromatic, toluene
    '''
    mol_wt = 92.1
    density = 866.0
    k_ow = 1000.0


class LeeHuibers(object):
    '''
    The combination of correlations by Huibers and Katritzky (1998)
    and Lee et al (1992) to estimate the correlation between
    a specific aromatic hydrocarbon's density and molecular weight
    with its partition coefficient.

        rho_arom = density of aromatic

        mol_wt = molecular weight

        s = solubility

        s = Huibers(rho_arom, mol_wt)

        k_ow = Lee(s)

             = Lee(Huibers(rho_arom, mol_wt))

    We calibrate an empiric coefficient A with the measured values of
    Toluene.
    '''
    A = (Toluene.k_ow *
         np.exp(-0.087 * Toluene.mol_wt / Toluene.density) *
         Toluene.mol_wt /
         Toluene.density)

    @classmethod
    def partition_coeff(cls, mol_wt, density):
        '''
        :param mol_wt: Molecular weight in kg/kmole
        :param density: Density in kg/m^3
        '''
        return cls.A * density / (np.exp(-0.087 * mol_wt / density) * mol_wt)
