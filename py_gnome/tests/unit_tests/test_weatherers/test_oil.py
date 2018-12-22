"""
test code for the gnome oil object
"""

import numpy as np

from gnome.weatherers.oil import Oil


# one component oil
sample_oil_1 =  {'name': 'benzene',
                 # Physical Properties
                 'api': 28.6,
                 'pour_point': 278.68,
                 'solubility': 1.78,  # kg/m^3
                 'densities':         [886.30,   859.55,   838.73,   814.86],
                 'density_ref_temps': [285.93,   310.93,   330.37,   352.59],
                 'kvis':              [8.513e-7, 7.357e-7, 5.415e-7, 3.89e-7,],
                 'kvis_ref_temp':     [283.15,   293.15,   323.15,   353.15],
                 # emulsification properties
                 'bullwinkle_fraction': None,
                 'bullwinkle_time': None,
                 'emulsion_water_fraction_max': None,
                 # Pseudo Components:
                 'mass_fractions': [1.0],
                 'boiling_points': [353.05],
                 'molecular_weights': [78.11, ],
                 'component_densities': [814.0, ],
                 'sara_types': ['aromatics'],
                 }


def test_init_1():
    oil = Oil(**sample_oil_1)
    assert np.alltrue(oil.boiling_points == np.array([353.05], dtype=np.float64))




