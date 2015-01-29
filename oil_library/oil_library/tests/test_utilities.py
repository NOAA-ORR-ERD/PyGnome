'''
test functions in utilities modules
'''
import numpy as np
import pytest

from oil_library import get_oil
from oil_library.utilities import (get_density,
                                   get_viscosity,
                                   get_v_max)


oil_ = get_oil('FUEL OIL NO.6')

# Test case - get ref temps from densities then append ref_temp for
# density at 0th index for a few more values:
#    density_test = [d.ref_temp_k for d in oil_.densities]
#    density_test.append(oil_.densities[0].ref_temp_k)
density_tests = [oil_.densities[ix].ref_temp_k if ix < len(oil_.densities)
                 else oil_.densities[0].ref_temp_k
                 for ix in range(0, len(oil_.densities) + 3)]
density_exp = [d.kg_m_3 for temp in density_tests for d in oil_.densities
               if abs(d.ref_temp_k - temp) == 0]

'''
test get_density for
- scalar
- list, tuple
- numpy arrays as row/column and with/without output arrays
'''


@pytest.mark.parametrize(("temps", "exp_value", "use_out"),
                         [(density_tests[0], density_exp[0], False),
                          (density_tests, density_exp, False),
                          (tuple(density_tests), density_exp, False),
                          (np.asarray(density_tests).reshape(len(density_tests), -1),
                           np.asarray(density_exp).reshape(len(density_tests), -1),
                           False),
                          (np.asarray(density_tests).reshape(len(density_tests), -1),
                           np.asarray(density_exp).reshape(len(density_tests), -1),
                           True),
                          (np.asarray(density_tests),
                           np.asarray(density_exp), False),
                          (np.asarray(density_tests),
                           np.asarray(density_exp), True)])
def test_get_density(temps, exp_value, use_out):
    if use_out:
        out = np.zeros_like(temps)
        get_density(oil_, temps, out)
    else:
        out = get_density(oil_, temps)
    assert np.all(out == exp_value)   # so it works for scalar + arrays


# Test case - get ref temps from kvis then append ref_temp for
# kvis at 0th index for a few more values:
#    viscosity_tests = [d.ref_temp_k for d in oil_.densities]
#    viscosity_tests.append(oil_.densities[0].ref_temp_k)
v_max = get_v_max(oil_)
viscosity_tests = [oil_.kvis[ix].ref_temp_k if ix < len(oil_.kvis)
                   else oil_.kvis[0].ref_temp_k
                   for ix in range(0, len(oil_.kvis) + 3)]
viscosity_exp = [(d.m_2_s, v_max)[v_max > d.m_2_s] for temp in viscosity_tests
                 for d in oil_.kvis if abs(d.ref_temp_k - temp) == 0]


@pytest.mark.parametrize(("temps", "exp_value", "use_out"),
                         [(viscosity_tests[0], viscosity_exp[0], False),
                          (viscosity_tests, viscosity_exp, False),
                          (tuple(viscosity_tests), viscosity_exp, False),
                          (np.asarray(viscosity_tests).reshape(len(viscosity_tests), -1),
                           np.asarray(viscosity_exp).reshape(len(viscosity_tests), -1),
                           False),
                          (np.asarray(viscosity_tests).reshape(len(viscosity_tests), -1),
                           np.asarray(viscosity_exp).reshape(len(viscosity_tests), -1),
                           True),
                          (np.asarray(viscosity_tests),
                           np.asarray(viscosity_exp), False),
                          (np.asarray(viscosity_tests),
                           np.asarray(viscosity_exp), True)])
def test_get_viscosity(temps, exp_value, use_out):
    if use_out:
        out = np.zeros_like(temps)
        get_viscosity(oil_, temps, out)
    else:
        out = get_viscosity(oil_, temps)
    assert np.all(out == exp_value)   # so it works for scalar + arrays
