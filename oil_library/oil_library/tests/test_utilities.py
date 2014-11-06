'''
test functions in utilities modules
'''
import numpy as np
import pytest

from oil_library import get_oil
from oil_library.utilities import get_density


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


# The viscosity expected value isn't so straight forward
#==============================================================================
# viscosity_tests = [oil_.kvis[ix].ref_temp_k if ix < len(oil_.kvis)
#                    else oil_.kvis[0].ref_temp_k
#                    for ix in range(0, len(oil_.kvis) + 3)]
#==============================================================================
#==============================================================================
# # moved test here though it needs to be updated after implementing
# # get_viscosity function in utilities module
# @pytest.mark.xfail
# @pytest.mark.parametrize(('oil', 'temp', 'viscosity'),
#                          [('FUEL OIL NO.6', 311.15, 0.000383211),
#                           ('FUEL OIL NO.6', 288.15, 0.045808748),
#                           ('FUEL OIL NO.6', 280.0, 0.045808749)
#                           ])
# def test_OilProps_Viscosity(oil, temp, viscosity):
#     """
#         test dbquery worked for an example like FUEL OIL NO.6
#         Here are the measured viscosities:
#            [<KVis(meters_squared_per_sec=1.04315461221, ref_temp=273.15, weathering=0.0)>,
#             <KVis(meters_squared_per_sec=0.0458087487284, ref_temp=288.15, weathering=0.0)>,
#             <KVis(meters_squared_per_sec=0.000211, ref_temp=323.15, weathering=0.0)>]
#     """
#     o = get_oil_props(oil)
#     assert abs((o.viscosity - viscosity)/viscosity) < 1e-5  # < 0.001 %
#==============================================================================
