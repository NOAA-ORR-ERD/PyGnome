'''
test functions in utilities modules
'''
import numpy as np
import pytest

from oil_library import get_oil
from oil_library.utilities import get_density


name = 'FUEL OIL NO.6'


class TestGetDensity:
    oil_ = get_oil(name)
    temps = [d.ref_temp_k for d in oil_.densities]
    temps.extend([oil_.densities[0].ref_temp_k] * 3)
    temps_array = np.asarray(temps).reshape(len(temps), -1)

    def assert_arrays(self, output):
        for ix, d in enumerate(self.oil_.densities):
            assert d.kg_m_3 == output[ix]

        assert all(output[len(self.oil_.densities):] ==
                   self.oil_.densities[0].kg_m_3)

    def test_temp_scalar(self):
        density = get_density(self.oil_, self.temps[0])
        assert self.oil_.densities[0].kg_m_3 == density

    def test_temp_list(self):
        d_array = get_density(self.oil_, self.temps)
        assert len(d_array) == len(self.temps)
        self.assert_arrays(d_array)

    def test_temp_array(self):
        out = np.zeros_like(self.temps_array)
        d_array = get_density(self.oil_, self.temps_array, out)
        self.assert_arrays(out)
        assert d_array is out


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
