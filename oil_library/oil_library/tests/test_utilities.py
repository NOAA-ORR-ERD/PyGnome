'''
test functions in utilities modules
'''
import numpy as np

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
