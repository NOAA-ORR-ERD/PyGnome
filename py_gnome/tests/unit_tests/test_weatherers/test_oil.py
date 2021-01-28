"""
tests for the GNOME Oil object

WARNING: very incomplete!
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import pytest
import numpy as np

from gnome.weatherers.oil import Oil

# NOTE: maybe better to have test oils defined here?
from gnome.spill.sample_oils import _sample_oils

@pytest.fixture
def empty_oil():
     return Oil(name='empty_oil',
                api=None,
                pour_point=None,
                solubility=None,  # kg/m^3
                # emulsification properties
                bullwinkle_fraction=None,
                bullwinkle_time=None,
                emulsion_water_fraction_max=None,
                densities=None,
                density_ref_temps=None,
                density_weathering=None,
                kvis=[],
                kvis_ref_temps=[],
                kvis_weathering=[],
                # PCs:
                mass_fraction=[],
                boiling_point=[],
                molecular_weight=[],
                component_density=[],
                sara_type=[],
                flash_point=[],
                adios_oil_id=''
                )

# from crude sample:
 # 'kvis': [0.0005, 0.0006, 8.3e-05, 8.53e-05],
 # 'kvis_ref_temps': [273.0, 288.0, 293.0, 311.0],

def test_can_init():
    """
    can we initialize from a complete sample
    """

    oil = Oil(**_sample_oils['oil_crude'])

    print(oil)


def test_kvis_at_temp_single(empty_oil):
    oil = empty_oil

    oil.kvis = [0.0006]
    oil.kvis_ref_temps = [288.0]
    oil.kvis_weathering = [0.0]

    kv = oil.kvis_at_temp(288.0)

    print(kv)

    assert kv == 0.0006


def test_kvis_at_temp_single_range(empty_oil):
    """
    test using the default value for the decay constant

    hand calculated, based on decay constant of 2100K
    value at 273C: 2.0213282964723807e-05
    value at 300C: 1.0115129451432752e-05
    """
    oil = empty_oil

    oil.kvis = [1.32e-05]
    oil.kvis_ref_temps = [289.01]
    oil.kvis_weathering = [0.0]

    kvis = oil.kvis_at_temp([273, 300])

    print(kvis)

    assert np.allclose(kvis, [2.021328e-05, 1.011512e-05], rtol=1e-4)


def test_kvis_at_temp_two(empty_oil):
    oil = empty_oil

    oil.kvis = [0.0006, 8.3e-05]
    oil.kvis_ref_temps = [288.0, 293.0]
    oil.kvis_weathering = [0.0, 0.0]

    assert np.allclose(oil.kvis_at_temp(288.0), 0.0006, rtol=1e-8)
    assert np.allclose(oil.kvis_at_temp(293.0), 8.3e-05, rtol=1e-8)


def test_kvis_at_temp_six(empty_oil):
    """
    Test the least squared fit to six points

    data from HOOPS BLEND, ExxonMobil
    """
    oil = empty_oil

    oil.kvis = np.array([19.6, 13.2, 9.85, 9.39, 6.99, 6.62]) * 1e-6
    oil.kvis_ref_temps = np.array([4.85, 15.85, 24.85, 26.85, 37.85, 39.85]) + 273.16
    oil.kvis_weathering = [0.0] * len(oil.kvis)

    # assert oil.kvis_at_temp(288.0) == 0.0006
    # assert oil.kvis_at_temp(293.0) == 8.3e-05

    kvis = oil.kvis_at_temp(oil.kvis_ref_temps)

    print(oil.kvis)
    print(kvis)

    assert np.allclose(kvis, oil.kvis, rtol=5e-2)



