"""
tests for the GNOME Oil object

WARNING: very incomplete!
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import pytest

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
                kvis=None,
                kvis_ref_temps=None,
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
    oil.kvis_weathering = []

    kv = oil.kvis_at_temp(288.0)

    print(kv)

    assert False



