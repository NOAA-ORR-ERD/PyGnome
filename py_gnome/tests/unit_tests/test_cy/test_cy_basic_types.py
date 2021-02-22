"""
tests for cy_basic types
"""

import gnome.cy_gnome.cy_basic_types as cbt

def test_nothing():
    assert True


def test_oil_status():
    """
    note: probably shouldn't check for specific values here
    """
    assert cbt.oil_status.not_released == 0
    assert cbt.oil_status.in_water == 2
    assert cbt.oil_status.on_land == 3
    assert cbt.oil_status.off_maps == 7
    assert cbt.oil_status.evaporated == 10
    assert cbt.oil_status.to_be_removed == 12
    assert cbt.oil_status.on_tideflat == 32

