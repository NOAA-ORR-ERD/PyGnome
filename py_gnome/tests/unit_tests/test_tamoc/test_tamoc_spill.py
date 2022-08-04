"""
tests for the tamoc_spill module

The TAMOC Spill module can be at least partially run without tamoc.

NOTE: we might want to "mock" tamoc for this ultimately

"""

from gnome.tamoc.tamoc_spill import TamocSpill
from gnome.spills.substance import NonWeatheringSubstance

def test_init():
    """
    just making sure we can initialize it with defaults
    """

    ts = TamocSpill(substance=NonWeatheringSubstance())


