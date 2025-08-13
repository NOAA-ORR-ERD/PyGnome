"""
tests for the tamoc_spill module

The TAMOC Spill module can be at least partially run without tamoc.

NOTE: we might want to "mock" tamoc for this ultimately

but there's hardly anything here anyway...

"""

from gnome.spills.substance import NonWeatheringSubstance
import pytest

try:
    # we don't actually need it directly in the tests -- but need to know if can be imported.
    import tamoc as tamoc_raw
    from gnome.tamoc.tamoc_spill import TamocSpill
except ImportError:
    # if we can't import the tamoc module all tests in this module are skipped.
    pytestmark = pytest.mark.skipif(True, reason="this test requires the tamoc package")



def test_init():
    """
    just making sure we can initialize it with defaults
    """

    ts = TamocSpill(substance=NonWeatheringSubstance())


