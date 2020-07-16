"""
tests for the tamoc_spill module

The TAMOC Spill module can be at least partially run without tamoc.

NOTE: we might want to "mock" tamoc for this ultimately

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from future import standard_library
standard_library.install_aliases()
from builtins import *
from gnome.tamoc.tamoc_spill import TamocSpill
from gnome.spill.substance import NonWeatheringSubstance

def test_init():
    """
    just making sure we can initialize it with defaults
    """

    ts = TamocSpill(substance=NonWeatheringSubstance())


