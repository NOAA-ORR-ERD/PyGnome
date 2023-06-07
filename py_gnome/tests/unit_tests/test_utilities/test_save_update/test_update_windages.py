"""
Testing the save file updater for moving windages from
WindageInit directly to the substance
"""

from pathlib import Path

from gnome.model import Model

import pytest

HERE = Path(__file__).parent

v1_savefile = HERE / "v1_non_weatherable_2.gnome"

@pytest.mark.filterwarnings("ignore: Provided map bounds")
def test_windages():
    model = Model.load_savefile(v1_savefile)

    for spill in model.spills:
        assert spill.substance.windage_persist == 600
        assert spill.substance.windage_range == (0.02, 0.04)

