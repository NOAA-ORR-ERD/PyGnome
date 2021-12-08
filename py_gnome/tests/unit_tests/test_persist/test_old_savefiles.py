"""
These tests will try load any *.gnome file in the sample_data
directory.

It should assure that old save files continue to work.

Note that we really should have a more comprehensive collection here
"""

from pathlib import Path

import pytest

from gnome.model import Model


SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"

save_files = SAMPLE_DATA_DIR.glob("*.gnome")

@pytest.mark.parametrize('savefilename', save_files)
def test_old_save_files(savefilename):
    print("testing loading of:", savefilename)
    model = Model.load_savefile(str(savefilename))

    # this is kinda slow, so not bothering.
    # model.full_run()



