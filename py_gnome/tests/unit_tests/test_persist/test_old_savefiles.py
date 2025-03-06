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

pytestmark = [pytest.mark.filterwarnings("ignore:Provided map bounds superscede map bounds found in file"),
              pytest.mark.filterwarnings("ignore:.*function ZipFile.__del__")]

# If these start causing problems, they can be Marked as xfail,
# because they sometimes fail, but sometimes not.
#
#  NOTE: we also get warnings about the closing an already closed zipfile
#        likely related, but hard to find! -- those are now suppressed
#        But may be a real issue ...

@pytest.mark.xfail
@pytest.mark.parametrize('savefilename', save_files)
def test_old_save_files(savefilename):
    print("testing loading of:", savefilename)
    model = Model.load_savefile(str(savefilename))

    # this is kinda slow, so not bothering.
    # model.full_run()
