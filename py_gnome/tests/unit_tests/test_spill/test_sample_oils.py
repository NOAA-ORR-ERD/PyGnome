"""
tests for sample oils

All this does at this point is make sure you can make a GnomeOil out of them
"""

from gnome.spills.gnome_oil import GnomeOil
from gnome.spills import sample_oils

import pytest

all_oils = sample_oils._sample_oils.keys()


@pytest.mark.parametrize('name', all_oils)
def test_create(name):
    go = GnomeOil(name)

    assert go.name == name

