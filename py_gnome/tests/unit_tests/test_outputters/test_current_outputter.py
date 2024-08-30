
from datetime import datetime

import numpy as np
import pytest

# from gnome.basic_types import oil_status
from gnome.utilities import time_utils

from gnome.model import Model
from gnome.maps.map import GnomeMap
from gnome.environment import Tide
from gnome.spills.spill import Spill, point_line_spill
from gnome.spills.release import Release
from gnome.movers import CatsMover
from gnome.outputters.json import CurrentJsonOutput

from ..conftest import testdata


td = Tide(filename=testdata['CatsMover']['tide'])
c_cats = CatsMover(testdata['CatsMover']['curr'], tide=td)


@pytest.fixture(scope='function')
def model(output_dir):
    """
    a simple model with only the mover / environment object needed
    """
    rel_time = datetime(2012, 9, 15, 12)
    time_step = 15 * 60  # seconds
    # model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time))

    model = Model(start_time=rel_time,
                  time_step=time_step)

    model.cache_enabled = True

    # model.environment += td
    model.movers += c_cats
    model.outputters += CurrentJsonOutput([c_cats])

    return model


def test_init():
    'simple initialization passes'
    g = CurrentJsonOutput([c_cats])
    assert g.current_movers[0] == c_cats


def test_current_grid_json_output(model):
    '''
    test geojson outputter with a model since simplest to do that
    '''
    for step in model:
        assert 'step_num' in step
        assert 'CurrentJsonOutput' in step

        fcs = step['CurrentJsonOutput']

        # There should be only one key, but we will iterate anyway.
        # We just want to verify here that our keys exist in the movers
        # collection.
        for k in list(fcs.keys()):
            try:
                model.movers[k]
            except KeyError:
                assert False, "mover is not in the movers collection"

        # Check that our structure is correct.
        for fc in list(fcs.values()):
            assert 'direction' in fc
            assert 'magnitude' in fc
            assert len(fc['direction']) > 0
            assert len(fc['magnitude']) > 0
            assert len(fc['magnitude']) == len(fc['direction'])
