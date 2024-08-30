'''
tests for geojson outputter
'''

import time
from datetime import datetime

import numpy as np
import pytest

from gnome.model import Model
from gnome.maps.map import GnomeMap
from gnome.spills.spill import Spill, point_line_spill
from gnome.spills.release import Release
from gnome.movers import IceMover
from gnome.outputters import IceJsonOutput

from ..conftest import testdata


curr_file = testdata['IceMover']['ice_curr_curv']
topology_file = testdata['IceMover']['ice_top_curv']
c_ice_mover = IceMover(curr_file, topology_file)


@pytest.fixture(scope='function')
def model(output_dir):
    """
    A simple model with only the mover / environment object needed
    """
    time_step = 15 * 60  # seconds

    model = Model(start_time = datetime(2015, 5, 14, 0),
                  time_step=time_step)

    model.movers += c_ice_mover
    model.outputters += IceJsonOutput([c_ice_mover])

    return model

def test_init():
    'simple initialization passes'
    g = IceJsonOutput([c_ice_mover])
    assert g.ice_movers[0] == c_ice_mover


def test_ice_geojson_output(model):
    '''
        test geojson outputter with a model since simplest to do that
    '''
    # default is to round data
    model.rewind()

    begin = time.time()
    for step in model:
        print('\n\ngot step at: ', time.time() - begin)

        assert 'step_num' in step
        assert 'IceJsonOutput' in step
        assert 'time_stamp' in step['IceJsonOutput']
        assert 'data' in step['IceJsonOutput']

        fcs = step['IceJsonOutput']['data']

        # There should be only one key, but we will iterate anyway.
        # We just want to verify here that our keys exist in the movers
        # collection.
        for k in list(fcs.keys()):
            try:
                model.movers[k]
            except KeyError:
                assert False, "mover is not in the movers collection"

        # Check that our structure is correct.
        for fc_list in list(fcs.values()):
            assert 'concentration' in fc_list
            assert 'thickness' in fc_list

            assert len(fc_list['concentration']) > 0
            assert len(fc_list['thickness']) > 0
            assert len(fc_list['concentration']) == len(fc_list['thickness'])

