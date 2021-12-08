'''
tests for geojson outputter
'''


import time
from datetime import datetime

import numpy as np
import pytest

from gnome.spills import Release, Spill, surface_point_line_spill
from gnome.movers import IceMover
from gnome.outputters import IceJsonOutput

from ..conftest import testdata


curr_file = testdata['IceMover']['ice_curr_curv']
topology_file = testdata['IceMover']['ice_top_curv']
c_ice_mover = IceMover(curr_file, topology_file)


@pytest.fixture(scope='function')
def model(sample_model, output_dir):
    model = sample_model['model']
    rel_start_pos = sample_model['release_start_pos']
    rel_end_pos = sample_model['release_end_pos']

    model.start_time = datetime(2015, 5, 14, 0)
    model.cache_enabled = True
    model.uncertain = True

    N = 10  # a line of ten points
    line_pos = np.zeros((N, 3), dtype=np.float64)
    line_pos[:, 0] = np.linspace(rel_start_pos[0], rel_end_pos[0], N)
    line_pos[:, 1] = np.linspace(rel_start_pos[1], rel_end_pos[1], N)

    start_pos = (-164.01696, 72.921024, 0)
    model.spills += surface_point_line_spill(1,
                                             start_position=start_pos,
                                             release_time=model.start_time,
                                             end_position=start_pos)

    release = Release(custom_positions=line_pos,
                             release_time=model.start_time)

    model.spills += Spill(release)

    model.movers += c_ice_mover

    model.outputters += IceJsonOutput([c_ice_mover])

    model.rewind()

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
            assert model.movers.index(k) > 0

        # Check that our structure is correct.
        for fc_list in list(fcs.values()):
            assert 'concentration' in fc_list
            assert 'thickness' in fc_list

            assert len(fc_list['concentration']) > 0
            assert len(fc_list['thickness']) > 0
            assert len(fc_list['concentration']) == len(fc_list['thickness'])

