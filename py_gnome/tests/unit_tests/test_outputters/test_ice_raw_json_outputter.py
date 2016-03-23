'''
tests for geojson outputter
'''
import time
from datetime import datetime
from numbers import Number

import numpy as np
import pytest

from gnome.spill import SpatialRelease, Spill, point_line_release_spill
from gnome.movers import IceMover
from gnome.outputters import IceRawJsonOutput

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
    model.spills += point_line_release_spill(1,
                                             start_position=start_pos,
                                             release_time=model.start_time,
                                             end_position=start_pos)

    release = SpatialRelease(start_position=line_pos,
                             release_time=model.start_time)

    model.spills += Spill(release)

    model.movers += c_ice_mover

    model.outputters += IceRawJsonOutput([c_ice_mover])

    model.rewind()

    return model


def test_init():
    'simple initialization passes'
    g = IceRawJsonOutput([c_ice_mover])
    assert g.ice_movers[0] == c_ice_mover


def test_ice_geojson_output(model):
    '''
        test geojson outputter with a model since simplest to do that
    '''
    # default is to round data
    model.rewind()

    begin = time.time()
    for step in model:
        print '\n\ngot step at: ', time.time() - begin

        assert 'step_num' in step
        assert 'IceRawJsonOutput' in step
        assert 'time_stamp' in step['IceRawJsonOutput']
        assert 'feature_collections' in step['IceRawJsonOutput']

        fcs = step['IceRawJsonOutput']['feature_collections']

        # There should be only one key, but we will iterate anyway.
        # We just want to verify here that our keys exist in the movers
        # collection.
        for k in fcs.keys():
            assert model.movers.index(k) > 0

        # Check that our structure is correct.
        for fc_list in fcs.values():

            # our feature collection should be a list of all data
            for feature in fc_list:
                assert len(feature) == 3
                print feature
                assert isinstance(feature[0], Number)
                assert isinstance(feature[1], Number)

                assert len(feature[2]) == 4
                assert [isinstance(n, Number) for c in feature[2] for n in c]

        print 'checked step at: ', time.time() - begin
