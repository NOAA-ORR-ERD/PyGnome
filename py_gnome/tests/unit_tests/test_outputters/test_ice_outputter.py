'''
tests for geojson outputter
'''
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2, width=120)

import time
from datetime import datetime

import numpy as np
import pytest

# from gnome.basic_types import oil_status
from gnome.utilities import time_utils

from gnome.spill import SpatialRelease, Spill, point_line_release_spill
from gnome.movers import IceMover
from gnome.outputters import IceGeoJsonOutput

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

    model.outputters += IceGeoJsonOutput([c_ice_mover])

    model.rewind()

    return model


def test_init():
    'simple initialization passes'
    g = IceGeoJsonOutput([c_ice_mover])
    assert g.ice_movers[0] == c_ice_mover


def test_ice_geojson_output(model):
    '''
        test geojson outputter with a model since simplest to do that
    '''
    # default is to round data
    model.rewind()

    begin = time.time()
    for step in model:
        print 'got step at: ', time.time() - begin
        print '\n\n\n', step.keys()
        assert 'IceGeoJsonOutput' in step
        assert 'step_num' in step['IceGeoJsonOutput']
        assert 'time_stamp' in step['IceGeoJsonOutput']
        assert 'feature_collections' in step['IceGeoJsonOutput']

        fcs = step['IceGeoJsonOutput']['feature_collections']

        # There should be only one key, but we will iterate anyway.
        # We just want to verify here that our keys exist in the movers
        # collection.
        for k in fcs.keys():
            assert model.movers.index(k) > 0

        # Check that our structure is correct.
        for fc in fcs.values():
            assert 'type' in fc
            assert fc['type'] == 'FeatureCollection'
            assert 'features' in fc
            assert len(fc['features']) > 0

            for feature in fc['features']:
                assert 'type' in feature
                assert feature['type'] == 'Feature'

                assert 'properties' in feature
                assert 'thickness' in feature['properties']
                assert 'coverage' in feature['properties']

                assert 'geometry' in feature
                geometry = feature['geometry']
                assert len(geometry) == 2

        print 'checked step at: ', time.time() - begin





















