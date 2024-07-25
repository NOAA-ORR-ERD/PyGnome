'''
tests for geojson outputter
'''

import time
from datetime import datetime

import numpy as np
import pytest

# from gnome.basic_types import oil_status
from gnome.utilities import time_utils

from gnome.maps.map import GnomeMap
from gnome.spills.spill import Spill, point_line_spill
from gnome.spills.release import Release
from gnome.movers import IceMover
from gnome.outputters import IceGeoJsonOutput, IceJsonOutput

from ..conftest import testdata

pytestmark = pytest.mark.skip("ice outputter not currently useful -- tests slow")


curr_file = testdata['IceMover']['ice_curr_curv']
topology_file = testdata['IceMover']['ice_top_curv']
c_ice_mover = IceMover(curr_file, topology_file)


# NOTE: we really don't need a full model here
#       unit tests should test as little as possible ...
@pytest.fixture(scope='function')
def model(sample_model, output_dir):
    model = sample_model['model']

    #reset map to water world, so we dont get overlapping issues
    model.map = GnomeMap()
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
    model.spills += point_line_spill(1,
                                             start_position=start_pos,
                                             release_time=model.start_time,
                                             end_position=start_pos)

    release = Release(custom_positions=line_pos,
                             release_time=model.start_time)

    model.spills += Spill(release)

    model.movers += c_ice_mover

    model.outputters += IceGeoJsonOutput([c_ice_mover])

    model.rewind()

    return model


@pytest.fixture(scope='function')
def model(output_dir):
    """
    A simple model with only the mover / environment object needed
    """
    time_step = 15 * 60  # seconds

    model = Model(start_time = datetime(2015, 5, 14, 0),
                  time_step=time_step)

    model.movers += c_ice_mover

    model.outputters += IceGeoJsonOutput([c_ice_mover])

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
        print('\n\ngot step at: ', time.time() - begin)

        assert 'step_num' in step
        assert 'IceGeoJsonOutput' in step
        assert 'time_stamp' in step['IceGeoJsonOutput']
        assert 'feature_collections' in step['IceGeoJsonOutput']

        fcs = step['IceGeoJsonOutput']['feature_collections']

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

            # our first feature collection should be for coverage
            fc = fc_list[0]
            assert 'type' in fc
            assert fc['type'] == 'FeatureCollection'
            assert 'features' in fc
            assert len(fc['features']) > 0

            for feature in fc['features']:
                assert 'type' in feature
                assert feature['type'] == 'Feature'

                assert 'properties' in feature
                assert 'coverage' in feature['properties']

                assert 'geometry' in feature
                geometry = feature['geometry']

                assert 'type' in geometry
                assert geometry['type'] == 'MultiPolygon'

                assert 'coordinates' in geometry
                assert len(geometry['coordinates']) > 0

            # our second feature collection should be for thickness
            fc = fc_list[1]
            assert 'type' in fc
            assert fc['type'] == 'FeatureCollection'
            assert 'features' in fc
            assert len(fc['features']) > 0

            for feature in fc['features']:
                assert 'type' in feature
                assert feature['type'] == 'Feature'

                assert 'properties' in feature
                assert 'thickness' in feature['properties']

                assert 'geometry' in feature
                geometry = feature['geometry']

                assert 'type' in geometry
                assert geometry['type'] == 'MultiPolygon'

                assert 'coordinates' in geometry
                assert len(geometry['coordinates']) > 0

        print('checked step at: ', time.time() - begin)
