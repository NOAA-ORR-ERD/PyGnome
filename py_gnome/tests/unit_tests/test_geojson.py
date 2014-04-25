'''
tests for geojson outputter
'''
import geojson
import os
import shutil

import numpy as np
import pytest

from gnome.outputters import GeoJson
from gnome.spill import SpatialRelease, Spill


basedir = os.path.dirname(__file__)
datadir = os.path.join(basedir, 'sample_data')
output_dir = os.path.join(basedir, 'geojson_output')


@pytest.fixture(scope='function')
def model(sample_model):
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    model = sample_model['model']
    rel_start_pos = sample_model['release_start_pos']
    rel_end_pos = sample_model['release_end_pos']

    model.cache_enabled = True
    model.uncertain = False

    model.outputters += GeoJson(output_dir=output_dir)

    N = 10  # a line of ten points
    line_pos = np.zeros((N, 3), dtype=np.float64)
    line_pos[:, 0] = np.linspace(rel_start_pos[0], rel_end_pos[0], N)
    line_pos[:, 1] = np.linspace(rel_start_pos[1], rel_end_pos[1], N)

    # print start_points

    release = SpatialRelease(start_position=line_pos,
                           release_time=model.start_time)

    model.spills += Spill(release)

    return model


def test_init():
    'simple initialization passes'
    g = GeoJson()
    assert g.output_dir == './'
    assert g.roundto == 4
    assert g.round_data


def test_model_outputgeojson(model):
    'test geojson outputter with a model since simplest to do that'
    model.full_run()
    assert os.path.exists(output_dir)
