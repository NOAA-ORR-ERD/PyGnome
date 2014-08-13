'''
tests for geojson outputter
'''
import os
import shutil
from glob import glob

import numpy as np
import pytest
import geojson

from gnome.basic_types import oil_status
from gnome.utilities.time_utils import date_to_sec

from gnome.spill import SpatialRelease, Spill, point_line_release_spill

from gnome.outputters import WeatheringOutput
from gnome.persist import load

here = os.path.dirname(__file__)
up_one = os.path.dirname(here)

datadir = os.path.join(up_one, 'sample_data')
output_dir = os.path.join(up_one, 'geojson_output')


@pytest.fixture(scope='module')
def model(sample_model, request):
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    model = sample_model['model']
    rel_start_pos = sample_model['release_start_pos']
    rel_end_pos = sample_model['release_end_pos']

    model.cache_enabled = True
    model.uncertain = True

    N = 10  # a line of ten points
    line_pos = np.zeros((N, 3), dtype=np.float64)
    line_pos[:, 0] = np.linspace(rel_start_pos[0], rel_end_pos[0], N)
    line_pos[:, 1] = np.linspace(rel_start_pos[1], rel_end_pos[1], N)

    # print start_points

    model.spills += point_line_release_spill(1,
                                             start_position=rel_start_pos,
                                             release_time=model.start_time,
                                             end_position=rel_end_pos)

    release = SpatialRelease(start_position=line_pos,
                           release_time=model.start_time)

    model.spills += Spill(release)
    model.outputters += WeatheringOutput(output_dir=output_dir)
    model.rewind()

    return model


def test_init():
    'simple initialization passes'
    g = WeatheringOutput()
    assert g.output_dir == './'


def test_model_dump_output(model):
    'Test weathering outputter with a model since simplest to do that'
    model.rewind()
    model.full_run()
    files = glob(os.path.join(output_dir, '*.geojson'))
    assert len(files) == model.num_time_steps
