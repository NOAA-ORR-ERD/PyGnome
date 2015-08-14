'''
tests for geojson outputter
'''
import os
from glob import glob
from datetime import timedelta

import numpy as np
import pytest

from gnome.outputters import TrajectoryGeoJsonOutput
from gnome.spill import SpatialRelease, Spill, point_line_release_spill
from gnome.basic_types import oil_status

from ..conftest import sample_model

@pytest.fixture(scope='function')
def model(sample_model, output_dir):
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

    model.outputters += TrajectoryGeoJsonOutput(output_dir=output_dir)
    model.rewind()
    return model

def test_init():
    'simple initialization passes'
    g = TrajectoryGeoJsonOutput()
    assert g.output_dir is None
    assert g.round_to == 4
    assert g.round_data


def test_clean_output_files(model, output_dir):
    'test geojson outputter with a model since simplest to do that'
    model.rewind()
    model.full_run()
    files = glob(os.path.join(output_dir, '*.geojson'))
    print files
    assert len(files) == model.num_time_steps

    model.outputters[-1].clean_output_files()

    files = glob(os.path.join(output_dir, '*.geojson'))
    print files
    assert len(files) == 0


@pytest.mark.slow
@pytest.mark.parametrize("output_ts_factor", [1, 2, 2.4])
def test_write_output_post_run(model, output_ts_factor, output_dir):
    model.rewind()
    o_geojson = model.outputters[-1]
    o_geojson.output_timestep = timedelta(seconds=model.time_step *
                                          output_ts_factor)

    del model.outputters[-1]

    model.full_run()
    # purge the output
    # note: there are two outputter on the model -- not sure why
    #       so still one after removing this one so need to clear output dir
    o_geojson.clean_output_files()

    files = glob(os.path.join(output_dir, '*.geojson'))
    assert len(files) == 0

    o_geojson.write_output_post_run(model.start_time,
                                    model.num_time_steps,
                                    cache=model._cache,
                                    spills=model.spills)

    files = glob(os.path.join(output_dir, '*.geojson'))

    assert len(files) == int((model.num_time_steps-2)/output_ts_factor) + 2

    o_geojson.output_timestep = None
    model.outputters += o_geojson


def test_geojson_multipoint_output(model):
    'test geojson outputter with a model since simplest to do that'
    # default is to round data
    odir = model.outputters[-1].output_dir
    model.outputters[-1].output_dir = None
    model.rewind()
    round_to = model.outputters[0].round_to
    for step in model:
        fc = step['TrajectoryGeoJsonOutput']['feature_collection']['features']
        assert 'output_filename' not in step['TrajectoryGeoJsonOutput']
        for feature in fc:
            if feature['properties']['sc_type'] == 'uncertain':
                uncertain = True
            else:
                uncertain = False

            print feature['geometry']['coordinates']
            np.allclose(model.spills.LE('positions', uncertain)[:, :2],
                        feature['geometry']['coordinates'],
                        atol=10 ** -round_to)

    model.outputters[-1].output_dir = odir
