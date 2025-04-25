'''
tests for geojson outputter
'''

# from builtins import int
# from future import standard_library
# standard_library.install_aliases()
# from builtins import *

import os
from glob import glob
from datetime import timedelta

import numpy as np
import pytest

from gnome.outputters import TrajectoryGeoJsonOutput
from gnome.spills.spill import Spill, point_line_spill
from gnome.spills.release import PolygonRelease
from gnome.basic_types import oil_status
from gnome.environment import constant_wind, Water
from gnome.weatherers import Evaporation

from ..conftest import sample_model_fcn, sample_model_weathering, test_oil


@pytest.fixture(scope='function')
def model(sample_model_fcn, output_dir):
    model = sample_model_weathering(sample_model_fcn, test_oil)

    rel_start_pos = sample_model_fcn['release_start_pos']
    rel_end_pos = sample_model_fcn['release_end_pos']

    model.cache_enabled = True
    model.uncertain = True

    water, wind = Water(), constant_wind(1., 0)
    model.environment += [water, wind]
    model.weatherers += Evaporation(water=water, wind=wind)

    N = 10  # a line of ten points
    line_pos = np.zeros((N, 3), dtype=np.float64)
    line_pos[:, 0] = np.linspace(rel_start_pos[0], rel_end_pos[0], N)
    line_pos[:, 1] = np.linspace(rel_start_pos[1], rel_end_pos[1], N)

    # print start_points

    model.spills += point_line_spill(1,
                                             start_position=rel_start_pos,
                                             release_time=model.start_time,
                                             end_position=rel_end_pos,
                                             substance=model.spills[0].substance,
                                             amount=100,
                                             units='tons')

    model.outputters += TrajectoryGeoJsonOutput(output_dir=output_dir)
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
    print(files)
    assert len(files) == model.num_time_steps

    model.outputters[-1].clean_output_files()

    files = glob(os.path.join(output_dir, '*.geojson'))
    print(files)
    assert len(files) == 0


# @pytest.mark.slow
@pytest.mark.parametrize("output_ts_factor", [1, 2, 2.4])
def test_write_output_post_run(model, output_ts_factor, output_dir):

    # maybe better to add the outputter here?
    o_geojson = model.outputters[-1]
    o_geojson.output_timestep = timedelta(seconds=model.time_step *
                                          output_ts_factor)

    # so the run won't write output
    model.outputters.clear()

    model.full_run()
    # purge the output (written by other tests) -- not good
    o_geojson.clean_output_files()

    # just making sure that it's clean
    files = glob(os.path.join(output_dir, '*.geojson'))
    assert len(files) == 0

    o_geojson.write_output_post_run(model_start_time=model.start_time,
                                    num_time_steps=model.num_time_steps,
                                    model_time_step=900,  # not used, but required by the method
                                    cache=model._cache,
                                    spills=model.spills)

    files = glob(os.path.join(output_dir, '*.geojson'))

    assert len(files) == int((model.num_time_steps - 2) / output_ts_factor) + 2

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
        uncertain_fc = step['TrajectoryGeoJsonOutput']['uncertain']['features']
        certain_fc = step['TrajectoryGeoJsonOutput']['certain']['features']
        fc = uncertain_fc + certain_fc
        assert 'output_filename' not in step['TrajectoryGeoJsonOutput']
        for feature in fc:
            if feature['properties']['sc_type'] == 'uncertain':
                uncertain = True
            else:
                uncertain = False

            assert 'mass' in feature['properties']
            assert feature['properties']['mass'] > 0

            assert 'spill_num' in feature['properties']
            assert isinstance(feature['properties']['spill_num'], int)
            # assert type(feature['properties']['spill_num']) is int

            mask = np.where(model.spills.LE('status_codes', uncertain) ==
                            feature['properties']['status_code'])

            np.allclose(model.spills.LE('positions', uncertain)[mask, :2],
                        feature['geometry']['coordinates'],
                        atol=10 ** -round_to)

    model.outputters[-1].output_dir = odir
