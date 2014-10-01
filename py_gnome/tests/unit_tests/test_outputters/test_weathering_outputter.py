'''
tests for geojson outputter
'''
import os
import shutil
from glob import glob

import numpy as np
import pytest
import geojson

from gnome.environment import constant_wind, Water
from gnome.weatherers import Evaporation
from gnome.spill.elements import floating_weathering

from gnome.spill import SpatialRelease, Spill, point_line_release_spill

from gnome.outputters import WeatheringOutput
from gnome.persist import load

here = os.path.dirname(__file__)

datadir = os.path.join(here, 'sample_data')
output_dir = os.path.join(here, 'weathering_output')


@pytest.fixture(scope='module')
def model(sample_model):
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    model = sample_model['model']
    rel_start_pos = sample_model['release_start_pos']
    rel_end_pos = sample_model['release_end_pos']

    model.cache_enabled = True
    model.uncertain = False

    print 'adding a Weatherer'
    model.environment += [Water(311.15),
                          constant_wind(1.0, 0.0)]
    # figure out mid-run save for mass_balance attribute, then add this in
    model.weatherers += Evaporation(model.environment[-2],
                                    model.environment[-1])

    N = 10  # a line of ten points
    line_pos = np.zeros((N, 3), dtype=np.float64)
    line_pos[:, 0] = np.linspace(rel_start_pos[0], rel_end_pos[0], N)
    line_pos[:, 1] = np.linspace(rel_start_pos[1], rel_end_pos[1], N)

    # print start_points
    et = floating_weathering(substance='FUEL OIL NO.6')
    model.spills += point_line_release_spill(1,
                                             start_position=rel_start_pos,
                                             release_time=model.start_time,
                                             end_position=rel_end_pos,
                                             amount=1000,
                                             units='kg',
                                             element_type=et)

    #release = SpatialRelease(start_position=line_pos,
    #                         release_time=model.start_time,
    #                         element_type=et)
    #model.spills += Spill(release)
    model.outputters += WeatheringOutput(output_dir=output_dir)
    model.rewind()

    return model


def test_init():
    'simple initialization passes'
    g = WeatheringOutput()
    assert g.output_dir is None


def test_model_webapi_output(model):
    'Test weathering outputter with a model since simplest to do that'
    model.rewind()
    for step in model:
        assert 'WeatheringOutput' in step
        assert 'evaporated' in step['WeatheringOutput']['mass_balance']
        print step['WeatheringOutput']['mass_balance']
        #if step['step_num'] == 0:
        #    assert 'mass_balance' not in step


def test_model_dump_output(model):
    'Test weathering outputter with a model since simplest to do that'
    model.outputters[0].output_dir = output_dir
    model.rewind()
    model.full_run()
    files = glob(os.path.join(output_dir, '*.json'))
    assert len(files) == model.num_time_steps
    model.outputters[0].output_dir = None
