'''
tests for weathering outputter
'''

import os
from glob import glob
from datetime import timedelta

import numpy as np
import pytest

from gnome.utilities.inf_datetime import InfDateTime
from gnome.scripting import hours

from gnome.environment import constant_wind, Water, Waves
from gnome.weatherers import Evaporation, ChemicalDispersion, Skimmer, Burn
from gnome.spills import surface_point_line_spill

from gnome.outputters import WeatheringOutput

from ..conftest import test_oil


@pytest.fixture(scope='function')
def model(sample_model):
    model = sample_model['model']
    model.make_default_refs = True

    rel_start_pos = sample_model['release_start_pos']
    rel_end_pos = sample_model['release_end_pos']

    # model.cache_enabled = True # why use the cache -- it'll just slow things down!!!
    model.uncertain = False

    wind = constant_wind(1.0, 0.0)
    water = Water(311.15)
    model.environment += water

    waves = Waves(wind, water)
    model.environment += waves

    print("the environment:", model.environment)

    start_time = model.start_time

    model.duration = timedelta(hours=12)
    end_time = start_time + timedelta(hours=1)
    spill = surface_point_line_spill(100,
                                     start_position=rel_start_pos,
                                     release_time=start_time,
                                     end_release_time=start_time + hours(1),
                                     end_position=rel_end_pos,
                                     substance=test_oil,
                                     amount=1000,
                                     units='kg')
    model.spills += spill

    # figure out mid-run save for weathering_data attribute, then add this in
    # rel_time = model.spills[0].release_time

    skim_start = start_time + timedelta(hours=1)
    amount = model.spills[0].amount
    units = model.spills[0].units

    skimmer = Skimmer(.3 * amount,
                      units=units,
                      efficiency=0.3,
                      active_range=(skim_start,
                                    skim_start + hours(1)))

    # thickness = 1m so area is just 20% of volume
    volume = spill.get_mass() / spill.substance.density_at_temp()

    burn = Burn(0.2 * volume, 1.0,
                active_range=(skim_start, InfDateTime('inf')),
                efficiency=0.9)

    c_disp = ChemicalDispersion(.1, efficiency=0.5,
                                active_range=(skim_start,
                                              skim_start + timedelta(hours=1)),
                                waves=waves)

    model.weatherers += [Evaporation(),
                         c_disp,
                         burn,
                         skimmer]

    model.outputters += WeatheringOutput()
    model.rewind()

    return model


def test_init():
    'simple initialization passes'
    g = WeatheringOutput()
    assert g.output_dir is None


def test_model_webapi_output(model, output_dir):
    '''
    Test weathering outputter with a model since simplest to do that
    '''
    model.outputters[-1].output_dir = output_dir
    model.rewind()

    # floating mass at beginning of step - though tests will only pass for
    # nominal values
    for step in model:
        assert 'step_num' in step
        assert 'WeatheringOutput' in step
        sum_mass = 0.0
        for key in step['WeatheringOutput']:
            if not isinstance(step['WeatheringOutput'][key], dict):
                continue

            for process in ('evaporated', 'burned', 'skimmed', 'dispersed'):
                assert (process in step['WeatheringOutput'][key])
                sum_mass += step['WeatheringOutput'][key][process]

            assert (step['WeatheringOutput'][key]['floating'] <=
                    step['WeatheringOutput'][key]['amount_released'])
            # For nominal, sum up all mass and ensure it equals the mass at
            # step initialization - ignore step 0
            sum_mass += step['WeatheringOutput'][key]['floating']
            np.isclose(sum_mass,
                       step['WeatheringOutput'][key]['amount_released'])

        print('Completed step: ', step['step_num'])

    # removed last test and do the assertion here itself instead of writing to
    # file again which takes awhile!
    if output_dir is not None:
        files = glob(os.path.join(output_dir, '*.json'))
        assert len(files) == model.num_time_steps
