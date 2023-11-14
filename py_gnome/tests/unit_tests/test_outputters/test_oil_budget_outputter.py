'''
tests for oil budget outputter

WARNING: not well tested! doesn't actually test the output -- hopefully that's being tested elsewhere!
'''

import os
from glob import glob
from datetime import timedelta

import numpy as np
import pytest

from gnome.utilities.inf_datetime import InfDateTime
import gnome.scripting as gs


from gnome.environment import constant_wind, Water, Waves
from gnome.weatherers import (Evaporation,
                              NaturalDispersion,
                              Emulsification,
                              ChemicalDispersion,
                              Skimmer,
                              Burn,
                              )
from gnome.spills import surface_point_line_spill

from gnome.outputters import OilBudgetOutput

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

    model.duration = gs.days(1)
    end_time = start_time + gs.hours(1)

    spill = surface_point_line_spill(100,
                                     start_position=rel_start_pos,
                                     release_time=start_time,
                                     end_release_time=start_time + gs.hours(1),
                                     end_position=rel_end_pos,
                                     substance=test_oil,
                                     amount=1000,
                                     units='kg')
    model.spills += spill

    model.add_weathering(which='standard')

    return model


def test_init():
    'Simple initialization passes'
    g = OilBudgetOutput("a_sample_filename")


def test_bad_format():
    """
    unknown file format
    """
    with pytest.raises(ValueError):
        g = OilBudgetOutput("a_sample_filename",
                            file_format='json',
                            )


def test_model_full_run_output(model, output_dir):
    '''
    Test weathering outputter with a model since simplest to do that

    (I'm being impatient -- I hope I don't regret that)
    '''

    outfilename = os.path.join(output_dir, "test_oil_budget.csv")

    model.outputters += OilBudgetOutput(outfilename,
                                        output_timestep=gs.hours(1))

    print(OilBudgetOutput.clean_output_files)

    model.rewind()

    model.full_run()

    # was the file created?

    out_filename = os.path.join(output_dir, outfilename)
    assert os.path.isfile(out_filename)


    # read the file in and test a couple things
    csv_file = open(out_filename).readlines()

    print(len(csv_file))

    assert len(csv_file) == 26

    assert csv_file[0].split(",")[0] == "Model Time"

    assert csv_file[1].split(",")[0].strip() == "2012-09-15 12:00"
    assert csv_file[2].split(",")[0].strip() == "2012-09-15 13:00"

    assert csv_file[-1].split(",")[0].strip() == "2012-09-16 12:00"


def test_model_full_run_output_short_interval(model, output_dir):
    '''
    Test weathering outputter with a model since simplest to do that

    (I'm being impatient -- I hope I don't regret that)
    '''

    outfilename = os.path.join(output_dir, "test_oil_budget2.csv")

    model.outputters += OilBudgetOutput(outfilename,
                                        output_timestep=gs.minutes(30))


    model.rewind()

    model.full_run()

    # was the file created?

    out_filename = os.path.join(output_dir, outfilename)
    assert os.path.isfile(out_filename)


    # read the file in and test a couple things
    csv_file = open(out_filename).readlines()

    assert len(csv_file) == 50

    assert csv_file[0].split(",")[0] == "Model Time"

    assert csv_file[1].split(",")[0].strip() == "2012-09-15 12:00"
    assert csv_file[2].split(",")[0].strip() == "2012-09-15 12:30"

    assert csv_file[-1].split(",")[0].strip() == "2012-09-16 12:00"

    # for line in csv_file:
    #     print(line)
    #     print()

    # assert False
