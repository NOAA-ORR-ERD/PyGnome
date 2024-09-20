'''
tests for binary (legacy binary format) outputter
'''

import os
from pathlib import Path
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import pytest
from pytest import raises

from gnome.outputters.binary import BinaryOutput


from gnome.spills.spill import Spill, point_line_spill
from gnome.spills.release import PolygonRelease
from gnome.spill_container import SpillContainerPair
from gnome.movers import RandomMover, constant_point_wind_mover
from gnome.model import Model

# file extension to use for test output files
#  this is used by the output_filename fixture in conftest:
FILE_EXTENSION = ".bin"



def local_dirname():
    dirname = Path(__file__).parent / "output_binary"
    dirname.mkdir(exist_ok=True)
    return dirname


@pytest.fixture(scope='function')
def model(sample_model, output_filename):
    model = sample_model['model']
    rel_start_pos = sample_model['release_start_pos']
    rel_end_pos = sample_model['release_end_pos']

    model.cache_enabled = False
    model.uncertain = True

    model.spills += point_line_spill(10,
                                     start_position=rel_start_pos,
                                     release_time=model.start_time,
                                     end_position=rel_end_pos)

    model.time_step = 3600
    model.duration = timedelta(hours=2)
    model.rewind()
    return model


def test_init(output_dir):
    'simple initialization passes'
    bino = BinaryOutput(os.path.join(output_dir, 'test.bin'))

# check_filename now happens in prepare_for_model_run
# def test_init_exceptions():
#     '''
#     test exceptions raised during __init__
#     '''
#     with pytest.raises(ValueError):
#         # must be filename, not dir name
#         BinaryOutput(os.path.abspath(os.path.dirname(__file__)))
#
#     with pytest.raises(ValueError):
#         BinaryOutput('invalid_path_to_file/file.bino')

def test_exceptions(output_filename):
    spill_pair = SpillContainerPair()

    # begin tests
    bino = BinaryOutput(output_filename)
    bino.rewind()  # delete temporary files


    with raises(ValueError):
        file_path = 'invalid_path_to_file/file.bin'
        BinaryOutput(file_path).prepare_for_model_run(datetime.now(), spill_pair)

    with raises(TypeError):
        # need to pass in model start time
        bino.prepare_for_model_run(num_time_steps=4)

    with raises(TypeError):
        # need to pass in model start time and spills
        bino.prepare_for_model_run()

    with raises(ValueError):
        # need a cache object
        bino.write_output(0)

    bino.prepare_for_model_run(model_start_time=datetime.now(),
                                 spills=spill_pair,
                                 num_time_steps=4)
    bino.prepare_for_model_run(model_start_time=datetime.now(),
                                 spills=spill_pair,
                                 num_time_steps=4)

def test_timesteps(model):
    # is it written correctly? I have no idea
    # but at least it doesn't crash and creates a file
    # this is with the default to make a zip file

    # though it looks pretty empty -- somethign is wrong

    filename = local_dirname() / "multi_timesteps.bin"

    bino = BinaryOutput(filename)
    model.outputters += bino

    # run the model
    model.full_run()

    outfilename = filename.with_suffix(".zip")
    assert outfilename.is_file()




# def test_rewind(model, output_dir):
#     'test outputter with a model since simplest to do that'
#     model.rewind()
#     model.full_run()
#     files = glob(os.path.join(output_dir, '*.bino'))

#     model.rewind()

#     files = glob(os.path.join(output_dir, '*.geojson'))
#     assert len(files) == 0


# @pytest.mark.slow
# @pytest.mark.parametrize("output_ts_factor", [1, 2, 2.4])
# def test_write_output_post_run(model, output_ts_factor, output_dir):
#     model.rewind()
#     o_geojson = model.outputters[-1]
#     o_geojson.output_timestep = timedelta(seconds=model.time_step *
#                                           output_ts_factor)
#     del model.outputters[-1]

#     model.full_run()
#     files = glob(os.path.join(output_dir, '*.geojson'))
#     assert len(files) == 0

#     o_geojson.write_output_post_run(model.start_time,
#                                     model.num_time_steps,
#                                     cache=model._cache,
#                                     spills=model.spills)
#     files = glob(os.path.join(output_dir, '*.geojson'))
#     assert len(files) == int((model.num_time_steps-2)/output_ts_factor) + 2
#     o_geojson.output_timestep = None
#     model.outputters += o_geojson
