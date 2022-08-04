'''
tests for kmz outputter
'''

import os
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import pytest
from pytest import raises

from gnome.outputters import KMZOutput
from gnome.outputters import kmz_templates


from gnome.spills import PolygonRelease, Spill, surface_point_line_spill
from gnome.spill_container import SpillContainerPair
from gnome.movers import RandomMover, constant_point_wind_mover
from gnome.model import Model

# file extension to use for test output files
#  this is used by the output_filename fixture in conftest:
FILE_EXTENSION = ".kmz"


def local_dirname():
    dirname = os.path.split(__file__)[0]
    dirname = os.path.join(dirname, "output_kmz")
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    return dirname


@pytest.fixture(scope='function')
def model(sample_model, output_filename):
    model = sample_model['model']
    rel_start_pos = sample_model['release_start_pos']
    rel_end_pos = sample_model['release_end_pos']

    model.cache_enabled = True
    model.uncertain = True

    model.spills += surface_point_line_spill(2,
                                             start_position=rel_start_pos,
                                             release_time=model.start_time,
                                             end_position=rel_end_pos)

    model.time_step = 3600
    model.duration = timedelta(hours=1)
    model.rewind()

    return model


def test_init(output_dir):
    'simple initialization passes'
    kmz = KMZOutput(os.path.join(output_dir, 'test.kmz'))

def test_init_exceptions():
    '''
    test exceptions raised during __init__
    '''
    with pytest.raises(ValueError):
        # must be filename, not dir name
        KMZOutput(os.path.abspath(os.path.dirname(__file__)))

    with pytest.raises(ValueError):
        KMZOutput('invalid_path_to_file/file.kmz')

def test_exceptions(output_filename):
    spill_pair = SpillContainerPair()

    # begin tests
    kmz = KMZOutput(output_filename)
    kmz.rewind()  # delete temporary files

    with raises(TypeError):
        # need to pass in model start time
        kmz.prepare_for_model_run(num_time_steps=4)

    with raises(TypeError):
        # need to pass in model start time and spills
        kmz.prepare_for_model_run()

    with raises(ValueError):
        # need a cache object
        kmz.write_output(0)

#    Maybe add ability to specify which data later on..
#    with raises(ValueError):
#        kmz.which_data = 'some random string'

    # changed renderer and netcdf ouputter to delete old files in
    # prepare_for_model_run() rather than rewind()
    # -- rewind() was getting called a lot
    # -- before there was time to change the output file names, etc.
    # So for this unit test, there should be no exception if we do it twice.
    kmz.prepare_for_model_run(model_start_time=datetime.now(),
                                 spills=spill_pair,
                                 num_time_steps=4)
    kmz.prepare_for_model_run(model_start_time=datetime.now(),
                                 spills=spill_pair,
                                 num_time_steps=4)


    # with raises(AttributeError):
    #     'cannot change after prepare_for_model_run has been called'
    #     kmz.which_data = 'most'

def test_timesteps(model):
    filename = os.path.join(local_dirname(), "multi_timesteps.kml")

    kmz = KMZOutput(filename)
    model.outputters += kmz

    #run the model
    model.full_run()



## test the kml templates
def test_element_template():
    template = kmz_templates.point_template

    formatted =  template.format(-24.34, 43.2)

    assert formatted == """             <Point>
                     <altitudeMode>relativeToGround</altitudeMode>
                     <coordinates>-24.340000,43.200000,1.000000</coordinates>
             </Point>
"""

def test_on_timestep_kml():
    floating_positions = [ (23.45, 45.2, 0),
                           (-13.45,12.2, 0),
    ]
    beached_positions = [ (-23.45, 145.2, 0),
                           (13.45, -12.2, 0),
                           (44.3, -23.1,  0)
    ]

    kml = kmz_templates.build_one_timestep(floating_positions,
                                           beached_positions,
                                           '2015-10-23T14:00:00',
                                           '2015-10-23T15:00:00',
                                           uncertain=True,
                                           )
    assert True






# def test_rewind(model, output_dir):
#     'test outputter with a model since simplest to do that'
#     model.rewind()
#     model.full_run()
#     files = glob(os.path.join(output_dir, '*.kmz'))

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
