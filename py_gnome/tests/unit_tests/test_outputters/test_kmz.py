'''
tests for kmz outputter
'''

import os
# from glob import glob
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gnome.outputters import KMZOutput
from gnome.outputters import kmz_templates
import gnome.scripting as gs
import zipfile


from gnome.spills.spill import Spill, point_line_spill
from gnome.spills.release import PolygonRelease
from gnome.spill_container import SpillContainerPair
from gnome.movers import RandomMover, constant_point_wind_mover
from gnome.model import Model

# file extension to use for test output files
#  this is used by the output_filename fixture in conftest:
FILE_EXTENSION = ".kmz"

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output_kmz"
OUTPUT_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope='function')
def model(sample_model, output_filename):
    model = sample_model['model']
    rel_start_pos = sample_model['release_start_pos']
    rel_end_pos = sample_model['release_end_pos']

    model.cache_enabled = True
    model.uncertain = True

    model.spills += point_line_spill(2,
                                     start_position=rel_start_pos,
                                     release_time=model.start_time,
                                     end_position=rel_end_pos)

    model.time_step = 3600
    model.duration = timedelta(hours=1)
    model.rewind()

    return model


def test_init(output_dir):
    'simple initialization passes'
    kmz = KMZOutput(output_dir / 'test.kmz')

# check_filename now happens in prepare_for_model_run
# def test_init_exceptions():
#     '''
#     test exceptions raised during __init__
#     '''
#     with pytest.raises(ValueError):
#         # must be filename, not dir name
#         KMZOutput(os.path.abspath(os.path.dirname(__file__)))
#
#     with pytest.raises(ValueError):
#         KMZOutput('invalid_path_to_file/file.kmz')

def test_exceptions(output_filename):
    spill_pair = SpillContainerPair()

    # begin tests
    kmz = KMZOutput(output_filename)
    kmz.rewind()  # delete temporary files

#     this test is now moot since kmz extension is added to the filename on init
#     with pytest.raises(ValueError):
#         # must be filename, not dir name
#         file_path = os.path.abspath(os.path.dirname(__file__))
#         KMZOutput(file_path).prepare_for_model_run(datetime.now(), spill_pair)

    with pytest.raises(ValueError):
        file_path = 'invalid_path_to_file/file.kmz'
        KMZOutput(file_path).prepare_for_model_run(datetime.now(), spill_pair)

    with pytest.raises(TypeError):
        # need to pass in model start time
        kmz.prepare_for_model_run(num_time_steps=4)

    with pytest.raises(TypeError):
        # need to pass in model start time and spills
        kmz.prepare_for_model_run()

    with pytest.raises(ValueError):
        # need a cache object
        kmz.write_output(0)

#    Maybe add ability to specify which data later on..
#    with pytest.raises(ValueError):
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


    # with pytest.raises(AttributeError):
    #     'cannot change after prepare_for_model_run has been called'
    #     kmz.which_data = 'most'

def test_timesteps(model):
    filename = OUTPUT_DIR / "multi_timesteps.kml"

    kmz = KMZOutput(filename)
    model.outputters += kmz

    #run the model
    for step in model:
        print(step)
        # making sure -- this was a bug at one point
        assert isinstance(step['KMZOutput']['output_filename'], str)


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

def test_serialize():
    """
    Can we serialize / deserialize a kmz outputter?
    """
    kmzo = KMZOutput(Path(__file__).parent / "example_kmz.kmz")

    print(repr(kmzo.filename))

    json = kmzo.serialize()

    print(json)

    {
        'obj_type':
        'gnome.outputters.kmz.KMZOutput',
        'id':
        '414ce584-0480-11f0-bcb0-acde48001122',
        'name':
        'KMZOutput_5',
        'on':
        True,
        'output_zero_step':
        True,
        'output_last_step':
        True,
        'output_single_step':
        False,
        'filename':
        '/Users/chris.barker/Hazmat/GitLab/pygnome/py_gnome/tests/unit_tests/test_outputters/example_kmz.kmz'
    }

    kmzo2 = KMZOutput.deserialize(json)

    assert kmzo == kmzo2


def count_files_in_zip(zip_filepath):
    """
    Counts the number of files in a ZIP archive.

    Args:
        zip_filepath (str): The path to the ZIP file.

    Returns:
        int: The number of files in the ZIP archive.
             Returns -1 if the file is not found or is not a valid ZIP file.
    """
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_file:
            return len(zip_file.namelist())
    except FileNotFoundError:
        print(f"Error: File not found: {zip_filepath}")
        return -1
    except zipfile.BadZipFile:
         print(f"Error: Not a valid ZIP file: {zip_filepath}")
         return -1

@pytest.mark.xfail
# NOTE: This currently fails because the model isn't allowing partial runs to output
def test_model_stops_in_middle(model):
    """
    If the model stops in the middle of a run:
    e.g. runs out of data, it should still output results.
    """
    filename = OUTPUT_DIR / "stop_in_middle"

    # set up a WindMover that's too short.
    times = [model.start_time + (gs.minutes(30) * i) for i in range(3)]
    # long enough record
    # times = [model.start_time + (gs.minutes(30) * i) for i in range(5)]

    winds = gs.wind_from_values([(dt, 5, 90) for dt in times])

    model.movers += gs.WindMover(winds)
    kmz = KMZOutput(filename,
                      output_timestep=gs.hours(1),
                      output_zero_step=False,
                      output_last_step=False,
                      output_single_step=False,
                      output_start_time=model.start_time,
                      )
    model.outputters += kmz

    print(model.movers)
    # run the model
    model.full_run()

    # check the zipfile has certain and uncertain for 1 output
    len_zip = count_files_in_zip(filename.with_suffix('.kmz'))
    assert len_zip == 3	# two icons and kml, just check it got created


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
