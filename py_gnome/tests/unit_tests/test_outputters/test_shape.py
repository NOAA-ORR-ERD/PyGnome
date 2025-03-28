
# tests for shapefile outputter

from datetime import datetime, timedelta
import dateutil
import geopandas as gpd
import os
from pathlib import Path
import tempfile
import zipfile

import pytest
from pytest import raises

from gnome.outputters import ShapeOutput
import gnome.scripting as gs
from gnome.spills.spill import point_line_spill
from gnome.spill_container import SpillContainerPair


# file extension to use for test output files
#  this is used by the output_filename fixture in conftest:
FILE_EXTENSION = ""

# fixme: there's also a output_dir fixture that should to the trick
HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output_shape"
OUTPUT_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope='function')
def model(sample_model_fcn, output_filename):
    model = sample_model_fcn['model']
    rel_start_pos = sample_model_fcn['release_start_pos']
    rel_end_pos = sample_model_fcn['release_end_pos']

    model.cache_enabled = True
    model.uncertain = True

    model.spills += point_line_spill(2,
                                             start_position=rel_start_pos,
                                             release_time=model.start_time,
                                             end_position=rel_end_pos)

    # Step by 1 hour, and go for 2 hours so we get a start, middle, and end.
    model.time_step = gs.minutes(60)
    model.duration = gs.hours(2)
    model.rewind()

    return model


def test_init(output_dir):
    'simple initialization passes'
    shp = ShapeOutput(output_dir / 'test')

    assert isinstance(shp, ShapeOutput)


def test_init_filename_exceptions():
    '''
    test exceptions raised for bad filenames
    '''
    spill_pair = SpillContainerPair()

    with pytest.raises(ValueError):
        # path must exist
        file_path = 'invalid_path_to_file/file.zip'
        ShapeOutput(file_path).prepare_for_model_run(datetime.now(), spill_pair)


def test_exceptions(output_filename):
    spill_pair = SpillContainerPair()
    # begin tests
    shp = ShapeOutput(output_filename)
    shp.rewind()  # delete temporary files

    with raises(TypeError):
        # need to pass in model start time
        shp.prepare_for_model_run(num_time_steps=4)

    with raises(TypeError):
        # need to pass in model start time and spills
        shp.prepare_for_model_run()

    with raises(ValueError):
        # need a cache object
        shp.write_output(0)

    # For this unit test, there should be no exception if we do it twice.
    shp.prepare_for_model_run(model_start_time=datetime.now(),
                              spills=spill_pair,
                              num_time_steps=4)
    shp.prepare_for_model_run(model_start_time=datetime.now(),
                              spills=spill_pair,
                              num_time_steps=4)


# Take a shapefile zipfile, and unpack the shapefile and return the data
def get_shape_details(filename):
    gpd_shapefile = None
    with tempfile.TemporaryDirectory() as tempdir:
        with zipfile.ZipFile(filename, 'r') as zipper:
            namelist = zipper.namelist()
            # Find the shapefile we are interested in...
            shapefiles = [f for f in zipper.namelist() if f.split('.')[-1] == 'shp']
            # Make sure we have just one .shp
            assert len(shapefiles) == 1
            # Extract to a temp dir so we can look at the shapefile
            zipper.extractall(tempdir)
            shp_path = os.path.join(tempdir, shapefiles[0])
            gpd_shapefile = gpd.read_file(shp_path, engine="pyogrio")
    return gpd_shapefile

# This is handy so as not to have to repeat code in tests.
#  But it should be integrated with the above -- and maybe more features added?
def get_shape_file_stats(filename):
    """
    reads a shape file as output by GNOME, and reports some stats

    :param filename: filename or geopandas dataframe
    :type filename: pathlike or geopandas dataframe

    :returns: dict of info: keys: 'total_points', 'timesteps'
    """

    if isinstance(filename, gpd.geodataframe.GeoDataFrame):
        gpd_shapefile = filename
    else:
        gpd_shapefile = gpd.read_file(filename, engine="pyogrio")

    info = {}
    info['total_points'] = len(gpd_shapefile)
    try:
        # I'm sure there's a pandas way to do this ...
        times = {time for time in gpd_shapefile['Time']}
        info['timesteps'] = sorted(gs.asdatetime(time) for time in times)
    except KeyError:
        # no times
        info['timesteps'] = []

    print(f"{info=}")
    return info


def test_multi_timesteps(model, output_dir):
    filename = output_dir / "multi_timesteps.zip"

    shp = ShapeOutput(filename, include_uncertain_boundary=True,
                      include_certain_boundary=True)
    model.outputters += shp

    #run the model
    model.full_run()

    # Since this will contain 4 shapefiles
    # we must first unpack those
    gpd_shapefile_certain = gpd_shapefile_uncertain = None
    with tempfile.TemporaryDirectory() as tempdir:
        with zipfile.ZipFile(filename, 'r') as zipper:
            namelist = zipper.namelist()
            # Find the zips
            zfiles = [f for f in zipper.namelist() if f.split('.')[-1] == 'zip']
            # Make sure we have 4 .zip... one certain, one uncertain,
            # one certain bounds, one uncertain bounds
            assert len(zfiles) == 4
            # Extract to a temp dir so we can look at the shapefile
            zipper.extractall(tempdir)
            # Now loop through them all and make sure they look good
            for zfile in zfiles:
                shape_details = get_shape_details(os.path.join(tempdir, zfile))
                if 'uncertain_boundary' in zfile:
                    assert shape_details.geom_type[0] in ['Polygon', 'MultiPolygon']
                    assert shape_details.crs.to_epsg() == 4326
                    # Only certain keys are required... others are optional
                    assert len(shape_details.keys()) >= 2
                    # Test the required ones are there
                    for key in ['time', 'geometry']:
                        assert key in shape_details.keys()
                    # For this particular test run, we expect 3 polygons
                    assert len(shape_details['geometry']) == 3
                elif 'uncertain' in zfile:
                    assert shape_details.geom_type[0] in ['Point', 'MultiPoint']
                    assert shape_details.crs.to_epsg() == 4326
                    # Only certain keys are required... others are optional
                    assert len(shape_details.keys()) >= 2
                    # Test the required ones are there
                    for key in ['LE_id', 'Spill_id', 'Depth', 'Mass', 'Age', 'StatusCode',
                                'Time', 'geometry']:
                        assert key in shape_details.keys()
                    # For this particular test run, we expect 6 points
                    assert len(shape_details['geometry']) == 6
                elif 'certain_boundary' in zfile:
                    assert shape_details.geom_type[0] in ['Polygon', 'MultiPolygon']
                    assert shape_details.crs.to_epsg() == 4326
                    # Only certain keys are required... others are optional
                    assert len(shape_details.keys()) >= 2
                    # Test the required ones are there
                    for key in ['time', 'geometry']:
                        assert key in shape_details.keys()
                    # For this particular test run, we expect 3 polygons
                    assert len(shape_details['geometry']) == 3
                elif 'certain' in zfile:
                    assert shape_details.geom_type[0] in ['Point', 'MultiPoint']
                    assert shape_details.crs.to_epsg() == 4326
                    # Only certain keys are required... others are optional
                    assert len(shape_details.keys()) >= 2
                    # Test the required ones are there
                    for key in ['LE_id', 'Spill_id', 'Depth', 'Mass', 'Age', 'StatusCode',
                                'Time', 'geometry']:
                        assert key in shape_details.keys()
                    # For this particular test run, we expect 6 points
                    assert len(shape_details['geometry']) == 6


def test_singlestep_model_start(model, output_dir):
    """
    Not a very useful case, but should work.

    Currently fails... but needs resolution
       # 3) output_single_step = True and output_zero_step and output_last_step
       #    set to False.  Should just output single set of points for the model
       #    start time.
    """
    filename = output_dir / "single_step_model_start.zip"
    # Just look at certain for this test...
    model.uncertain = False
    # model_step = timedelta(seconds=model.time_step)
    # random_step = timedelta(seconds=7)

    model.rewind()
    shp = ShapeOutput(filename,
                      output_single_step=True,
                      output_zero_step=False,
                      output_last_step=False)
    model.outputters += shp
    # run the model
    model.full_run()
    # Open up the output and verify we have a shapefile with the
    # right results (i.e. number of geometries etc)
    gpd_shapefile = get_shape_details(filename)
    stats = get_shape_file_stats(gpd_shapefile)
    # # For this particular test run, we expect 2 particles
    # assert stats['total_points'] == 2
    # Make sure the output times look correct
    assert len(stats['timesteps']) == 1
    assert stats['timesteps'][0] == model.start_time


def test_singlestep(model, output_dir):
    filename = output_dir / "single_step.zip"
    # Just look at certain for this test...
    model.uncertain = False
    model_step = timedelta(seconds=model.time_step)
    random_step = timedelta(seconds=7)

    # 1) output_single_step = True, others default (i.e. output_zero_step
    #    and output_last_step set to True).  In this case we should get
    #    the first and last steps because they are written regardless, and
    #    the default output_start_time is the initial model output time.
    #    We should get two sets of points (4 total).
    #model.rewind()
    shp = ShapeOutput(filename, output_single_step=True)
    model.outputters += shp
    #run the model
    model.full_run()
    # Open up the output and verify we have a shapefile with the
    # right results (i.e. number of geometries etc)
    gpd_shapefile = get_shape_details(filename)
    # Now check that things look good with the data
    # These are only checked in the first case... since they should only
    # need to be verified once.
    assert gpd_shapefile.geom_type[0] == 'Point'
    assert gpd_shapefile.crs.to_epsg() == 4326
    assert len(gpd_shapefile.keys()) >= 8
    for key in ['LE_id', 'Spill_id', 'Depth', 'Mass', 'Age', 'StatusCode',
                'Time', 'geometry']:
        assert key in gpd_shapefile.keys()
    # For this particular test run, we expect 4 particles
    assert len(gpd_shapefile['geometry']) == 4
    # Make sure the first point is at start_time... and last point is at end time
    assert dateutil.parser.parse(gpd_shapefile['Time'][0]) == model.start_time
    assert dateutil.parser.parse(gpd_shapefile['Time'][3]) == model.start_time+model.duration

    # 2) output_single_step = True, others default (i.e. output_zero_step
    #    and output_last_step set to True) but also set output_start_time to
    #    the middle timestep.  We should get three sets of points (6 total).
    model.rewind()
    shp = ShapeOutput(filename, output_single_step=True,
                      output_start_time=model.start_time+model_step)
    model.outputters += shp
    #run the model
    model.full_run()
    # Open up the output and verify we have a shapefile with the
    # right results (i.e. number of geometries etc)
    gpd_shapefile = get_shape_details(filename)
    # For this particular test run, we expect 6 particles
    assert len(gpd_shapefile['geometry']) == 6
    # Make sure the point times look correct
    assert dateutil.parser.parse(gpd_shapefile['Time'][0]) == model.start_time
    assert dateutil.parser.parse(gpd_shapefile['Time'][2]) == model.start_time+model_step
    assert dateutil.parser.parse(gpd_shapefile['Time'][5]) == model.start_time+model.duration

# Currently fails... but needs resolution
#    # 3) output_single_step = True and output_zero_step and output_last_step
#    #    set to False.  Should just output single set of points for the model
#    #    start time.
#    model.rewind()
#    shp = ShapeOutput(filename, output_single_step=True,
#                      output_zero_step=False, output_last_step=False)
#    model.outputters += shp
#    #run the model
#    model.full_run()
#    # Open up the output and verify we have a shapefile with the
#    # right results (i.e. number of geometries etc)
#    gpd_shapefile = get_shape_details(filename)
#    # For this particular test run, we expect 2 particles
#    assert len(gpd_shapefile['geometry']) == 2
#    # Make sure the point times look correct
#    assert dateutil.parser.parse(gpd_shapefile['Time'][0]) == model.start_time

    # 4) output_single_step = True and output_zero_step and output_last_step
    #    set to False.  Set output_start_time to end time.  This should just
    #    output single set of points for the model end time.
    model.rewind()
    shp = ShapeOutput(filename, output_single_step=True,
                      output_zero_step=False, output_last_step=False,
                      output_start_time=model.start_time+model.duration)
    model.outputters += shp
    #run the model
    model.full_run()
    # Open up the output and verify we have a shapefile with the
    # right results (i.e. number of geometries etc)
    gpd_shapefile = get_shape_details(filename)
    # For this particular test run, we expect 2 particles
    assert len(gpd_shapefile['geometry']) == 2
    # Make sure the point times look correct
    assert dateutil.parser.parse(gpd_shapefile['Time'][0]) == model.start_time+model.duration

    # 5) output_single_step = True and output_zero_step and output_last_step
    #    set to False.  Set output_start_time to middle time.  This should just
    #    output single set of points for the model middle output time.
    model.rewind()
    shp = ShapeOutput(filename, output_single_step=True,
                      output_zero_step=False, output_last_step=False,
                      output_start_time=model.start_time+model_step)
    model.outputters += shp
    #run the model
    model.full_run()
    # Open up the output and verify we have a shapefile with the
    # right results (i.e. number of geometries etc)
    gpd_shapefile = get_shape_details(filename)
    # For this particular test run, we expect 2 particles
    assert len(gpd_shapefile['geometry']) == 2
    # Make sure the point times look correct
    assert dateutil.parser.parse(gpd_shapefile['Time'][0]) == model.start_time+model_step


def test_nozip(model, output_dir):
    filename = output_dir / "shapefile_no_zip.shp"

    shp = ShapeOutput(filename, zip_output=False)
    model.outputters += shp
    # Uncertain forces zip... so we dont want that for this test
    model.uncertain = False

    # run the model
    model.full_run()
    files = filename.parent.glob(filename.with_suffix('.*').name)
    assert len(list(files)) == 5
    for f in files:
        assert f.suffix in ['.cpg', '.dbf', '.prj', '.shp', '.shx']


def test_timesteps2(model, output_dir):
    """
    Probably a duplicate test -- but I had already written it.
     - CHB
    """
    filename = output_dir / "multi_timesteps2"
    print(filename)
    model.uncertain = False
    model.time_step = gs.minutes(15)
    # two hour run, 30 min output -- total 5 timesteps output (first and last)
    shp = ShapeOutput(filename,
                      output_timestep=gs.minutes(30),
                      output_zero_step=True,
                      output_single_step=False,
                      )
    print(f"{model.outputters=}")
    model.outputters += shp
    print(f"{model.outputters[0].output_timestep=}")
    print(f"{type(model.outputters[0].output_timestep)=}")
    print(f"{model.outputters=}")
    print(f"{model.duration=}")
    print(f"{model.time_step=}")

    # run the model
    model.full_run()

    # check the shapefile
    results = get_shape_file_stats(filename.with_suffix('.zip'))

    print(model.start_time)
    print(results['timesteps'])
    # breakpoint()

    assert len(results['timesteps']) == 5
    assert gs.asdatetime(results['timesteps'][0]) == model.start_time
    assert gs.asdatetime(results['timesteps'][-1]) == model.start_time + model.duration
    assert results['total_points'] == 10

    # assert False


@pytest.mark.xfail
# NOTE: This currently fails because the model isn't
#       calling post_model_run after a failure -- it's just crashing out.
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
    model.uncertain = False
    shp = ShapeOutput(filename,
                      output_timestep=gs.hours(1),
                      output_zero_step=False,
                      output_last_step=False,
                      output_single_step=False,
                      output_start_time=model.start_time,
                      )
    model.outputters += shp

    print(model.movers)
    # run the model
    model.full_run()

    # check the shapefile
    results = get_shape_file_stats(filename.with_suffix('.zip'))

    assert len(results['timesteps']) == 1
    assert gs.asdatetime(results['timesteps'][0]) == model.start_time + gs.hours(1)
