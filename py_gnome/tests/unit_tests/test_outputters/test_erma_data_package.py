
# tests for ERMA Data Package outputter

from datetime import datetime, timedelta
import geopandas as gpd
import io
import json
import os
import pathlib
import pytest
import tempfile
import zipfile

from gnome.outputters import ERMADataPackageOutput
import gnome.scripting as gs
from gnome.spills.spill import point_line_spill
from gnome.spill_container import SpillContainerPair

from .conftest import count_files_in_zip

# file extension to use for test output files
#  this is used by the output_filename fixture in conftest:
FILE_EXTENSION = ""


@pytest.fixture(scope='function')
def model(sample_model_fcn, output_filename):
    model = sample_model_fcn['model']
    rel_start_pos = sample_model_fcn['release_start_pos']
    rel_end_pos = sample_model_fcn['release_end_pos']

    model.cache_enabled = True
    model.uncertain = True

    model.spills += point_line_spill(10,
                                             start_position=rel_start_pos,
                                             release_time=model.start_time,
                                             end_position=rel_end_pos)
    model.time_step = gs.minutes(60)
    model.duration = gs.hours(10)
    model.rewind()
    return model


def test_init(output_dir):
    'simple initialization passes'
    package = ERMADataPackageOutput(os.path.join(output_dir, 'erma_data_package_test.zip'))

def test_exceptions(output_filename):
    spill_pair = SpillContainerPair()

    # begin tests
    package = ERMADataPackageOutput(output_filename)
    package.rewind()

    with pytest.raises(ValueError):
        # need a cache object
        package.write_output(0)

def make_sure_filename_is_valid(filename, package):
    # If we started asking for a zip... they filenames should match exactly
    if pathlib.Path(filename).suffix == '.zip':
        assert filename == str(package.filename)
    # Make sure the stems are the same
    assert pathlib.Path(filename).stem == package.filename.stem
    assert pathlib.Path(filename).parent == package.filename.parent
    # Assert that the package ALWAYS comes back as a zip
    assert package.filename.suffix == '.zip'
    # Assert that the package was created
    assert os.path.exists(package.filename)

def test_filenames(model, output_dir, output_filename):
    # First test normal test output_dir and output_filename combo
    filename = os.path.join(output_dir, output_filename)
    package1 = ERMADataPackageOutput(filename)
    model.outputters += package1
    # Run the model
    model.full_run()
    make_sure_filename_is_valid(filename, package1)
    model.rewind()

    # Lets make sure if you dont pass an ext, it still gets set
    filename = os.path.join(output_dir, "file_no_ext")
    package2 = ERMADataPackageOutput(filename)
    model.outputters.replace(package1.id, package2)
    model.full_run()
    make_sure_filename_is_valid(filename, package2)
    model.rewind()

    # Lets make sure if we pass in a junk ext, it still gets set correctly
    filename = os.path.join(output_dir, "file_junk_ext.foo")
    package3 = ERMADataPackageOutput(filename)
    model.outputters.replace(package2.id, package3)
    model.full_run()
    make_sure_filename_is_valid(filename, package3)
    model.rewind()

def test_simple_package(model, output_dir):
    filename = os.path.join(output_dir, "simple_erma_data_package.zip")
    package = ERMADataPackageOutput(filename)
    model.outputters += package
    # Run the model
    model.full_run()

    # Test the filename
    make_sure_filename_is_valid(filename, package)

    # Now lets look inside.  We should see the expected directory structure
    # and files.  Since we have uncertainty on, we need to see two
    with zipfile.ZipFile(filename, 'r') as zipper:
        namelist = zipper.namelist()
        # Check the source shapefile
        certain_path = 'source_files/simple_erma_data_package_certain.zip'
        assert certain_path in namelist
        uncertain_path = 'source_files/simple_erma_data_package_uncertain.zip'
        assert uncertain_path in namelist
        # Check the layer files
        layer1_json_path = 'layers/1.json'
        assert layer1_json_path in namelist
        layer2_json_path = 'layers/2.json'
        assert layer2_json_path in namelist
        # Check that there is an empty attachment directory
        assert 'attachment/' in namelist

        # Now lets look at the shapefiles to make sure they look valid
        for shp_path in [certain_path, uncertain_path]:
            # A new temp dir for each inner zip... tempfile will clean up for us
            with tempfile.TemporaryDirectory() as tempdir:
                # Open the shapefile zip inside the original zip
                with zipper.open(shp_path) as inner_zip:
                    # Grab the data... and open that zip
                    zip_data =  io.BytesIO(inner_zip.read())
                    with zipfile.ZipFile(zip_data) as shp_zip:
                        # Find the shapefile we are interested in...
                        shapefiles = [f for f in shp_zip.namelist() if f.split('.')[-1] == 'shp']
                        # Make sure we have just one .shp
                        assert len(shapefiles) == 1
                        # Extract to a temp dir so we can look at the shapefile
                        shp_zip.extractall(tempdir)
                        shp_path = os.path.join(tempdir, shapefiles[0])
                        gpd_shapefile = gpd.read_file(shp_path, engine="pyogrio")
                        # Now check that things look good with the data
                        assert (gpd_shapefile.geom_type[0] == 'Point' or
                                gpd_shapefile.geom_type[0] == 'Polygon')
                        assert gpd_shapefile.crs.to_epsg() == 4326
                        if gpd_shapefile.geom_type[0] == 'Point':
                            assert len(gpd_shapefile.keys()) >= 8
                        if gpd_shapefile.geom_type[0] == 'Polygon':
                            assert len(gpd_shapefile.keys()) >= 2
                        # For this particular test run, we have 10 particles
                        # for 11 timesteps, but get some beached, so we just check
                        # that we are over 100
                        if gpd_shapefile.geom_type[0] == 'Point':
                            assert len(gpd_shapefile['geometry']) > 100
                        if gpd_shapefile.geom_type[0] == 'Polygon':
                            # We just expect one boundary polygon
                            assert len(gpd_shapefile['geometry']) > 1
                        # Make sure the keys look good
                        if gpd_shapefile.geom_type[0] == 'Point':
                            for key in ['LE_id', 'Spill_id', 'Depth', 'Mass', 'Age', 'StatusCode',
                                        'Time', 'geometry']:
                                assert key in gpd_shapefile.keys()
                        if gpd_shapefile.geom_type[0] == 'Polygon':
                            for key in ['time', 'geometry']:
                                assert key in gpd_shapefile.keys()
        # Look inside the json files as well...
        # layer1 is certain, layer2 is uncertain
        with zipper.open(layer1_json_path) as layer1_json_file:
            certain_json =  json.load(layer1_json_file)
            # Need to come up with some good validation of the json here
        with zipper.open(layer2_json_path) as layer2_json_file:
            uncertain_json =  json.load(layer2_json_file)
            # Need to come up with some good validation of the json here


#@pytest.mark.xfail
# NOTE: This currently fails because the model isn't allowing partial runs to output
def test_model_stops_in_middle(model, output_dir):
    filename = os.path.join(output_dir, "stop_in_middle.zip")
    # set up a WindMover that's too short.
    times = [model.start_time + (gs.minutes(30) * i) for i in range(3)]
    # long enough record
    # times = [model.start_time + (gs.minutes(30) * i) for i in range(21)]

    winds = gs.wind_from_values([(dt, 5, 90) for dt in times])

    model.movers += gs.WindMover(winds)

    package = ERMADataPackageOutput(filename)
    model.outputters += package

    print(model.movers)
    # Run the model
    with pytest.raises(Exception):
        model.full_run()

    # check the zipfile has expected number of files
    len_zip = count_files_in_zip(filename)
    assert len_zip == 11
