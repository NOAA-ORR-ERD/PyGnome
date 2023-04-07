'''
test functionality of the save_load module used to persist save files
'''

import os
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED

from gnome.utilities.time_utils import sec_to_date
from gnome.utilities.inf_datetime import InfDateTime

from gnome.persist import (References, is_savezip_valid)
from gnome.gnomeobject import class_from_objtype

from gnome.environment import (Wind,
                               Tide,
                               Water,
                               constant_wind,
                               GridCurrent)

from gnome.movers import (constant_point_wind_mover,
                          SimpleMover,
                          RandomMover,
                          RandomMover3D,
                          PointWindMover,
                          CatsMover,
                          ComponentMover,
                          CurrentCycleMover,
                          c_GridCurrentMover,
                          WindMover,
                          CurrentMover,
                          c_GridWindMover)

from gnome.weatherers import (Evaporation,
                              Skimmer,
                              Burn,
                              ChemicalDispersion)

from gnome.outputters import (Renderer,
                              WeatheringOutput,
                              NetCDFOutput,
                              TrajectoryGeoJsonOutput)

from gnome.maps import MapFromBNA
from gnome.spills.spill import (surface_point_line_spill,
                                PointLineRelease,
                                )

from gnome.spills.substance import Substance, NonWeatheringSubstance
from gnome.spills.gnome_oil import GnomeOil


# following is modified for testing only
from gnome.persist import save_load

from ..conftest import testdata

import pytest
from testfixtures import LogCapture
from ..conftest import test_oil


def test_warning_logged():
    '''
    warning is logged if we try to get a class from 'obj_type' that is not
    in the gnome namespace
    '''
    with LogCapture() as lc:
        with pytest.raises(AttributeError):
            class_from_objtype('os.path')

        lc.check(('gnome.gnomeobject',
                  'WARNING',
                  'os.path is not part of gnome namespace'))


def test_class_from_objtype():
    '''
    test that correct class is returned by class_from_objtype
    '''
    cls = class_from_objtype('gnome.movers.PointWindMover')
    assert cls is PointWindMover


def test_exceptions():
    a = 1
    refs = References()
    refs.reference(a, 'a')
    refs.reference(a, 'a')  # should not do anything

    assert refs.retrieve('a') is a

    with pytest.raises(ValueError):
        refs.reference(a, 'b')

    with pytest.raises(ValueError):
        refs.reference(2, 'a')


def test_reference_object():
    '''
    get a reference to an object,then retrieve the object by reference
    '''
    a = 1
    refs = References()
    r1 = refs.reference(a)
    obj = refs.retrieve(r1)

    assert obj is a

    r2 = refs.reference(a)

    assert r2 == r1


def test_gnome_obj_reference():
    '''
    create two equal but different objects and make sure a new reference is
    created for each
    '''
    objs = [constant_point_wind_mover(0, 0) for _i in range(2)]
    assert objs[0] is not objs[1]

    refs = References()
    r_objs = [refs.reference(item) for item in objs]
    assert len(r_objs) == len(objs)
    assert r_objs[0] != r_objs[1]

    for ix, ref in enumerate(r_objs):
        assert refs.retrieve(ref) is objs[ix]
        assert objs[ix] in refs   # double check __contains__

    unknown = constant_point_wind_mover(0, 0)
    assert unknown not in refs  # check __contains__


'''
Run the following save/load test on multiple pygnome objects so collect tests
here and parametrize it by the objects
'''


base_dir = os.path.dirname(__file__)


# For PointWindMover test_save_load in test_wind_mover
g_objects = (
    GridCurrent.from_netCDF(testdata['c_GridCurrentMover']['curr_tri']),
    Tide(testdata['CatsMover']['tide']),
    # Wind(filename=testdata['ComponentMover']['wind']),
    constant_wind(5., 270, 'knots'),
    Wind(timeseries=(sec_to_date(24 * 60 * 60),
                     (0, 0)), units='mps'),
    Water(temperature=273),

    RandomMover(),
    CatsMover(testdata['CatsMover']['curr']),
    CatsMover(testdata['CatsMover']['curr'],
              tide=Tide(testdata['CatsMover']['tide'])),
    ComponentMover(testdata['ComponentMover']['curr']),
    ComponentMover(testdata['ComponentMover']['curr'],
                   wind=constant_wind(5., 270, 'knots')),
                   # wind=Wind(filename=testdata['ComponentMover']['wind'])),
     WindMover(testdata['c_GridWindMover']['wind_rect']),
     #WindMover(testdata['c_GridWindMover']['wind_curv']), #variable names wrong
     CurrentMover(testdata['c_GridCurrentMover']['curr_tri']),
     CurrentMover(testdata['c_GridCurrentMover']['curr_reg']),
    RandomMover3D(),
    SimpleMover(velocity=(10.0, 10.0, 0.0)),

    MapFromBNA(testdata['MapFromBNA']['testmap'], 6),
    NetCDFOutput(os.path.join(base_dir, 'xtemp.nc')), Renderer(testdata['Renderer']['bna_sample'],
             os.path.join(base_dir, 'output_dir')),
    WeatheringOutput(),
    PointLineRelease(release_time=datetime.now(),
                           num_elements=10,
                           start_position=(0, 0, 0)),
    surface_point_line_spill(10, (0, 0, 0), datetime.now()),
    Substance(windage_range=(0.05, 0.07)),
    GnomeOil(test_oil, windage_range=(0.05, 0.07)),
    NonWeatheringSubstance(windage_range=(0.05, 0.07)),
    Skimmer(amount=100, efficiency=0.3, active_range=(datetime(2014, 1, 1, 0, 0), datetime(2014, 1, 1, 4, 0)), units='kg'),
    Burn(area=100, thickness=1, active_range=(datetime(2014, 1, 1, 0, 0), datetime(2014, 1, 1, 4, 0)),
                    efficiency=.9),
    ChemicalDispersion(fraction_sprayed=.2, active_range=(datetime(2014, 1, 1, 0, 0), datetime(2014, 1, 1, 4, 0)),
                                  efficiency=.3),
    # todo: ask Caitlin how to fix
    # movers.RiseVelocityMover(),
    # todo: This is incomplete - no _schema for
    #       PolygonRelease, GeoJson
    # spill.PolygonRelease(datetime.now(), ((0, 0, 0), (1, 2, 0))),
    TrajectoryGeoJsonOutput(),
)


@pytest.mark.parametrize("obj", g_objects)
def test_serial_deserial(saveloc_, obj):
    'test save/load functionality'
    print("********")
    print("Checking:", type(obj))
    print("********")
    json_ = obj.serialize()
    obj2 = obj.__class__.deserialize(json_)

    assert obj == obj2


@pytest.mark.parametrize("obj", g_objects)
def test_save_load_fun(saveloc_, obj):
    'test save/load functionality'
    print("********")
    print("Checking:", type(obj))
    print("********")

    _json_, zipfile_, _refs = obj.save(saveloc_)
    obj2 = obj.__class__.load(zipfile_)

    assert obj == obj2


# Wind objects need more work - the data is being written out to file in
# 'r-theta' format accurate to 2 decimal places and in knots.
# All the conversions mean the allclose() check on timeseries fails - xfail
# for now.  When loading for a file it works fine, no decimal places stored
# in file for magnitude
@pytest.mark.parametrize("obj",
                         (Wind(timeseries=(sec_to_date(24 * 3600), (1, 30)),
                               units='meters per second'),
                          Evaporation(wind=constant_wind(1., 30.),
                                      water=Water(333.0)),)
                         )
def test_serialize_deserialize_wind_objs(saveloc_, obj):
    'test serialize/deserialize functionality'
    json_ = obj.serialize()
    obj2 = obj.__class__.deserialize(json_)

    assert obj == obj2


@pytest.mark.parametrize("obj",
                         (Wind(timeseries=(sec_to_date(24 * 3600), (1, 30)),
                               units='meters per second'),
                          Evaporation(wind=constant_wind(1., 30.),
                                      water=Water(333.0)),)
                         )
def test_save_load_wind_objs(saveloc_, obj):
    'test save/load functionality'
    _json_, zipfile_, _refs = obj.save(saveloc_,)
    obj2 = obj.__class__.load(zipfile_)

    assert obj == obj2


# Following movers fail on windows with fixture. This is causing an issue in
# windows for the NetCDF files - for some reason it is not able to delete the
# netcdf data files. All files are being closed in C++.
l_movers2 = (CurrentCycleMover(testdata['CurrentCycleMover']['curr'],
                               topology_file=testdata['CurrentCycleMover']['top'],
                               tide=Tide(testdata['CurrentCycleMover']['tide'])),
             CurrentCycleMover(testdata['CurrentCycleMover']['curr'],
                               topology_file=testdata['CurrentCycleMover']['top']),
             c_GridCurrentMover(testdata['c_GridCurrentMover']['curr_tri'],
                              testdata['c_GridCurrentMover']['top_tri']),
             c_GridWindMover(testdata['c_GridWindMover']['wind_curv'],
                           testdata['c_GridWindMover']['top_curv']),
             )


@pytest.mark.parametrize("obj", l_movers2)
def test_serialize_deserialize_grids(saveloc_, obj):
    'test serialize/deserialize functionality'
    json_ = obj.serialize()
    obj2 = obj.__class__.deserialize(json_)

    assert obj == obj2


@pytest.mark.parametrize("obj", l_movers2)
def test_save_load_grids(saveloc_, obj):
    'test save/load functionality'
    _json_, zipfile_, _refs = obj.save(saveloc_,)
    obj2 = obj.__class__.load(zipfile_)

    assert obj == obj2
    # ==========================================================================
    # temp = os.path.join(dump_folder, 'temp')
    # for dir_ in (temp, os.path.relpath(temp)):
    #     refs = obj.save(dir_)
    #     obj2 = load(os.path.join(dir_, refs.reference(obj)))
    #     assert obj == obj2
    # ==========================================================================


class TestSaveZipIsValid:
    here = os.path.dirname(__file__)

    def test_invalid_zip(self):
        ''' invalid zipfile '''
        with LogCapture() as lc:
            assert not is_savezip_valid('junk.zip')
            lc.check(('gnome.persist.save_load',
                      'WARNING',
                      'junk.zip is not a valid zipfile'))

    # need a bad zip that fails CRC check
    # check max_json_filesize
    def test_max_json_filesize(self):
        '''
        create a fake zip containing
        'sample_data/boston_data/MerrimackMassCoastOSSM.json'
        change _max_json_filesize 4K
        '''
        save_load._max_json_filesize = 8 * 1024
        badzip = os.path.join(self.here,
                              'sample_data/badzip_max_json_filesize.zip')
        filetoobig = 'filetoobig.json'
        with ZipFile(badzip, 'w', compression=ZIP_DEFLATED) as z:
            z.write(testdata['boston_data']['cats_ossm'], filetoobig)

        with LogCapture() as lc:
            assert not is_savezip_valid(badzip)
            lc.check(('gnome.persist.save_load',
                      'WARNING',
                      'Filesize of {0} is {1}. It must be less than {2}. '
                      'Rejecting zipfile.'
                      .format(filetoobig,
                              z.NameToInfo[filetoobig].file_size,
                              save_load._max_json_filesize)))

        save_load._max_json_filesize = 1 * 1024

    def test_check_max_compress_ratio(self):
        '''
        create fake zip containing 1000 '0' as string. The compression ratio
        should be big
        '''
        badzip = os.path.join(self.here,
                              'sample_data/badzip_max_compress_ratio.zip')
        badfile = 'badcompressratio.json'
        with ZipFile(badzip, 'w', compression=ZIP_DEFLATED) as z:
            z.writestr(badfile, ''.join(['0'] * 1000))

        with LogCapture() as lc:
            assert not is_savezip_valid(badzip)
            zi = z.NameToInfo[badfile]
            lc.check(('gnome.persist.save_load',
                      'WARNING',
                      ('file compression ratio is {0}. '
                       'maximum must be less than {1}. '
                       'Rejecting zipfile'
                       .format(zi.file_size / zi.compress_size,
                               save_load._max_compress_ratio))))

    def test_filenames_dont_contain_dotdot(self):
        '''
        '''
        badzip = os.path.join(self.here,
                              'sample_data/badzip_max_compress_ratio.zip')
        badfile = './../badpath.json'
        with ZipFile(badzip, 'w', compression=ZIP_DEFLATED) as z:
            z.writestr(badfile, 'bad file, contains path')

        with LogCapture() as lc:
            assert not is_savezip_valid(badzip)
            lc.check(('gnome.persist.save_load',
                      'WARNING',
                      'Found ".." in {}. Rejecting zipfile'.format(badfile)))
