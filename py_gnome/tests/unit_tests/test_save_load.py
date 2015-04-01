'''
test functionality of the save_load module used to persist save files
'''
import os
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED

from gnome.persist import References, load
from gnome.movers import constant_wind_mover
from gnome import movers, outputters, environment, map, spill, weatherers
from gnome.persist import class_from_objtype, is_savezip_valid
# following is modified for testing only
from gnome.persist import save_load

from conftest import testdata, test_oil

import pytest
from testfixtures import LogCapture


def test_warning_logged():
    '''
    warning is logged if we try to get a class from 'obj_type' that is not
    in the gnome namespace
    '''
    with LogCapture() as l:
        with pytest.raises(AttributeError):
            class_from_objtype('os.path')

        l.check(('gnome.persist.save_load',
                 'WARNING',
                 'os.path is not part of gnome namespace'))


def test_class_from_objtype():
    '''
    test that correct class is returned by class_from_objtype
    '''
    cls = class_from_objtype('gnome.movers.WindMover')
    assert cls is movers.WindMover


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
    l_ = [constant_wind_mover(0, 0) for i in range(2)]
    assert l_[0] == l_[1]
    assert l_[0] is not l_[1]

    refs = References()
    r_l = [refs.reference(item) for item in l_]
    assert len(r_l) == len(l_)
    assert r_l[0] != r_l[1]

    for ix, ref in enumerate(r_l):
        assert refs.retrieve(ref) is l_[ix]
        assert l_[ix] in refs   # double check __contains__

    unknown = constant_wind_mover(0, 0)
    assert unknown not in refs  # check __contains__


'''
Run the following save/load test on multiple pygnome objects so collect tests
here and parametrize it by the objects
'''


base_dir = os.path.dirname(__file__)


# For WindMover test_save_load in test_wind_mover
g_objects = (environment.Tide(testdata['CatsMover']['tide']),
             environment.Wind(filename=testdata['ComponentMover']['wind']),
             environment.Wind(timeseries=(0, (0, 0)), units='mps'),
             environment.Water(temperature=273),
             movers.random_movers.RandomMover(),
             movers.CatsMover(testdata['CatsMover']['curr']),
             movers.CatsMover(testdata['CatsMover']['curr'],
                 tide=environment.Tide(testdata['CatsMover']['tide'])),
             movers.ComponentMover(testdata['ComponentMover']['curr']),
             movers.ComponentMover(testdata['ComponentMover']['curr'],
                 wind=environment.Wind(
                     filename=testdata['ComponentMover']['wind'])),
             movers.RandomVerticalMover(),
             movers.SimpleMover(velocity=(10.0, 10.0, 0.0)),
             map.MapFromBNA(testdata['MapFromBNA']['testmap'], 6),
             outputters.NetCDFOutput(os.path.join(base_dir, u'xtemp.nc')),
             outputters.Renderer(testdata['Renderer']['bna_sample'],
                                 os.path.join(base_dir, 'output_dir')),
             outputters.WeatheringOutput(),
             spill.PointLineRelease(release_time=datetime.now(),
                                    num_elements=10,
                                    start_position=(0, 0, 0)),
             spill.point_line_release_spill(10, (0, 0, 0), datetime.now()),
             spill.elements.ElementType(substance=test_oil),
             weatherers.Skimmer(100, 'kg', 0.3, datetime(2014, 1, 1, 0, 0),
                                datetime(2014, 1, 1, 4, 0)),
             weatherers.Burn(100, 1, datetime(2014, 1, 1, 0, 0)),
            # todo: ask Caitlin how to fix
            #movers.RiseVelocityMover(),
            # todo: This is incomplete - no _schema for SpatialRelease, GeoJson
            #spill.SpatialRelease(datetime.now(), ((0, 0, 0), (1, 2, 0))),
            outputters.GeoJson(),
            )


@pytest.mark.parametrize("obj", g_objects)
def test_save_load(saveloc_, obj):
    'test save/load functionality'
    refs = obj.save(saveloc_)
    obj2 = load(os.path.join(saveloc_, refs.reference(obj)))
    assert obj == obj2


'''
Wind objects need more work - the data is being written out to file in
'r-theta' format accurate to 2 decimal places and in knots. All the conversions
mean the allclose() check on timeseries fails - xfail for now. When loading
for a file it works fine, no decimal places stored in file for magnitude
'''


@pytest.mark.parametrize("obj",
                         (environment.Wind(timeseries=(0, (1, 30)),
                                           units='meters per second'),
                          weatherers.Evaporation(environment.
                                                 constant_wind(1., 30.),
                                                 environment.Water(333.0)),)
                         )
def test_save_load_wind_objs(saveloc_, obj):
    'test save/load functionality'
    refs = obj.save(saveloc_)
    obj2 = load(os.path.join(saveloc_, refs.reference(obj)))
    assert obj == obj2


'''
Following movers fail on windows with fixture. This is causing an issue in
windows for the NetCDF files - for some reason it is not able to delete the
netcdf data files. All files are being closed in C++.
'''

l_movers2 = (movers.CurrentCycleMover(testdata['CurrentCycleMover']['curr'],
                topology_file=testdata['CurrentCycleMover']['top'],
                tide=environment.Tide(testdata['CurrentCycleMover']['tide'])),
             movers.CurrentCycleMover(testdata['CurrentCycleMover']['curr'],
               topology_file=testdata['CurrentCycleMover']['top']),
             movers.GridCurrentMover(testdata['GridCurrentMover']['curr_tri'],
                                     testdata['GridCurrentMover']['top_tri']),
             movers.GridWindMover(testdata['GridWindMover']['wind_curv'],
                                  testdata['GridWindMover']['top_curv']),
             )


@pytest.mark.parametrize("obj", l_movers2)
def test_save_load_grids(saveloc_, obj):
    'test save/load functionality'
    refs = obj.save(saveloc_)
    obj2 = load(os.path.join(saveloc_, refs.reference(obj)))
    assert obj == obj2
    #==========================================================================
    # temp = os.path.join(dump, 'temp')
    # for dir_ in (temp, os.path.relpath(temp)):
    #     refs = obj.save(dir_)
    #     obj2 = load(os.path.join(dir_, refs.reference(obj)))
    #     assert obj == obj2
    #==========================================================================


class TestSaveZipIsValid:
    def test_invalid_zip(self):
        ''' invalid zipfile '''
        with LogCapture() as l:
            assert not is_savezip_valid('junk.zip')
            l.check(('gnome.persist.save_load',
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
        badzip = 'sample_data/badzip_max_json_filesize.zip'
        filetoobig = 'filetoobig.json'
        with ZipFile(badzip, 'a', compression=ZIP_DEFLATED) as z:
            z.write(testdata['boston_data']['cats_ossm'], filetoobig)

        with LogCapture() as l:
            assert not is_savezip_valid(badzip)
            l.check(('gnome.persist.save_load',
                     'WARNING',
                     "Filesize of {0} is {1}. It must be less than "
                     "_max_json_filesize: {2}. Rejecting zipfile.".
                     format(filetoobig,
                            z.NameToInfo[filetoobig].file_size,
                            save_load._max_json_filesize)))

        save_load._max_json_filesize = 1 * 1024

    def test_check_max_compress_ratio(self):
        '''
        create fake zip containing 100 '0' as string. The compression ratio
        should be big
        '''
        badzip = 'sample_data/badzip_max_compress_ratio.zip'
        badfile = 'badcompressratio.json'
        with ZipFile(badzip, 'a', compression=ZIP_DEFLATED) as z:
            z.writestr(badfile, ''.join(['0'] * 100))

        with LogCapture() as l:
            assert not is_savezip_valid(badzip)
            zi = z.NameToInfo[badfile]
            l.check(('gnome.persist.save_load',
                     'WARNING',
                     ("uncompressed filesize is {0} time compressed filesize."
                      "_max_compress_ratio must be less than {1}. Rejecting "
                      "zipfile".format(zi.file_size/zi.compress_size,
                                       save_load._max_compress_ratio))))

    def test_filenames_dont_contain_dotdot(self):
        '''
        '''
        badzip = 'sample_data/badzip_max_compress_ratio.zip'
        badfile = './../badpath.json'
        with ZipFile(badzip, 'a', compression=ZIP_DEFLATED) as z:
            z.writestr(badfile, 'bad file, contains path')

        with LogCapture() as l:
            assert not is_savezip_valid(badzip)
            l.check(('gnome.persist.save_load',
                     'WARNING',
                     "Found '..' in " + badfile + ". Rejecting zipfile"))
