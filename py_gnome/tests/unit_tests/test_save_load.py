'''
test functionality of the save_load module used to persist save files
'''
import os
from datetime import datetime
import shutil

from gnome.persist import References, load, Savable
from gnome.movers import constant_wind_mover
from gnome import movers, outputters, environment, map, spill, weatherers
from conftest import testdata

import pytest

testdata = testdata()
base_dir = os.path.dirname(__file__)


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


def test_savloc_created():
    'unit test for _make_saveloc method'
    sav = Savable()
    temp = os.path.join(base_dir, 'temp')
    sav._make_saveloc(temp)

    assert os.path.exists(temp)
    shutil.rmtree(temp)


# For WindMover test_save_load in test_wind_mover
l_movers = (environment.Tide(testdata['CatsMover']['tide']),
            environment.Wind(filename=testdata['ComponentMover']['wind']),
            environment.Wind(timeseries=(0, (1, 4)), units='mps'),
            movers.random_movers.RandomMover(),
            movers.CatsMover(testdata['CatsMover']['curr']),
            movers.CatsMover(testdata['CatsMover']['curr'],
                tide=environment.Tide(testdata['CatsMover']['tide'])),
            movers.ComponentMover(testdata['ComponentMover']['curr']),
            movers.ComponentMover(testdata['ComponentMover']['curr'],
                wind=environment.Wind(
                    filename=testdata['ComponentMover']['wind'])),
            #===================================================================
            # movers.CurrentCycleMover(testdata['CurrentCycleMover']['curr'],
            #   topology_file=testdata['CurrentCycleMover']['topology'],
            #   tide=environment.Tide(testdata['CurrentCycleMover']['tide'])),
            # movers.CurrentCycleMover(testdata['CurrentCycleMover']['curr'],
            #   topology_file=testdata['CurrentCycleMover']['topology']),
            # movers.GridCurrentMover(testdata['GridCurrentMover']['curr'],
            #   testdata['GridCurrentMover']['topology']),
            # movers.GridWindMover(testdata['GridWindMover']['wind'],
            #   testdata['GridWindMover']['topology']),
            #===================================================================
            movers.RandomVerticalMover(),
            movers.SimpleMover(velocity=(10.0, 10.0, 0.0)),
            map.MapFromBNA(testdata['MapFromBNA']['testmap'], 6),
            outputters.NetCDFOutput(os.path.join(base_dir, u'xtemp.nc')),
            outputters.Renderer(testdata['Renderer']['bna_sample'],
               testdata['Renderer']['output_dir']),
            spill.PointLineRelease(datetime.now(), 10, (0, 0, 0)),
            spill.point_line_release_spill(10, (0, 0, 0), datetime.now()),
            weatherers.Weatherer(),
            # todo: ask Caitlin how to fix
            #movers.RiseVelocityMover(),
            # todo: This is incomplete - no _schema for SpatialRelease, GeoJson
            #spill.SpatialRelease(datetime.now(), ((0, 0, 0), (1, 2, 0))),
            outputters.GeoJson(),
            )


@pytest.mark.parametrize("obj", l_movers)
def test_save_load(clean_temp, obj):
    'test save/load functionality'
    refs = obj.save(clean_temp)
    obj2 = load(os.path.join(clean_temp, refs.reference(obj)))
    assert obj == obj2


'''
Following movers fail on windows with clean_temp fixture. The clean_temp fixture
deletes ./temp directory before each run of test_save_load(). This is causing
an issue in windows for the NetCDF files - for some reason it is not able to
delete the netcdf data files. All files are being closed in C++, but until we
find the solution, lets break up above tests and not call clean_temp for
following tests.
'''

l_movers2 = (movers.CurrentCycleMover(testdata['CurrentCycleMover']['curr'],
                topology_file=testdata['CurrentCycleMover']['topology'],
                tide=environment.Tide(testdata['CurrentCycleMover']['tide'])),
             movers.CurrentCycleMover(testdata['CurrentCycleMover']['curr'],
               topology_file=testdata['CurrentCycleMover']['topology']),
             movers.GridCurrentMover(testdata['GridCurrentMover']['curr'],
               testdata['GridCurrentMover']['topology']),
             movers.GridWindMover(testdata['GridWindMover']['wind'],
               testdata['GridWindMover']['topology']),
            )


@pytest.mark.parametrize("obj", l_movers2)
def test_save_load2(obj):
    'test save/load functionality'

    temp = os.path.join(base_dir, 'temp')
    for dir_ in (temp, os.path.relpath(temp, base_dir)):
        refs = obj.save(dir_)
        obj2 = load(os.path.join(dir_, refs.reference(obj)))
        assert obj == obj2


@pytest.mark.xfail
@pytest.mark.parametrize("obj", l_movers2[:1])
def test_save_load_netcdf(clean_temp, obj):
    'test save/load functionality'
    refs = obj.save(clean_temp)
    obj2 = load(os.path.join(clean_temp, refs.reference(obj)))
    assert obj == obj2
