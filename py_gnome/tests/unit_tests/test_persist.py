'''
primarily tests the operations of the scenario module, the colander schemas,
and the ability of Model to be recreated in midrun
'''

import os
import shutil
from datetime import datetime, timedelta
import json

import pytest
from pytest import raises

import numpy
np = numpy

from gnome.basic_types import datetime_value_2d
from gnome.map import MapFromBNA
from gnome.environment import Wind, Tide, Water
from gnome.model import Model
from gnome.persist import load
from gnome.spill import point_line_release_spill
from gnome.movers import RandomMover, WindMover, CatsMover, ComponentMover
from gnome.weatherers import Evaporation, Skimmer
from gnome.outputters import Renderer
# from gnome.utilities.remote_data import get_datafile

from conftest import dump, testdata


saveloc_ = os.path.join(dump(), 'save_model')
webapi_files = os.path.join(dump(), 'webapi_json')


# clean up saveloc_ if it exists
# let Scenario.__init__() create saveloc_
def del_saveloc(saveloc_):
    if os.path.exists(saveloc_):
        shutil.rmtree(saveloc_)

del_saveloc(saveloc_)


@pytest.fixture(scope='module')
def images_dir(dump):
    '''
    create images dir
    '''
    images_dir = os.path.join(dump, 'test_images')
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)

    os.makedirs(images_dir)

    return images_dir


def make_model(images_dir, uncertain=False):
    '''
    Create a model from the data in sample_data/boston_data
    It contains:
      - the GeoProjection
      - wind mover
      - random mover
      - cats shio mover
      - cats ossm mover
      - plain cats mover
    '''

    start_time = datetime(2013, 2, 13, 9, 0)
    model = Model(start_time=start_time,
                  duration=timedelta(days=2),
                  time_step=timedelta(minutes=30).total_seconds(),
                  uncertain=uncertain,
                  map=MapFromBNA(testdata['boston_data']['map'],
                                 refloat_halflife=1))

    print 'adding a renderer'

    model.outputters += Renderer(testdata['boston_data']['map'],
                                 images_dir, size=(800, 600))

    print 'adding a spill'
    start_position = (144.664166, 13.441944, 0.0)
    end_release_time = start_time + timedelta(hours=6)
    spill_amount = 1000.0
    spill_units = 'kg'
    model.spills += \
        point_line_release_spill(num_elements=1000,
                                 start_position=start_position,
                                 release_time=start_time,
                                 end_release_time=end_release_time,
                                 amount=spill_amount,
                                 units=spill_units,
                                 substance='ALAMO')

    # need a scenario for SimpleMover
    # model.movers += SimpleMover(velocity=(1.0, -1.0, 0.0))

    print 'adding a RandomMover:'
    model.movers += RandomMover(diffusion_coef=100000)

    print 'adding a wind mover:'

    series = np.zeros((2, ), dtype=datetime_value_2d)
    series[0] = (start_time, (5, 180))
    series[1] = (start_time + timedelta(hours=18), (5, 180))

    w_mover = WindMover(Wind(timeseries=series, units='m/s'))

    model.movers += w_mover
    model.environment += w_mover.wind

    print 'adding a cats shio mover:'

    c_mover = CatsMover(testdata['boston_data']['cats_curr2'],
                        tide=Tide(testdata['boston_data']['cats_shio']))

    # c_mover.scale_refpoint should automatically get set from tide object
    c_mover.scale = True  # default value
    c_mover.scale_value = -1

    # tide object automatically gets added by model
    model.movers += c_mover

    print 'adding a cats ossm mover:'

    c_mover = CatsMover(testdata['boston_data']['cats_curr2'],
                        tide=Tide(testdata['boston_data']['cats_ossm']))

    c_mover.scale = True  # but do need to scale (based on river stage)
    c_mover.scale_refpoint = (-70.65, 42.58333)
    c_mover.scale_value = 1.

    print 'adding a cats mover:'

    c_mover = CatsMover(testdata['boston_data']['cats_curr3'])
    c_mover.scale = True  # but do need to scale (based on river stage)
    c_mover.scale_refpoint = (-70.78333, 42.39333)

    # the scale factor is 0 if user inputs no sewage outfall effects
    c_mover.scale_value = .04
    model.movers += c_mover

    # todo: seg faulting for component mover - comment test for now
    # print "adding a component mover:"
    # comp_mover = ComponentMover(testdata['boston_data']['component_curr1'],
    #                             testdata['boston_data']['component_curr2'],
    #                             w_mover.wind)
    # #todo: callback did not work correctly below - fix!
    # #comp_mover = ComponentMover(component_file1,component_file2,Wind(timeseries=series, units='m/s'))

    # comp_mover.ref_point = (-70.855, 42.275)
    # comp_mover.pat1_angle = 315
    # comp_mover.pat1_speed = 19.44
    # comp_mover.pat1_speed_units = 1
    # comp_mover.pat1ScaleToValue = .138855
    # comp_mover.pat2_angle = 225
    # comp_mover.pat2_speed = 19.44
    # comp_mover.pat2_speed_units = 1
    # comp_mover.pat2ScaleToValue = .05121

    # model.movers += comp_mover

    print 'adding a Weatherer'
    model.water = Water(311.15)
    skim_start = start_time + timedelta(hours=3)
    model.weatherers += [Evaporation(model.water, w_mover.wind),
                         Skimmer(spill_amount * .5,
                                 spill_units,
                                 efficiency=.3,
                                 active_start=skim_start,
                                 active_stop=skim_start + timedelta(hours=2))]

    return model


def test_init_exception(images_dir):
    m = make_model(images_dir)
    with raises(ValueError):
        m.save(os.path.join(saveloc_, 'x', 'junk'))


def test_dir_gets_created(images_dir):
    model = make_model(images_dir, True)
    assert not os.path.exists(saveloc_)
    model.save(os.path.join(saveloc_))
    assert os.path.exists(saveloc_)


@pytest.mark.slow
@pytest.mark.parametrize('uncertain', [False, True])
def test_save_load_model(images_dir, uncertain):
    '''
    create a model, save it, then load it back up and check it is equal to
    original model
    '''
    model = make_model(images_dir, uncertain)

    print 'saving scenario ..'
    model.save(saveloc_)

    print 'loading scenario ..'
    model2 = load(saveloc_)

    assert model == model2


@pytest.mark.slow
@pytest.mark.parametrize('uncertain', [False, True])
def test_save_load_midrun_scenario(images_dir, uncertain):
    """
    create model, save it after 1step, then load and check equality of original
    model and persisted model
    """

    model = make_model(images_dir, uncertain)

    model.step()
    print 'saving scnario ..'
    model.save(saveloc_)

    print 'loading scenario ..'
    model2 = load(os.path.join(saveloc_, 'Model.json'))

    for sc in zip(model.spills.items(), model2.spills.items()):
        sc[0]._array_allclose_atol = 1e-5  # need to change both atol
        sc[1]._array_allclose_atol = 1e-5

    assert model.spills == model2.spills
    assert model == model2


@pytest.mark.slow
@pytest.mark.parametrize('uncertain', [False, True])
def test_save_load_midrun_no_movers(images_dir, uncertain):
    """
    create model, save it after 1step, then load and check equality of original
    model and persisted model
    Remove all movers and ensure it still works as expected
    """

    model = make_model(images_dir, uncertain)

    for mover in model.movers:
        del model.movers[mover.id]

    model.step()
    print 'saving scenario ..'
    model.save(saveloc_)

    print 'loading scenario ..'
    model2 = load(os.path.join(saveloc_, 'Model.json'))

    for sc in zip(model.spills.items(), model2.spills.items()):
        # need to change both atol since reading persisted data
        sc[0]._array_allclose_atol = 1e-5
        sc[1]._array_allclose_atol = 1e-5

    assert model.spills == model2.spills
    assert model == model2


@pytest.mark.slow
@pytest.mark.parametrize('uncertain', [False, True])
def test_load_midrun_ne_rewound_model(images_dir, uncertain):
    """
    Load the same model that was persisted previously after 1 step
    This time rewind the original model and test that the two are not equal.
    The data arrays in the spill container must not match
    """

    # data arrays in model.spills no longer equal

    model = make_model(images_dir, uncertain)

    model.step()
    print 'saving scenario ..'
    model.save(saveloc_)

    model.rewind()
    model2 = load(os.path.join(saveloc_, 'Model.json'))

    assert model.spills != model2.spills
    assert model != model2


class TestWebApi:

    def _write_to_file(self, fname, data):
        with open(fname, 'w') as outfile:
            json.dump(data, outfile, indent=True)

    def _dump_collection(self, coll_):
        # for each object in the model, dump the json
        for count, obj in enumerate(coll_):
            serial = obj.serialize('webapi')
            fname = os.path.join(webapi_files,
                                 '{0}_{1}.json'.format(obj.__class__.__name__,
                                                       count))
            self._write_to_file(fname, serial)

    @pytest.mark.parametrize('uncertain', [False, True])
    def test_dump_webapi_option(self, images_dir, uncertain):
        model = make_model(images_dir, uncertain)
        del_saveloc(webapi_files)
        os.makedirs(webapi_files)
        serial = model.serialize('webapi')
        fname = os.path.join(webapi_files, 'Model.json')
        self._write_to_file(fname, serial)

        for coll in ['movers', 'weatherers', 'environment', 'outputters']:
            self._dump_collection(getattr(model, coll))

        for sc in model.spills.items():
            if sc is not uncertain:
                self._dump_collection(sc.spills)
                # dump release object and element_type object
                for idx, spill in enumerate(sc.spills):
                    for obj in ['release', 'element_type']:
                        serial = getattr(spill, obj).serialize('webapi')
                        fname = os.path.join(webapi_files,
                                             'Spill{0}_{1}.json'.format(idx,
                                                                        obj))
                        self._write_to_file(fname, serial)

    @pytest.mark.parametrize('uncertain', [False, True])
    def test_model_rt(self, images_dir, uncertain):
        model = make_model(images_dir, uncertain)
        deserial = Model.deserialize(model.serialize('webapi'))

        # update the dict so it gives a valid model to load
        deserial['map'] = model.map
        deserial['water'] = model.water
        for coll in ['movers', 'weatherers', 'environment', 'outputters',
                     'spills']:
            for ix, item in enumerate(deserial[coll]):
                deserial[coll][ix] = getattr(model, coll)[item['id']]

        m2 = Model.new_from_dict(deserial)
        assert m2 == model
        print m2


def test_location_file():
    '''
    Simple test to check if json_ contains nothing - default model is created
    '''
    model = Model.load('.', {'json_': 'save'})
    assert model == Model()
