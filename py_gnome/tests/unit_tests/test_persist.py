'''
primarily tests the operations of the scenario module, the colander schemas,
and the ability of Model to be recreated in midrun
tests save/load to directory - original functionality and save/load to zip
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
from gnome.weatherers import Evaporation, Skimmer, Burn

from conftest import dump, testdata, test_oil


def make_model(uncertain=False):
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
                                 substance=test_oil)
    spill = model.spills[-1]
    spill_volume = spill.get_mass()/spill.get('substance').get_density()
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
    c_mover.scale_refpoint = (-70.65, 42.58333, 0.0)
    c_mover.scale_value = 1.

    print 'adding a cats mover:'

    c_mover = CatsMover(testdata['boston_data']['cats_curr3'])
    c_mover.scale = True  # but do need to scale (based on river stage)
    c_mover.scale_refpoint = (-70.78333, 42.39333, 0.0)

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
    model.environment += Water(311.15)
    skim_start = start_time + timedelta(hours=3)
    model._make_default_refs = True
    model.weatherers += [Evaporation(),
                         Skimmer(spill_amount * .5,
                                 spill_units,
                                 efficiency=.3,
                                 active_start=skim_start,
                                 active_stop=skim_start + timedelta(hours=2)),
                         Burn(0.2 * spill_volume, 1.0, skim_start,
                              efficiency=0.9)]

    return model


def zipname(saveloc, mdl):
    # put common two lines of functionality here
    if mdl.zipsave:
        # default name of zip file is same as model.name attribute
        return os.path.join(saveloc, mdl.name + '.zip')
    return saveloc


def test_init_exception(saveloc_):
    m = make_model(False)
    with raises(IOError):
        m.save(os.path.join(saveloc_, 'x', 'junk'))


@pytest.mark.slow
@pytest.mark.parametrize(('uncertain', 'zipsave'),
                         [(False, False), (True, False),
                          (False, True), (True, True)])
def test_save_load_model(uncertain, zipsave, saveloc_):
    '''
    create a model, save it, then load it back up and check it is equal to
    original model
    '''
    model = make_model(uncertain)
    model.zipsave = zipsave

    print 'saving scenario ..'
    model.save(saveloc_)

    print 'loading scenario ..'
    model2 = load(zipname(saveloc_, model))

    assert model == model2


@pytest.mark.slow
@pytest.mark.parametrize(('uncertain', 'zipsave'),
                         [(False, False), (True, False),
                          (False, True), (True, True)])
def test_save_load_midrun_scenario(uncertain, zipsave, saveloc_):
    """
    create model, save it after 1step, then load and check equality of original
    model and persisted model
    """

    model = make_model(uncertain)

    model.step()
    print 'saving scnario ..'
    model.save(saveloc_)

    print 'loading scenario ..'
    model2 = load(zipname(saveloc_, model))

    for sc in zip(model.spills.items(), model2.spills.items()):
        sc[0]._array_allclose_atol = 1e-5  # need to change both atol
        sc[1]._array_allclose_atol = 1e-5

    assert model.spills == model2.spills
    assert model == model2


@pytest.mark.slow
@pytest.mark.parametrize(('uncertain', 'zipsave'),
                         [(False, False), (True, False),
                          (False, True), (True, True)])
def test_save_load_midrun_no_movers(uncertain, zipsave, saveloc_):
    """
    create model, save it after 1step, then load and check equality of original
    model and persisted model
    Remove all movers and ensure it still works as expected
    """

    model = make_model(uncertain)

    for mover in model.movers:
        del model.movers[mover.id]

    model.step()
    print 'saving scenario ..'
    model.save(saveloc_)

    print 'loading scenario ..'
    model2 = load(zipname(saveloc_, model))

    for sc in zip(model.spills.items(), model2.spills.items()):
        # need to change both atol since reading persisted data
        sc[0]._array_allclose_atol = 1e-5
        sc[1]._array_allclose_atol = 1e-5

    assert model.spills == model2.spills
    assert model == model2


@pytest.mark.slow
@pytest.mark.parametrize('uncertain', [False, True])
def test_load_midrun_ne_rewound_model(uncertain, saveloc_):
    """
    Load the same model that was persisted previously after 1 step
    This time rewind the original model and test that the two are not equal.
    The data arrays in the spill container must not match
    """

    # data arrays in model.spills no longer equal

    model = make_model(uncertain)

    model.step()
    print 'saving scenario ..'
    model.save(saveloc_)

    model.rewind()
    model2 = load(zipname(saveloc_, model))

    assert model.spills != model2.spills
    assert model != model2


class TestWebApi:

    webapi_files = os.path.join(dump(), 'webapi_json')

    # clean up saveloc_ if it exists
    # let Scenario.__init__() create saveloc_
    def del_saveloc(self, saveloc_):
        if os.path.exists(saveloc_):
            shutil.rmtree(saveloc_)

    def _write_to_file(self, fname, data):
        with open(fname, 'w') as outfile:
            json.dump(data, outfile, indent=True)

    def _dump_collection(self, coll_):
        # for each object in the model, dump the json
        for count, obj in enumerate(coll_):
            serial = obj.serialize('webapi')
            fname = os.path.join(self.webapi_files,
                                 '{0}_{1}.json'.format(obj.__class__.__name__,
                                                       count))
            self._write_to_file(fname, serial)

    @pytest.mark.parametrize('uncertain', [False, True])
    def test_dump_webapi_option(self, uncertain):
        model = make_model(uncertain)
        self.del_saveloc(self.webapi_files)
        os.makedirs(self.webapi_files)
        serial = model.serialize('webapi')
        fname = os.path.join(self.webapi_files, 'Model.json')
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
                        fname = os.path.join(self.webapi_files,
                                             'Spill{0}_{1}.json'.format(idx,
                                                                        obj))
                        self._write_to_file(fname, serial)

    @pytest.mark.parametrize('uncertain', [False, True])
    def test_model_rt(self, uncertain):
        model = make_model(uncertain)
        deserial = Model.deserialize(model.serialize('webapi'))

        # update the dict so it gives a valid model to load
        deserial['map'] = model.map
        water = model._find_by_attr('_ref_as', 'water', model.environment)
        deserial['water'] = water
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
    model = Model.loads({'json_': 'save'}, '.')
    assert model == Model()


def test_load_fails(saveloc_):
    '''
    if load fails on map or any of the collections, no model is created
    '''
    model = make_model()
    model.zipsave = False
    model.save(saveloc_)
    model_json = json.load(open(os.path.join(saveloc_, 'Model.json'), 'r'))
    model_json['map']['filename'] = 'junk.bna'

    with open(os.path.join(saveloc_, 'Model.json'), 'w') as fd:
        json.dump(model_json, fd, indent=True)

    with pytest.raises(Exception):
        load(saveloc_)
