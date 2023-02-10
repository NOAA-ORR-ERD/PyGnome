#!/usr/bin/env python
'''
test code for the model class
'''

import os
import shutil
from datetime import datetime, timedelta

import numpy as np

import pytest
from pytest import raises

import gnome

import gnome.scripting as gs
from gnome.basic_types import datetime_value_2d
from gnome.utilities.inf_datetime import InfDateTime

from gnome.maps import GnomeMap, MapFromBNA
from gnome.environment import Wind, Tide, constant_wind, Water, Waves
from gnome.model import Model

from gnome.spills import (Spill,
                         surface_point_line_spill,
                         Release)

from gnome.movers import SimpleMover, RandomMover, PointWindMover, CatsMover

from gnome.weatherers import (HalfLifeWeatherer,
                              Evaporation,
                              NaturalDispersion,
                              ChemicalDispersion,
                              Burn,
                              Skimmer,
                              Emulsification)
from gnome.outputters import Renderer, TrajectoryGeoJsonOutput

from .conftest import sample_model_weathering, testdata, test_oil
from gnome.spills.substance import NonWeatheringSubstance

from gnome.exceptions import ReferencedObjectNotSet


@pytest.fixture(scope='function')
def model(sample_model_fcn, tmpdir):
    '''
    Utility to setup up a simple, but complete model for tests
    '''
    images_dir = tmpdir.mkdir('Test_images').strpath

    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    model = sample_model_fcn['model']
    rel_start_pos = sample_model_fcn['release_start_pos']
    rel_end_pos = sample_model_fcn['release_end_pos']

    model.cache_enabled = True
    model.uncertain = False

#     model.outputters += Renderer(model.map.filename, images_dir,
#                                   image_size=(400, 300))

    N = 10  # a line of ten points
    line_pos = np.zeros((N, 3), dtype=np.float64)
    line_pos[:, 0] = np.linspace(rel_start_pos[0], rel_end_pos[0], N)
    line_pos[:, 1] = np.linspace(rel_start_pos[1], rel_end_pos[1], N)

    # print start_points

    release = Release(custom_positions=line_pos,
                             release_time=model.start_time)

    model.spills += Spill(release, substance=test_oil)

    # for weatherers and environment objects, make referenced to default
    # wind/water/waves
    model.set_make_default_refs(True)

    return model


def test_init():
    model = Model()

    assert True

# def test_init_invalid_name():
    # """
    # you should not be able to create a name with a slash in it
    # """
    # with pytest.raises(ValueError):
        # model = Model("this/that")

    # with pytest.raises(ValueError):
        # model = Model("this\\that")



def test_update_model():
    mdl = Model()
    d = {'name': 'Model2'}

    assert mdl.name == 'Model'

    upd = mdl.update(d)
    assert mdl.name == d['name']
    assert upd is True

    d['duration'] = 43200
    upd = mdl.update(d)

    assert mdl.duration.seconds == 43200


def test_init_with_mode():
    model = Model()
    assert model.mode == 'gnome'

    model = Model(mode='gnome')
    assert model.mode == 'gnome'

    model = Model(mode='adios')
    assert model.mode == 'adios'

    with raises(ValueError):
        model = Model(mode='bogus')


def test_start_time():
    model = Model()

    st = datetime.now()
    model.start_time = st
    assert model.start_time == st
    assert model.current_time_step == -1

    model.step()

    st = datetime(2012, 8, 12, 13)
    model.start_time = st

    assert model.current_time_step == -1
    assert model.start_time == st

def test_start_time_string():
    """
    model start time can also be set with a string
    """

    st = datetime.now()
    st = st.replace(minute=0, second=0, microsecond=0)

    st_str = st.isoformat()

    model = Model(start_time=st_str)

    assert model.start_time == st

    st = datetime.now()
    model.start_time = st
    assert model.start_time == st
    assert model.current_time_step == -1

    model.step()

    st = datetime(2012, 8, 12, 13)
    model.start_time = '2012-08-12T13'

    assert model.start_time == datetime(2012, 8, 12, 13)
    assert model.current_time_step == -1
    assert model.start_time == st


def test_model_time_and_current_time_in_sc():
    model = Model()
    model.start_time = datetime.now()

    assert model.current_time_step == -1
    assert model.model_time == model.start_time

    for step in range(4):
        model.step()

        assert model.current_time_step == step
        assert (model.model_time ==
                model.start_time + timedelta(seconds=step * model.time_step))

        for sc in model.spills.items():
            assert model.model_time == sc.current_time_stamp


@pytest.mark.parametrize("duration", [1, 2])
def test_release_end_of_step(duration):
    '''
    - Tests that elements released at end of step are recorded with their
      initial conditions with correct timestamp
    - Also tests the age is set correctly.
    todo: write separate test for checking age
    '''
    model = Model(time_step=timedelta(minutes=15),
                  duration=timedelta(hours=duration))

    end_release_time = model.start_time + model.duration
    model.spills += surface_point_line_spill(10, (0.0, 0.0, 0.0),
                                             model.start_time,
                                             end_release_time=end_release_time)

    model.movers += SimpleMover(velocity=(1., -1., 0.0))

    print('\n---------------------------------------------')
    print('model_start_time: {0}'.format(model.start_time))

    prev_rel = 0
    for step in model:
        #only half the particles from the previous step would be new
        new_particles = int((1.0*len(model.spills.LE('positions')) - prev_rel)//2)
        if new_particles > 0:
            assert np.all(model.spills.LE('positions')[-new_particles:, :] ==
                          0)
            assert np.all(model.spills.LE('age')[-new_particles//2:] == 0)
            # assert np.all(model.spills.LE('age')[-new_particles:] ==
            #            (model.model_time + timedelta(seconds=model.time_step)
            #             - model.start_time).seconds)

        if prev_rel > 0:
            assert np.all(model.spills.LE('positions')[:prev_rel, :2] != 0)
            assert np.all(model.spills.LE('age')[:prev_rel] >= model.time_step)

        prev_rel = len(model.spills.LE('positions'))

        print(('current_time_stamp: {0}'
               .format(model.spills.LE('current_time_stamp'))))
        print('particle ID: {0}'.format(model.spills.LE('id')))
        print('positions: \n{0}'.format(model.spills.LE('positions')))
        print('age: \n{0}'.format(model.spills.LE('age')))
        print('just ran: %s' % step)
        print('particles released: %s' % new_particles)
        print('---------------------------------------------')

    print('\n===============================================')


def test_timestep():
    model = Model()

    ts = timedelta(hours=1)
    model.time_step = ts
    assert model.time_step == ts.total_seconds()

    dur = timedelta(days=3)
    model.duration = dur
    assert model._duration == dur


def test_simple_run_rewind():
    '''
    Pretty much all this tests is that the model will run
    and the seed is set during first run, then set correctly
    after it is rewound and run again
    '''

    start_time = datetime(2012, 9, 15, 12, 0)

    model = Model()

    model.map = GnomeMap()
    a_mover = SimpleMover(velocity=(1., 2., 0.))

    model.movers += a_mover
    assert len(model.movers) == 1

    spill = surface_point_line_spill(num_elements=10,
                                     start_position=(0., 0., 0.),
                                     release_time=start_time)

    model.spills += spill
    assert len(model.spills) == 1

    # model.add_spill(spill)

    model.start_time = spill.release.release_time

    # test iterator
    for step in model:
        print('just ran time step: %s' % model.current_time_step)
        assert step['step_num'] == model.current_time_step

    pos = np.copy(model.spills.LE('positions'))

    # rewind and run again:
    print('rewinding')
    model.rewind()

    # test iterator is repeatable
    for step in model:
        print('just ran time step: %s' % model.current_time_step)
        assert step['step_num'] == model.current_time_step

    assert np.all(model.spills.LE('positions') == pos)


def test_simple_run_with_map():
    '''
    pretty much all this tests is that the model will run
    '''

    start_time = datetime(2012, 9, 15, 12, 0)

    model = Model()

    model.map = MapFromBNA(testdata['MapFromBNA']['testmap'],
                                     refloat_halflife=6)  # hours
    a_mover = SimpleMover(velocity=(1., 2., 0.))

    model.movers += a_mover
    assert len(model.movers) == 1

    spill = surface_point_line_spill(num_elements=10,
                                     start_position=(0., 0., 0.),
                                     release_time=start_time)

    model.spills += spill
    assert len(model.spills) == 1
    model.start_time = spill.release.release_time

    # test iterator
    for step in model:
        print('just ran time step: %s' % step)
        assert step['step_num'] == model.current_time_step

    # reset and run again
    model.reset()

    # test iterator is repeatable
    for step in model:
        print('just ran time step: %s' % step)
        assert step['step_num'] == model.current_time_step


def test_simple_run_with_image_output(tmpdir):
    '''
    Pretty much all this tests is that the model will run and output images
    '''
    images_dir = tmpdir.mkdir('Test_images').strpath

    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    start_time = datetime(2012, 9, 15, 12, 0)

    # the land-water map
    gnome_map = MapFromBNA(testdata['MapFromBNA']['testmap'],
                                     refloat_halflife=6)  # hours
    renderer = gnome.outputters.Renderer(testdata['MapFromBNA']['testmap'],
                                         images_dir, image_size=(400, 300))
    geo_json = TrajectoryGeoJsonOutput(output_dir=images_dir)

    model = Model(time_step=timedelta(minutes=15),
                  start_time=start_time, duration=timedelta(hours=1),
                  map=gnome_map,
                  uncertain=False, cache_enabled=False)

    model.outputters += renderer
    model.outputters += geo_json

    a_mover = SimpleMover(velocity=(1., -1., 0.))
    model.movers += a_mover
    assert len(model.movers) == 1

    N = 10  # a line of ten points
    start_points = np.zeros((N, 3), dtype=np.float64)
    start_points[:, 0] = np.linspace(-127.1, -126.5, N)
    start_points[:, 1] = np.linspace(47.93, 48.1, N)
    # print start_points

    spill = Spill(release=Release(custom_positions=start_points,
                                 release_time=start_time))

    model.spills += spill
    assert len(model.spills) == 1

    model.start_time = spill.release.release_time

    # image_info = model.next_image()
    num_steps_output = 0
    while True:
        try:
            model.step()
            num_steps_output += 1
        except StopIteration:
            print('Done with the model run')
            break

    # There is the zeroth step, too.
    calculated_steps = (model.duration.total_seconds() / model.time_step) + 1
    assert num_steps_output == calculated_steps


def test_simple_run_with_image_output_uncertainty(tmpdir):
    '''
    Pretty much all this tests is that the model will run and output images
    '''
    images_dir = tmpdir.mkdir('Test_images2').strpath

    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    start_time = datetime(2012, 9, 15, 12, 0)

    # the land-water map
    gmap = MapFromBNA(testdata['MapFromBNA']['testmap'],
                                refloat_halflife=6)  # hours
    renderer = gnome.outputters.Renderer(testdata['MapFromBNA']['testmap'],
                                         images_dir, image_size=(400, 300))

    model = Model(start_time=start_time,
                  time_step=timedelta(minutes=15), duration=timedelta(hours=1),
                  map=gmap,
                  uncertain=True, cache_enabled=False,
                  )

    model.outputters += renderer
    a_mover = SimpleMover(velocity=(1., -1., 0.))
    model.movers += a_mover

    N = 10  # a line of ten points
    start_points = np.zeros((N, 3), dtype=np.float64)
    start_points[:, 0] = np.linspace(-127.1, -126.5, N)
    start_points[:, 1] = np.linspace(47.93, 48.1, N)
    # print start_points

    release = Release(custom_positions=start_points,
                             release_time=start_time)

    model.spills += Spill(release)

    # model.add_spill(spill)

    model.start_time = release.release_time

    # image_info = model.next_image()

    model.uncertain = True
    num_steps_output = 0
    while True:
        try:
            image_info = model.step()
            num_steps_output += 1
            print(image_info)
        except StopIteration:
            print('Done with the model run')
            break

    # there is the zeroth step, too.
    calculated_steps = (model.duration.total_seconds() / model.time_step) + 1
    assert num_steps_output == calculated_steps

    # fixme -- do an assertion looking for red in images?
    #          or at least make sure they are created?


def test_mover_api():
    '''
    Test the API methods for adding and removing movers to the model.
    '''
    start_time = datetime(2012, 1, 1, 0, 0)

    model = Model()
    model.duration = timedelta(hours=12)
    model.time_step = timedelta(hours=1)
    model.start_time = start_time

    mover_1 = SimpleMover(velocity=(1., -1., 0.))
    mover_2 = SimpleMover(velocity=(1., -1., 0.))
    mover_3 = SimpleMover(velocity=(1., -1., 0.))
    mover_4 = SimpleMover(velocity=(1., -1., 0.))

    # test our add object methods

    model.movers += mover_1
    model.movers += mover_2

    # test our get object methods

    assert model.movers[mover_1.id] == mover_1
    assert model.movers[mover_2.id] == mover_2
    with raises(KeyError):
        temp = model.movers['Invalid']
        print(temp)

    # test our iter and len object methods
    assert len(model.movers) == 2
    assert len([m for m in model.movers]) == 2
    for (m1, m2) in zip(model.movers, [mover_1, mover_2]):
        assert m1 == m2

    # test our add objectlist methods
    model.movers += [mover_3, mover_4]
    assert [m for m in model.movers] == [mover_1, mover_2, mover_3, mover_4]

    # test our remove object methods
    del model.movers[mover_3.id]
    assert [m for m in model.movers] == [mover_1, mover_2, mover_4]

    with raises(KeyError):
        # our key should also be gone after the delete
        temp = model.movers[mover_3.id]
        print(temp)

    # test our replace method
    model.movers[mover_2.id] = mover_3
    assert [m for m in model.movers] == [mover_1, mover_3, mover_4]
    assert model.movers[mover_3.id] == mover_3

    with raises(KeyError):
        # our key should also be gone after the delete
        temp = model.movers[mover_2.id]
        print(temp)


# model start_time, No. of time_steps after which LEs release,
# duration as No. of timesteps

test_cases = [(datetime(2012, 1, 1, 0, 0), 0, 12),
              (datetime(2012, 1, 1, 0, 0), 12, 12),
              (datetime(2012, 1, 1, 0, 0), 13, 12)]


@pytest.mark.parametrize(('start_time', 'release_delay', 'duration'),
                         test_cases)
def test_all_movers(start_time, release_delay, duration):
    '''
    Tests that all the movers at least can be run

    Add new ones as they come along!
    '''

    model = Model()
    model.time_step = timedelta(hours=1)
    model.duration = timedelta(seconds=model.time_step * duration)
    model.start_time = start_time
    start_loc = (1., 2., 0.)  # random non-zero starting points

    # a spill - release after 5 timesteps

    release_time = (start_time +
                    timedelta(seconds=model.time_step * release_delay))
    model.spills += surface_point_line_spill(num_elements=10,
                                             start_position=start_loc,
                                             release_time=release_time)

    # the land-water map
    model.map = GnomeMap()  # the simplest of maps

    # simple mover
    model.movers += SimpleMover(velocity=(1., -1., 0.))
    assert len(model.movers) == 1

    # random mover
    model.movers += RandomMover(diffusion_coef=100000)
    assert len(model.movers) == 2

    # wind mover
    series = np.array((start_time, (10, 45)),
                      dtype=datetime_value_2d).reshape((1, ))
    model.movers += PointWindMover(Wind(timeseries=series,
                              units='meter per second'))
    assert len(model.movers) == 3

    # CATS mover
    model.movers += CatsMover(testdata['CatsMover']['curr'])
    assert len(model.movers) == 4

    # run the model all the way...
    num_steps_output = 0
    for step in model:
        num_steps_output += 1
        print('running step:', step)

    # test release happens correctly for all cases
    if release_delay < duration:
        # at least one get_move has been called after release
        assert np.all(model.spills.LE('positions')[:, :2] != start_loc[:2])
    elif release_delay == duration:
        # particles are released after last step so no motion,
        # only initial _state
        assert np.all(model.spills.LE('positions') == start_loc)
    else:
        # release_delay > duration so nothing released though model ran
        assert len(model.spills.LE('positions')) == 0

    # there is the zeroth step, too.
    calculated_steps = (model.duration.total_seconds() / model.time_step) + 1
    assert num_steps_output == calculated_steps


# 0 is infinite persistence

@pytest.mark.parametrize('wind_persist', [-1, 900, 5])
def test_linearity_of_wind_movers(wind_persist):
    '''
    PointWindMover is defined as a linear operation - defining a model
    with a single PointWindMover with 15 knot wind is equivalent to defining
    a model with three WindMovers each with 5 knot wind. Or any number of
    PointWindMover's such that the sum of their magnitude is 15knots and the
    direction of wind is the same for both cases.

    Below is an example which defines two models and runs them.
    In model2, there are multiple winds defined so the windage parameter
    is reset 3 times for one timestep.
    Since windage range and persistence do not change, this only has the effect
    of doing the same computation 3 times. However, the results are the same.

    The mean and variance of the positions for both models are close.
    As windage_persist is decreased, the values become closer.
    Setting windage_persist=0 gives the large difference between them.
    '''
    units = 'meter per second'
    start_time = datetime(2012, 1, 1, 0, 0)
    series1 = np.array((start_time, (15, 45)),
                       dtype=datetime_value_2d).reshape((1, ))
    series2 = np.array((start_time, (6, 45)),
                       dtype=datetime_value_2d).reshape((1, ))
    series3 = np.array((start_time, (3, 45)),
                       dtype=datetime_value_2d).reshape((1, ))

    num_LEs = 1000

    model1 = Model(name='model1')
    model1.duration = timedelta(hours=1)
    model1.time_step = timedelta(hours=1)
    model1.start_time = start_time
    sp = surface_point_line_spill(num_elements=num_LEs,
                                 start_position=(1., 2., 0.),
                                 release_time=start_time,
                                 substance=NonWeatheringSubstance(windage_persist=wind_persist))
    model1.spills += sp


    model1.movers += PointWindMover(Wind(timeseries=series1, units=units),
                               make_default_refs=False)

    model2 = Model(name='model2')
    model2.duration = timedelta(hours=10)
    model2.time_step = timedelta(hours=1)
    model2.start_time = start_time
    model2.spills += surface_point_line_spill(num_elements=num_LEs,
                                              start_position=(1., 2., 0.),
                                              release_time=start_time,
                                              substance=NonWeatheringSubstance(windage_persist=wind_persist))

    # todo: CHECK RANDOM SEED
    # model2.movers += PointWindMover(Wind(timeseries=series1, units=units))

    model2.movers += PointWindMover(Wind(timeseries=series2, units=units))
    model2.movers += PointWindMover(Wind(timeseries=series2, units=units))
    model2.movers += PointWindMover(Wind(timeseries=series3, units=units))
    model2.set_make_default_refs(False)

    while True:
        try:
            next(model1)
        except StopIteration as ex:
            # print message
            print(ex)
            break

    while True:
        try:
            next(model2)
        except StopIteration as ex:
            # print message
            print(ex)
            break

    # mean and variance at the end should be fairly close
    # look at the mean of the position vector. Assume m1 is truth
    # and m2 is approximation - look at the absolute error between
    # mean position of m2 in the 2 norm.
    # rel_mean_error =(np.linalg.norm(np.mean(model2.spills.LE('positions'), 0)
    #                  - np.mean(model1.spills.LE('positions'), 0)))
    # assert rel_mean_error <= 0.5

    # Similarly look at absolute error in variance of position of m2
    # in the 2 norm.

    rel_var_error = np.linalg.norm(np.var(model2.spills.LE('positions'), 0) -
                                   np.var(model1.spills.LE('positions'), 0))
    assert rel_var_error <= 0.0015


def test_model_release_after_start():
    '''
    This runs the model for a simple spill, that starts after the model starts
    '''
    units = 'meter per second'
    seconds_in_minute = 60
    start_time = datetime(2013, 2, 22, 0)

    model = Model(time_step=30 * seconds_in_minute,
                  start_time=start_time, duration=timedelta(hours=3))

    # add a spill that starts after the run begins.
    release_time = start_time + timedelta(hours=1)
    model.spills += surface_point_line_spill(num_elements=5,
                                             start_position=(0, 0, 0),
                                             release_time=release_time)

    # and another that starts later..

    model.spills += surface_point_line_spill(num_elements=4,
                                             start_position=(0, 0, 0),
                                             release_time=(start_time +
                                                           timedelta(hours=2))
                                             )

    # Add a Wind mover:
    series = np.array((start_time, (10, 45)),
                      dtype=datetime_value_2d).reshape((1, ))
    model.movers += PointWindMover(Wind(timeseries=series, units=units))

    for step in model:
        print('running a step')
        assert step['step_num'] == model.current_time_step

        for sc in model.spills.items():
            print('num_LEs', len(sc['positions']))


def test_release_at_right_time():
    '''
    Tests that the elements get released when they should

    There are issues in that we want the elements to show
    up in the output for a given time step if they were
    supposed to be released then.  Particularly for the
    first time step of the model.
    '''
    # default to now, rounded to the nearest hour
    # seconds_in_minute = 60
    # minutes_in_hour = 60
    # seconds_in_hour = seconds_in_minute * minutes_in_hour

    start_time = datetime(2013, 1, 1, 0)
    time_step = gs.hours(2)

    model = Model(time_step=time_step,
                  start_time=start_time,
                  duration=timedelta(hours=12))

    # add a spill that starts right when the run begins

    model.spills += surface_point_line_spill(num_elements=12,
                                             start_position=(0, 0, 0),
                                             release_time=start_time,
                                             end_release_time=start_time + gs.hours(6),
                                             )

    # before the run - no elements present since data_arrays get defined after
    # 1st step (prepare_for_model_run):

    assert model.spills.items()[0].num_released == 0

    model.step()
    assert model.spills.items()[0].num_released == 0

    model.step()
    assert model.spills.items()[0].num_released == 4

    model.step()
    assert model.spills.items()[0].num_released == 8

    model.step()
    assert model.spills.items()[0].num_released == 12

    model.step()
    assert model.spills.items()[0].num_released == 12


# @pytest.mark.skip(reason="Segfault on CI server")
@pytest.mark.parametrize("traj_only", [False, True])
def test_full_run(model, dump_folder, traj_only):
    'Test doing a full run'
    # model = setup_simple_model()
    if traj_only:
        for spill in model.spills:
            spill.substance = None
        model.weatherers.clear()

    results = model.full_run()
    print(results)

    # check the number of time steps output is right
    # there is the zeroth step, too.
    calculated_steps = (model.duration.total_seconds() / model.time_step) + 1
    assert len(results) == calculated_steps

    # check if the images are there:
    # The Renderer instantiated with default arguments has two output formats
    # which create a background image and an animated gif in addition to the
    # step images.
#JAH: Shutting this off cause Renderer removed (segfaults on CI)
#     num_images = len(os.listdir(model.outputters[-1].output_dir))
#     assert num_images == model.num_time_steps + 2


''' Test Callbacks on OrderedCollections '''


def test_callback_add_mover():
    'Test callback after add mover'
    units = 'meter per second'

    model = Model()
    model.start_time = datetime(2012, 1, 1, 0, 0)
    model.duration = timedelta(hours=10)
    model.time_step = timedelta(hours=1)

    # start_loc = (1.0, 2.0, 0.0)  # random non-zero starting points

    # add Movers
    model.movers += SimpleMover(velocity=(1., -1., 0.))
    series = np.array((model.start_time, (10, 45)),
                      dtype=datetime_value_2d).reshape((1, ))
    model.movers += PointWindMover(Wind(timeseries=series, units=units))

    # this should create a Wind object
    new_wind = Wind(timeseries=series, units=units)
    model.environment += new_wind
    assert new_wind in model.environment
    assert len(model.environment) == 2

    tide_ = Tide(filename=testdata['CatsMover']['tide'])

    d_file = testdata['CatsMover']['curr']
    model.movers += CatsMover(d_file, tide=tide_)

    model.movers += CatsMover(d_file)

    for mover in model.movers:
        assert mover.active_range == (InfDateTime('-inf'),
                                      InfDateTime('inf'))

        if hasattr(mover, 'wind'):
            assert mover.wind in model.environment

        if hasattr(mover, 'tide'):
            if mover.tide is not None:
                assert mover.tide in model.environment

    # Add a mover with user defined active start / active stop values
    # - these should not be updated

    active_on = model.start_time + timedelta(hours=1)
    active_off = model.start_time + timedelta(hours=4)
    custom_mover = SimpleMover(velocity=(1., -1., 0.),
                               active_range=(active_on, active_off))
    model.movers += custom_mover

    assert model.movers[custom_mover.id].active_range == (active_on,
                                                          active_off)


def test_callback_add_mover_midrun():
    'Test callback after add mover called midway through the run'
    model = Model()
    model.start_time = datetime(2012, 1, 1, 0, 0)
    model.duration = timedelta(hours=10)
    model.time_step = timedelta(hours=1)

    # start_loc = (1.0, 2.0, 0.0)  # random non-zero starting points

    # model = setup_simple_model()

    for _i in range(2):
        model.step()

    assert model.current_time_step > -1

    # now add another mover and make sure model rewinds
    model.movers += SimpleMover(velocity=(2., -2., 0.))
    assert model.current_time_step == -1


def test_callback_add_weather():
    '''
    Test callback when weatherer is added
    '''
    model = Model()
    water = Water()
    wind = constant_wind(1, 30)
    assert len(model.environment) == 0

    model.weatherers += Evaporation(water, wind)

    # wind and water added to environment collection
    assert len(model.environment) == 2
    assert wind in model.environment
    assert water in model.environment


def test_simple_run_no_spills(model):
    # model = setup_simple_model()

    for spill in model.spills:
        del model.spills[spill.id]

    assert len(model.spills) == 0

    for step in model:
        print('just ran time step: %s' % model.current_time_step)
        assert step['step_num'] == model.current_time_step


@pytest.mark.parametrize("add_langmuir", (False, True))
def test_all_weatherers_in_model(model, add_langmuir):
    '''
    test model run with weatherer
    todo: integrate Langmuir in Model; can ensure 'frac_coverage' gets added
        to spill container data
    '''
    model.weatherers += HalfLifeWeatherer()

    model.environment += Water()
    model.full_run()

    expected_keys = {'mass_components'}
    assert expected_keys.issubset(model.spills.LE_data)

@pytest.mark.xfail()
def test_setup_model_run(model):
    'turn of movers/weatherers and ensure data_arrays change'
    model.environment += Water()
    model.rewind()
    model.step()
    exp_keys = {'windages', 'windage_range', 'mass_components',
                'windage_persist'}
    # no exp_keys in model data_arrays
    assert not exp_keys.intersection(model.spills.LE_data)

    cwm = gnome.movers.constant_point_wind_mover(1., 0.)

    model.weatherers += [HalfLifeWeatherer(), Evaporation()]
    model.movers += cwm
    model.rewind()
    model.step()

    assert exp_keys.issubset(model.spills.LE_data)

    cwm.on = False
    for w in range(2):
        model.weatherers[w].on = False

    model.rewind()
    model.step()
    assert not exp_keys.intersection(model.spills.LE_data)


def test_contains_object(sample_model_fcn):
    '''
    Test that we can find all contained object types with a model.
    '''
    model = sample_model_weathering(sample_model_fcn, test_oil)

    gnome_map = model.map = GnomeMap()    # make it all water

    rel_time = model.spills[0].release_time
    model.start_time = rel_time - timedelta(hours=1)
    model.duration = timedelta(days=1)

    water, wind = Water(), constant_wind(1., 0)
    model.environment += [water, wind]

    sp = surface_point_line_spill(500, (0, 0, 0),
                                  rel_time + timedelta(hours=1),
                                  substance=model.spills[0].substance,
                                  amount=100,
                                  units='tons')
    rel = sp.release
    model.spills += sp

    movers = [m for m in model.movers]

    evaporation = Evaporation()
    skim_start = sp.release_time + timedelta(hours=1)
    skimmer = Skimmer(.5 * sp.amount, units=sp.units, efficiency=0.3,
                      active_range=(skim_start,
                                    skim_start + timedelta(hours=1)))
    burn = burn_obj(sp)
    disp_start = skim_start + timedelta(hours=1)
    dispersion = ChemicalDispersion(0.1,
                                    active_range=(disp_start,
                                                  disp_start +
                                                  timedelta(hours=1)))

    model.weatherers += [evaporation, dispersion, burn, skimmer]

    renderer = Renderer(images_dir='junk', image_size=(400, 300))
    model.outputters += renderer

    for o in (gnome_map, sp,
              water, wind,
              evaporation, dispersion, burn, skimmer,
              renderer):
        assert model.contains_object(o.id)

#     for o in initializers:
#         assert model.contains_object(o.id)

    for o in movers:
        assert model.contains_object(o.id)


def make_skimmer(spill, delay_hours=1, duration=2):
    'make a skimmer for sample model tests'
    rel_time = spill.release_time
    skim_start = rel_time + timedelta(hours=delay_hours)
    amount = spill.amount
    units = spill.units
    skimmer = Skimmer(.3 * amount, units=units, efficiency=0.3,
                      active_range=(skim_start,
                                    skim_start + timedelta(hours=duration)))
    return skimmer


def chemical_disperson_obj(spill, delay_hours=1, duration=1):
    '''
    apply chemical dispersion to 10% of spill
    '''
    rel_time = spill.release_time
    disp_start = rel_time + timedelta(hours=delay_hours)

    return ChemicalDispersion(.1,
                              active_range=(disp_start,
                                            disp_start +
                                            timedelta(hours=duration)),
                              efficiency=0.3)


def burn_obj(spill, delay_hours=1.5):
    rel_time = spill.release_time
    burn_start = rel_time + timedelta(hours=delay_hours)
    volume = spill.get_mass() / spill.substance.density_at_temp()
    thick = 1   # in meters
    area = (0.2 * volume) / thick

    return Burn(area, thick, active_range=(burn_start, InfDateTime('inf')))


@pytest.mark.parametrize("delay", [timedelta(hours=0),
                                   timedelta(hours=1)])
def test_staggered_spills_weathering(sample_model_fcn, delay):
    '''
    Test that a model with weathering and spills staggered in time runs
    without errors. Also test that a continuous + instant release works
    correctly where the total amount_released is the sum of oil removed by
    weathering processes

    test exposed a bug, which is now fixed
    '''
    model = sample_model_weathering(sample_model_fcn, test_oil)
    model.map = GnomeMap()    # make it all water
    model.uncertain = False
    rel_time = model.spills[0].release_time
    model.start_time = rel_time - timedelta(hours=1)
    model.duration = timedelta(days=1)

    # test with outputter + w/ cache enabled
    model.cache = True
    model.outputters += gnome.outputters.WeatheringOutput()

    cs = surface_point_line_spill(500, (0, 0, 0),
                                  rel_time + delay,
                                  end_release_time=(rel_time + delay +
                                                    timedelta(hours=1)),
                                  substance=model.spills[0].substance,
                                  amount=1,
                                  units='tonnes')
    model.spills += cs

    # ensure amount released is equal to exp_total_mass
    exp_total_mass = 0.0
    for spill in model.spills:
        exp_total_mass += spill.get_mass()

    skimmer = make_skimmer(model.spills[0])
    burn = burn_obj(model.spills[0])
    c_disp = chemical_disperson_obj(model.spills[0], 4)

    model.environment += [Water(), constant_wind(1., 0)]
    model.weatherers += [Evaporation(),
                         c_disp,
                         burn,
                         skimmer]
    model.set_make_default_refs(True)

    # model.full_run()
    for step in model:
        if not step['valid']:
            print(step['messages'])
            raise RuntimeError("Model has error in setup_model_run")

        for sc in model.spills.items():
            print("completed step {0}".format(step))
            # sum up all the weathered mass + mass of LEs marked for weathering
            # and ensure this equals the total amount released
            print((sc.mass_balance['beached'],
                   sc.mass_balance['burned'],
                   sc.mass_balance['chem_dispersed'],
                   sc.mass_balance['evaporated'],
                   sc.mass_balance['floating'],
                   sc.mass_balance['skimmed'],
                   ))
            sum_ = (sc.mass_balance['beached'] +
                    sc.mass_balance['burned'] +
                    sc.mass_balance['chem_dispersed'] +
                    sc.mass_balance['evaporated'] +
                    sc.mass_balance['floating'] +
                    sc.mass_balance['skimmed']
                    )

            assert np.isclose(sum_, sc.mass_balance['amount_released'])

    assert sc.mass_balance['burned'] > 0
    assert sc.mass_balance['skimmed'] > 0

    assert np.isclose(exp_total_mass, sc.mass_balance['amount_released'])


def test_two_substance_same(sample_model_fcn, s0=test_oil, s1=test_oil):
    '''
    The model (SpillContainer) does not allow two different substances.

    Keeping this test in case we do want to extend it some day.

    only tests data arrays are correct and we don't end up with stale data
    in substance_data structure of spill container. It models each substance
    independently

    We don't accurately model two oils at present. This is a basic test,
    maybe a useful example when extending code to multiple oils. It is also
    useful for catching bugs when doing a refactor so leave it in.
    '''
    model = sample_model_weathering(sample_model_fcn, s0)
    model.map = GnomeMap()    # make it all water
    model.uncertain = False
    rel_time = model.spills[0].release_time
    model.duration = timedelta(days=1)

    cs = surface_point_line_spill(500, (0, 0, 0),
                                  rel_time,
                                  end_release_time=(rel_time +
                                                    timedelta(hours=1)),
                                  substance=model.spills[0].substance,
                                  amount=1,
                                  units='tonnes')

    if s0 == s1:
        print("substances are the same -- it should work")
        model.spills += cs
    else:
        print("two different substances -- expect an error")
        with pytest.raises(ValueError):
            model.spills += cs

    # ensure amount released is equal to exp_total_mass
    exp_total_mass = 0.0
    for spill in model.spills:
        exp_total_mass += spill.get_mass()

    model.environment += [Water(), constant_wind(1., 0)]
    # model will automatically setup default references
    model.weatherers += Evaporation()
    if s0 == s1:
        '''
        multiple substances will not work with Skimmer or Burn
        '''
        c_disp = chemical_disperson_obj(model.spills[0], 3)
        burn = burn_obj(model.spills[0], 2.5)
        skimmer = make_skimmer(model.spills[0], 2)

        model.weatherers += [c_disp, burn, skimmer]

    model.set_make_default_refs(True)

    # model.full_run()
    for step in model:
        for sc in model.spills.items():
            # sum up all the weathered mass + mass of LEs marked for weathering
            # and ensure this equals the total amount released
            sum_ = 0.0

            if s0 == s1:
                # mass marked for skimming/burning/dispersion that is not yet
                # removed - cleanup operations only work on single substance
                sum_ += (sc.mass_balance['burned'] +
                         sc.mass_balance['chem_dispersed'] +
                         sc.mass_balance['skimmed'])

            sum_ += (sc.mass_balance['beached'] +
                     sc.mass_balance['evaporated'] +
                     sc.mass_balance['floating'])

            assert np.isclose(sum_, sc.mass_balance['amount_released'])

        print("completed step {0}".format(step))

    assert np.isclose(exp_total_mass, sc.mass_balance['amount_released'])


def test_two_substance_different(sample_model_fcn, s0=test_oil, s1="oil_crude"):
    '''
    The model (SpillContainer) does not allow two different substances.

    Keeping this test in case we do want to extend it some day.

    only tests data arrays are correct and we don't end up with stale data
    in substance_data structure of spill container. It models each substance
    independently

    We don't accurately model two oils at present. This is a basic test,
    maybe a useful example when extending code to multiple oils. It is also
    useful for catching bugs when doing a refactor so leave it in.
    '''
    model = sample_model_weathering(sample_model_fcn, s0)
    model.map = GnomeMap()    # make it all water
    model.uncertain = False
    rel_time = model.spills[0].release_time
    model.duration = timedelta(days=1)

    cs = surface_point_line_spill(500, (0, 0, 0),
                                  rel_time,
                                  end_release_time=(rel_time +
                                                    timedelta(hours=1)),
                                  substance=s1,
                                  amount=1,
                                  units='tonnes')

    with pytest.raises(ValueError):
        model.spills += cs


def test_weathering_data_attr():
    '''
    mass_balance is initialized/written if we have weatherers
    '''
    ts = 900
    s1_rel = datetime.now().replace(microsecond=0)
    s2_rel = s1_rel + timedelta(seconds=ts)

    s = [surface_point_line_spill(10, (0, 0, 0), s1_rel),
         surface_point_line_spill(10, (0, 0, 0), s2_rel)]

    model = Model(time_step=ts, start_time=s1_rel)
    model.spills += s
    model.step()

    for sc in model.spills.items():
        assert len(sc.mass_balance) == 7
        for key in ('floating', 'avg_density', 'avg_viscosity', 'non_weathering', 'amount_released', 'beached', 'off_maps'):
            assert key in sc.mass_balance

    model.environment += [Water(), constant_wind(0., 0)]
    model.weatherers += [Evaporation(model.environment[0],
                                     model.environment[1])]

    # use different element_type and initializers for both spills
    s[0].amount = 10.0
    s[0].units = 'kg'
    model.rewind()
    model.step()

    for sc in model.spills.items():
        # since no substance is defined, all the LEs are marked as
        # nonweathering
        assert sc.mass_balance['non_weathering'] == sc['mass'].sum()
        assert sc.mass_balance['non_weathering'] == s[0].amount

    s[1].amount = 5.0
    s[1].units = 'kg'
    model.rewind()
    exp_rel = 0.0

    for ix in range(2):
        model.step()
        exp_rel += s[ix].amount
        for sc in model.spills.items():
            assert sc.mass_balance['non_weathering'] == sc['mass'].sum()
            assert sc.mass_balance['non_weathering'] == exp_rel

    model.rewind()

    assert sc.mass_balance == {}


def test_run_element_type_no_initializers(model):
    '''
    run model with only one spill, it contains an element_type.
    However, element_type has no initializers - will not work if weatherers
    are present that require runtime array_types defined. The initializers
    currently define these runtime array_types -- need to rethink how this
    should work
    '''
    model.uncertain = False
    model.rewind()

    for ix, spill in enumerate(model.spills):
        if ix == 0:
            spill.initializers = []
        else:
            del model.spills[spill.id]
    assert len(model.spills) == 1

    model.full_run()

    assert True


class TestMergeModels:
    def test_merge_from_empty_model(self, model):
        '''
        merge empty model - nothing to merge
        deepcopy fails on model due to cache - we don't anticipate needing
        to deepcopy models so do a hack for testing for now.
        '''
        m = Model()
        model.merge(m)
        for oc in m._oc_list:
            for item in getattr(m, oc):
                assert item in getattr(model, oc)

    def test_load_location_file(self, saveloc_, model):
        '''
        create a model
        load save file from script_boston which contains a spill. Then merge
        the created model into the model loaded from save file
        '''
        m = Model()
        m.environment += [Water(), constant_wind(1., 0.)]
        m.weatherers += Evaporation(m.environment[0], m.environment[-1])
        # has to have the same substance as the sample model
        m.spills += surface_point_line_spill(10, (0, 0, 0),
                                             datetime(2014, 1, 1, 12, 0),
                                             substance=test_oil)

        # create save model
        sample_save_file = os.path.join(saveloc_, 'SampleSaveModel.zip')
        model.save(sample_save_file)

        if os.path.exists(sample_save_file):
            model = Model.load(sample_save_file)
            model.merge(m)

            for oc in m._oc_list:
                for item in getattr(m, oc):
                    model_oc = getattr(model, oc)
                    assert item is model_oc[item.id]

            for spill in m.spills:
                assert spill is model.spills[spill.id]

            # merge the other way and ensure model != m
            m.merge(model)
            assert model != m


# test sorting function weatherer_sort
def test_weatherer_sort():
    '''
    Sample model with weatherers - only tests sorting of weathereres. The
    Model will likely not run
    '''
    model = Model()

    skimmer = Skimmer(amount=100, units='kg', efficiency=0.3,
                      active_range=(datetime(2014, 1, 1, 0, 0),
                                    datetime(2014, 1, 1, 0, 3)))
    burn = Burn(100, 1,
                active_range=(datetime(2014, 1, 1, 0, 0), InfDateTime('inf')))
    c_disp = ChemicalDispersion(.3,
                                active_range=(datetime(2014, 1, 1, 0, 0),
                                              datetime(2014, 1, 1, 0, 3)),
                                efficiency=0.2)
    weatherers = [Emulsification(),
                  Evaporation(Water(),
                              constant_wind(1, 0)),
                  burn,
                  c_disp,
                  skimmer]

    exp_order = [weatherers[ix] for ix in (3, 4, 2, 1, 0)]

    model.environment += [Water(), constant_wind(5, 0), Waves()]
    model.weatherers += weatherers

    # WeatheringData and FayGravityViscous automatically get added to
    # weatherers. Only do assertion on weatherers contained in list above
    assert list(model.weatherers.values())[:len(exp_order)] != exp_order

    model.setup_model_run()

    assert list(model.weatherers.values())[:len(exp_order)] == exp_order

    # check second time around order is kept
    model.rewind()
    assert list(model.weatherers.values())[:len(exp_order)] == exp_order

    # Burn, ChemicalDispersion are at same sorting level so appending
    # another Burn to the end of the list will sort it to be just after
    # ChemicalDispersion so index 2
    burn = Burn(50, 1, active_range=(datetime(2014, 1, 1, 0, 0),
                                     InfDateTime('inf')))
    exp_order.insert(3, burn)

    model.weatherers += exp_order[3]  # add this and check sorting still works
    assert list(model.weatherers.values())[:len(exp_order)] != exp_order

    model.setup_model_run()

    assert list(model.weatherers.values())[:len(exp_order)] == exp_order


class TestValidateModel():
    ''' Group several model validation tests in one place '''
    start_time = datetime(2015, 1, 1, 12, 0)

    def test_validate_model_spills_time_mismatch_warning(self):
        '''
        test warning messages output for no spills and model start time
        mismatch with release time
        '''
        model = Model(start_time=self.start_time)
        (msgs, isvalid) = model.check_inputs()

        print(model.environment)
        print(msgs, isvalid)
        assert len(msgs) == 1 and isvalid
        assert ('{0} contains no spills'.format(model.name) in msgs[0])

        model.spills += Spill(Release(self.start_time + timedelta(hours=1), 1))
        (msgs, isvalid) = model.check_inputs()

        assert len(msgs) == 2 and isvalid
        assert ('{} has release time after model start time'
                .format(model.spills[0].name)
                in msgs[0])

        model.spills[0].release_time = self.start_time - timedelta(hours=1)
        (msgs, isvalid) = model.check_inputs()

        assert len(msgs) == 1 and not isvalid
        assert ('{} has release time before model start time'
                .format(model.spills[0].name)
                in msgs[0])

    def make_model_incomplete_waves(self):
        '''
        create a model with waves objects with no referenced wind, water.
        Include Spill so we don't get warnings for it
        '''
        model = Model(start_time=self.start_time)
        model.spills += Spill(Release(self.start_time, 1))

        waves = Waves()
        model.environment += waves

        return (model, waves)

    @pytest.mark.parametrize("obj_make_default_refs", (False, True))
    def test_validate_model_env_obj(self, obj_make_default_refs):
        '''
        test that Model is invalid if make_default_refs is True and referenced
        objects are not in model's environment collection
        '''
        # object is complete but model must contain
        (model, waves) = self.make_model_incomplete_waves()
        waves.water = Water()
        waves.wind = constant_wind(5, 0)

        assert len(model.environment) == 1

        waves.make_default_refs = obj_make_default_refs
        (msgs, isvalid) = model.validate()
        print(msgs)

        if obj_make_default_refs:
            assert not isvalid
            assert len(msgs) > 0
            assert ('warning: Model: water not found in environment collection'
                    in msgs)
            assert ('warning: Model: wind not found in environment collection'
                    in msgs)
        else:
            assert isvalid
            assert len(msgs) == 0

    def test_model_weatherer_off(self):
        model = Model(start_time=self.start_time)
        model.weatherers += Evaporation(on=False)

        print(model.validate())


class Test_add_weathering(object):
    """
    tests for the add_weathering method
    """
    def test_add_standard(self):
        model = Model()
        model.add_weathering()

        assert len(model.weatherers) == \
            len(gnome.weatherers.standard_weatherering_sets['standard'])

    def test_add_standard2(self):
        model = Model()
        model.add_weathering('standard')

        assert len(model.weatherers) == \
            len(gnome.weatherers.standard_weatherering_sets['standard'])

    def test_add_invalid(self):
        model = Model()
        with pytest.raises(ValueError):
            model.add_weathering('fred')

    def test_add_one(self):
        model = Model()
        model.add_weathering(['evaporation'])

        assert len(model.weatherers) == 1
        assert isinstance(model.weatherers[0], Evaporation)

    def test_add_two(self):
        model = Model()
        model.add_weathering(['evaporation',
                              'dispersion'])

        assert len(model.weatherers) == 2
        assert isinstance(model.weatherers[0], Evaporation)
        assert isinstance(model.weatherers[1], NaturalDispersion)

    def test_water_missing(self):
        model = Model()
        model.add_weathering(['evaporation'])
        with pytest.raises(ReferencedObjectNotSet):
            model.full_run()


if __name__ == '__main__':

    # test_all_movers()
    # test_release_at_right_time()
    # test_simple_run_with_image_output()

    test_simple_run_with_image_output_uncertainty()
