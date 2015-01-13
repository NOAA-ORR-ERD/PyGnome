#!/usr/bin/env python

'''
test code for the model class
'''
import os
import shutil
from datetime import datetime, timedelta

import numpy
np = numpy

import pytest
from pytest import raises

from gnome.basic_types import datetime_value_2d, oil_status
from gnome.utilities import inf_datetime
from gnome.persist import load

import gnome.map
from gnome.environment import Wind, Tide, constant_wind, Water
from gnome.model import Model

from gnome.spill import Spill, SpatialRelease, point_line_release_spill
from gnome.spill.elements import floating

from gnome.movers import SimpleMover, RandomMover, WindMover, CatsMover

from gnome.weatherers import (HalfLifeWeatherer,
                              Evaporation,
                              Dispersion,
                              Burn,
                              Skimmer)
from gnome.outputters import Renderer, GeoJson

from conftest import sample_model_weathering, testdata


@pytest.fixture(scope='function')
def model(sample_model_fcn, dump):
    '''
    Utility to setup up a simple, but complete model for tests
    '''
    images_dir = os.path.join(dump, 'Test_images')

    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    model = sample_model_fcn['model']
    rel_start_pos = sample_model_fcn['release_start_pos']
    rel_end_pos = sample_model_fcn['release_end_pos']

    model.cache_enabled = True
    model.uncertain = False

    model.outputters += Renderer(model.map.filename, images_dir,
                                 size=(400, 300))

    N = 10  # a line of ten points
    line_pos = np.zeros((N, 3), dtype=np.float64)
    line_pos[:, 0] = np.linspace(rel_start_pos[0], rel_end_pos[0], N)
    line_pos[:, 1] = np.linspace(rel_start_pos[1], rel_end_pos[1], N)

    # print start_points

    release = SpatialRelease(start_position=line_pos,
                             release_time=model.start_time)

    model.spills += Spill(release)

    return model


def test_init():
    model = Model()


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


def test_model_time_and_current_time_in_sc():
    model = Model()
    model.start_time = datetime.now()

    assert model.current_time_step == -1
    assert model.model_time == model.start_time

    for step in range(4):
        model.step()

        assert model.current_time_step == step
        assert model.model_time == model.start_time + timedelta(seconds=step
                                                            * model.time_step)

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
    model.spills += point_line_release_spill(10, (0.0, 0.0, 0.0),
                                             model.start_time,
                                             end_release_time=end_release_time)

    model.movers += SimpleMover(velocity=(1., -1., 0.0))

    print '\n---------------------------------------------'
    print 'model_start_time: {0}'.format(model.start_time)

    prev_rel = len(model.spills.LE('positions'))
    for step in model:
        new_particles = len(model.spills.LE('positions')) - prev_rel
        if new_particles > 0:
            assert np.all(model.spills.LE('positions')[-new_particles:, :] ==
                          0)
            assert np.all(model.spills.LE('age')[-new_particles:] == 0)
            #assert np.all(model.spills.LE('age')[-new_particles:] ==
            #            (model.model_time + timedelta(seconds=model.time_step)
            #             - model.start_time).seconds)

        if prev_rel > 0:
            assert np.all(model.spills.LE('positions')[:prev_rel, :2] != 0)
            assert np.all(model.spills.LE('age')[:prev_rel] >= model.time_step)

        prev_rel = len(model.spills.LE('positions'))

        print 'current_time_stamp: {0}'.format(
                                        model.spills.LE('current_time_stamp'))
        print 'particle ID: {0}'.format(model.spills.LE('id'))
        print 'positions: \n{0}'.format(model.spills.LE('positions'))
        print 'age: \n{0}'.format(model.spills.LE('age'))
        print 'just ran: %s' % step
        print 'particles released: %s' % new_particles
        print '---------------------------------------------'

    print '\n==============================================='


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

    model.map = gnome.map.GnomeMap()
    a_mover = SimpleMover(velocity=(1., 2., 0.))

    model.movers += a_mover
    assert len(model.movers) == 1

    spill = point_line_release_spill(num_elements=10,
            start_position=(0., 0., 0.), release_time=start_time)

    model.spills += spill
    assert len(model.spills) == 1

    # model.add_spill(spill)

    model.start_time = spill.release.release_time

    # test iterator
    for step in model:
        print 'just ran time step: %s' % model.current_time_step
        assert step['step_num'] == model.current_time_step

    pos = np.copy(model.spills.LE('positions'))

    # rewind and run again:
    print 'rewinding'
    model.rewind()

    # test iterator is repeatable
    for step in model:
        print 'just ran time step: %s' % model.current_time_step
        assert step['step_num'] == model.current_time_step

    assert np.all(model.spills.LE('positions') == pos)


def test_simple_run_with_map():
    '''
    pretty much all this tests is that the model will run
    '''

    start_time = datetime(2012, 9, 15, 12, 0)

    model = Model()

    model.map = gnome.map.MapFromBNA(testdata['MapFromBNA']['testmap'],
                                     refloat_halflife=6)  # hours
    a_mover = SimpleMover(velocity=(1., 2., 0.))

    model.movers += a_mover
    assert len(model.movers) == 1

    spill = point_line_release_spill(num_elements=10,
                                     start_position=(0., 0., 0.),
                                     release_time=start_time)

    model.spills += spill
    assert len(model.spills) == 1
    model.start_time = spill.release.release_time

    # test iterator
    for step in model:
        print 'just ran time step: %s' % step
        assert step['step_num'] == model.current_time_step

    # reset and run again
    model.reset()

    # test iterator is repeatable
    for step in model:
        print 'just ran time step: %s' % step
        assert step['step_num'] == model.current_time_step


def test_simple_run_with_image_output(dump):
    '''
    Pretty much all this tests is that the model will run and output images
    '''
    images_dir = os.path.join(dump, 'Test_images')

    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    start_time = datetime(2012, 9, 15, 12, 0)

    # the land-water map
    gnome_map = gnome.map.MapFromBNA(testdata['MapFromBNA']['testmap'],
                                     refloat_halflife=6)  # hours
    renderer = gnome.outputters.Renderer(testdata['MapFromBNA']['testmap'],
                                         images_dir, size=(400, 300))
    geo_json = GeoJson(output_dir=images_dir)

    model = Model(time_step=timedelta(minutes=15),
                  start_time=start_time, duration=timedelta(hours=1),
                  map=gnome_map,
                  uncertain=False, cache_enabled=False,
                  )

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

    spill = Spill(SpatialRelease(start_position=start_points,
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
            print 'Done with the model run'
            break

    # There is the zeroth step, too.
    calculated_steps = (model.duration.total_seconds() / model.time_step) + 1
    assert num_steps_output == calculated_steps


def test_simple_run_with_image_output_uncertainty(dump):
    '''
    Pretty much all this tests is that the model will run and output images
    '''
    images_dir = os.path.join(dump, 'Test_images2')

    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    start_time = datetime(2012, 9, 15, 12, 0)

    # the land-water map
    gmap = gnome.map.MapFromBNA(testdata['MapFromBNA']['testmap'],
                                refloat_halflife=6)  # hours
    renderer = gnome.outputters.Renderer(testdata['MapFromBNA']['testmap'],
                                         images_dir, size=(400, 300))

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

    release = SpatialRelease(start_position=start_points,
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
            print image_info
        except StopIteration:
            print 'Done with the model run'
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
        print temp

    # test our iter and len object methods
    assert len(model.movers) == 2
    assert len([m for m in model.movers]) == 2
    for (m1, m2) in zip(model.movers, [mover_1, mover_2]):
        assert m1 == m2

    # test our add objectlist methods
    model.movers += [mover_3, mover_4]
    assert [m for m in model.movers] == [mover_1, mover_2, mover_3,
            mover_4]

    # test our remove object methods
    del model.movers[mover_3.id]
    assert [m for m in model.movers] == [mover_1, mover_2, mover_4]
    with raises(KeyError):
        # our key should also be gone after the delete
        temp = model.movers[mover_3.id]
        print temp

    # test our replace method
    model.movers[mover_2.id] = mover_3
    assert [m for m in model.movers] == [mover_1, mover_3, mover_4]
    assert model.movers[mover_3.id] == mover_3
    with raises(KeyError):
        # our key should also be gone after the delete
        temp = model.movers[mover_2.id]
        print temp


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

    release_time = start_time + timedelta(seconds=model.time_step
            * release_delay)
    model.spills += point_line_release_spill(num_elements=10,
                                             start_position=start_loc,
                                             release_time=release_time)

    # the land-water map
    model.map = gnome.map.GnomeMap()  # the simplest of maps

    # simple mover
    model.movers += SimpleMover(velocity=(1., -1., 0.))
    assert len(model.movers) == 1

    # random mover
    model.movers += RandomMover(diffusion_coef=100000)
    assert len(model.movers) == 2

    # wind mover
    series = np.array((start_time, (10, 45)),
                      dtype=datetime_value_2d).reshape((1, ))
    model.movers += WindMover(Wind(timeseries=series,
                              units='meter per second'))
    assert len(model.movers) == 3

    # CATS mover
    model.movers += CatsMover(testdata['CatsMover']['curr'])
    assert len(model.movers) == 4

    # run the model all the way...
    num_steps_output = 0
    for step in model:
        num_steps_output += 1
        print 'running step:', step

    # test release happens correctly for all cases
    if release_delay < duration:
        # at least one get_move has been called after release
        assert np.all(model.spills.LE('positions')[:, :2]
                      != start_loc[:2])
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
    WindMover is defined as a linear operation - defining a model
    with a single WindMover with 15 knot wind is equivalent to defining
    a model with three WindMovers each with 5 knot wind. Or any number of
    WindMover's such that the sum of their magnitude is 15knots and the
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
    model1 = Model()
    model1.duration = timedelta(hours=1)
    model1.time_step = timedelta(hours=1)
    model1.start_time = start_time
    model1.spills += point_line_release_spill(num_elements=num_LEs,
                        start_position=(1., 2., 0.), release_time=start_time,
                        element_type=floating(windage_persist=wind_persist))

    model1.movers += WindMover(Wind(timeseries=series1, units=units))

    model2 = Model()
    model2.duration = timedelta(hours=10)
    model2.time_step = timedelta(hours=1)
    model2.start_time = start_time

    model2.spills += point_line_release_spill(num_elements=num_LEs,
            start_position=(1., 2., 0.), release_time=start_time,
            element_type=floating(windage_persist=wind_persist))

    # todo: CHECK RANDOM SEED
    # model2.movers += WindMover(Wind(timeseries=series1, units=units))

    model2.movers += WindMover(Wind(timeseries=series2, units=units))
    model2.movers += WindMover(Wind(timeseries=series2, units=units))
    model2.movers += WindMover(Wind(timeseries=series3, units=units))

    while True:
        try:
            model1.next()
        except StopIteration:
            print 'Done model1 ..'
            break

    while True:
        try:
            model2.next()
        except StopIteration:
            print 'Done model2 ..'
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

    rel_var_error = np.linalg.norm(np.var(model2.spills.LE('positions'), 0)
                                   - np.var(model1.spills.LE('positions'), 0))
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
    model.spills += point_line_release_spill(num_elements=5,
            start_position=(0, 0, 0), release_time=release_time)

    # and another that starts later..

    model.spills += point_line_release_spill(num_elements=4,
            start_position=(0, 0, 0), release_time=start_time
            + timedelta(hours=2))

    # Add a Wind mover:
    series = np.array((start_time, (10, 45)),
                      dtype=datetime_value_2d).reshape((1, ))
    model.movers += WindMover(Wind(timeseries=series, units=units))

    for step in model:
        print 'running a step'
        assert step['step_num'] == model.current_time_step

        for sc in model.spills.items():
            print 'num_LEs', len(sc['positions'])


def test_release_at_right_time():
    '''
    Tests that the elements get released when they should

    There are issues in that we want the elements to show
    up in the output for a given time step if they were
    supposed to be released then.  Particularly for the
    first time step of the model.
    '''
    # default to now, rounded to the nearest hour
    seconds_in_minute = 60
    minutes_in_hour = 60
    seconds_in_hour = seconds_in_minute * minutes_in_hour

    start_time = datetime(2013, 1, 1, 0)
    time_step = 2 * seconds_in_hour

    model = Model(time_step=time_step, start_time=start_time,
                  duration=timedelta(hours=12))

    # add a spill that starts right when the run begins

    model.spills += \
        point_line_release_spill(num_elements=12,
            start_position=(0, 0, 0), release_time=datetime(2013, 1, 1,
            0), end_release_time=datetime(2013, 1, 1, 6))

    # before the run - no elements present since data_arrays get defined after
    # 1st step (prepare_for_model_run):

    assert model.spills.items()[0].num_released == 0

    model.step()
    assert model.spills.items()[0].num_released == 4

    model.step()
    assert model.spills.items()[0].num_released == 8

    model.step()
    assert model.spills.items()[0].num_released == 12

    model.step()
    assert model.spills.items()[0].num_released == 12


def test_full_run(model, dump):
    'Test doing a full run'
    # model = setup_simple_model()

    results = model.full_run()
    print results

    # check the number of time steps output is right
    # there is the zeroth step, too.
    calculated_steps = (model.duration.total_seconds() / model.time_step) + 1
    assert len(results) == calculated_steps

    # check if the images are there:
    # (1 extra for background image)
    num_images = len(os.listdir(os.path.join(dump, 'Test_images')))
    assert num_images == model.num_time_steps + 1


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
    model.movers += WindMover(Wind(timeseries=series, units=units))

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
        assert mover.active_start == inf_datetime.InfDateTime('-inf')
        assert mover.active_stop == inf_datetime.InfDateTime('inf')

        if hasattr(mover, 'wind'):
            assert mover.wind in model.environment

        if hasattr(mover, 'tide'):
            if mover.tide is not None:
                assert mover.tide in model.environment

    # Add a mover with user defined active_start / active_stop values
    # - these should not be updated

    active_on = model.start_time + timedelta(hours=1)
    active_off = model.start_time + timedelta(hours=4)
    custom_mover = SimpleMover(velocity=(1., -1., 0.),
                               active_start=active_on,
                               active_stop=active_off)
    model.movers += custom_mover

    assert model.movers[custom_mover.id].active_start == active_on
    assert model.movers[custom_mover.id].active_stop == active_off


def test_callback_add_mover_midrun():
    'Test callback after add mover called midway through the run'
    model = Model()
    model.start_time = datetime(2012, 1, 1, 0, 0)
    model.duration = timedelta(hours=10)
    model.time_step = timedelta(hours=1)

    # start_loc = (1.0, 2.0, 0.0)  # random non-zero starting points

    # model = setup_simple_model()

    for i in range(2):
        model.step()

    assert model.current_time_step > -1

    # now add another mover and make sure model rewinds
    model.movers += SimpleMover(velocity=(2., -2., 0.))
    assert model.current_time_step == -1


def test_simple_run_no_spills(model):
    # model = setup_simple_model()

    for spill in model.spills:
        del model.spills[spill.id]

    assert len(model.spills) == 0

    for step in model:
        print 'just ran time step: %s' % model.current_time_step
        assert step['Renderer']['step_num'] == model.current_time_step

    assert True


def test_all_weatherers_in_model(model):
    '''
    test model run with weatherer
    '''
    model.weatherers += HalfLifeWeatherer()
    print 'model.weatherers:', model.weatherers

    model.full_run()

    expected_keys = {'mass_components'}
    assert expected_keys.issubset(model.spills.LE_data)


def test_setup_model_run(model):
    'turn of movers/weatherers and ensure data_arrays change'
    model.rewind()
    model.step()
    exp_keys = {'windages', 'windage_range', 'mass_components',
                'windage_persist'}
    # no exp_keys in model data_arrays
    assert not exp_keys.intersection(model.spills.LE_data)

    model.weatherers += HalfLifeWeatherer()
    model.movers += gnome.movers.constant_wind_mover(1., 0.)
    model.rewind()
    model.step()
    assert exp_keys.issubset(model.spills.LE_data)

    model.movers[-1].on = False
    model.weatherers[-1].on = False
    model.rewind()
    model.step()
    assert not exp_keys.intersection(model.spills.LE_data)


def test_contains_object(sample_model_fcn):
    '''
    Test that we can find all contained object types with a model.
    '''
    model = sample_model_weathering(sample_model_fcn, 'ALAMO')

    gnome_map = model.map = gnome.map.GnomeMap()    # make it all water

    rel_time = model.spills[0].get('release_time')
    model.start_time = rel_time - timedelta(hours=1)
    model.duration = timedelta(days=1)

    water, wind = Water(), constant_wind(1., 0)
    model.water = water
    model.environment += wind

    et = floating(substance=model.spills[0].get('substance').name)
    sp = point_line_release_spill(500, (0, 0, 0),
                                  rel_time + timedelta(hours=1),
                                  element_type=et,
                                  amount=100,
                                  units='tons')
    rel = sp.release
    initializers = et.initializers
    model.spills += sp

    movers = [m for m in model.movers]

    evaporation = Evaporation(model.water, model.environment[0])
    dispersion, burn = Dispersion(), Burn()
    skim_start = sp.get('release_time') + timedelta(hours=1)
    skimmer = Skimmer(.5*sp.amount, units=sp.units, efficiency=0.3,
                      active_start=skim_start,
                      active_stop=skim_start + timedelta(hours=1))
    model.weatherers += [evaporation, dispersion, burn, skimmer]

    renderer = Renderer(images_dir='Test_images',
                        size=(400, 300))
    model.outputters += renderer

    for o in (gnome_map, sp, rel, et,
              water, wind,
              evaporation, dispersion, burn, skimmer,
              renderer):
        assert model.contains_object(o.id)

    for o in initializers:
        assert model.contains_object(o.id)

    for o in movers:
        assert model.contains_object(o.id)


def make_skimmer(spill, delay_hours=1, duration=2):
    'make a skimmer for sample model tests'
    rel_time = spill.get('release_time')
    skim_start = rel_time + timedelta(hours=delay_hours)
    amount = spill.amount
    units = spill.units
    skimmer = Skimmer(.5*amount, units=units, efficiency=0.3,
                      active_start=skim_start,
                      active_stop=skim_start + timedelta(hours=duration))
    return skimmer


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
    model = sample_model_weathering(sample_model_fcn, 'ALAMO')
    model.map = gnome.map.GnomeMap()    # make it all water
    model.uncertain = False
    rel_time = model.spills[0].get('release_time')
    model.start_time = rel_time - timedelta(hours=1)
    model.duration = timedelta(days=1)

    et = floating(substance=model.spills[0].get('substance').name)
    cs = point_line_release_spill(500, (0, 0, 0),
                                  rel_time + delay,
                                  end_release_time=(rel_time + delay +
                                                    timedelta(hours=1)),
                                  element_type=et,
                                  amount=1,
                                  units='tonnes')
    model.spills += cs
    model.water = Water()
    model.environment += constant_wind(1., 0)
    skimmer = make_skimmer(model.spills[0])
    model.weatherers += [Evaporation(model.water,
                                     model.environment[0]),
                         Dispersion(),
                         Burn(),
                         skimmer]
    # model.full_run()
    for step in model:
        for sc in model.spills.items():
            unaccounted = sc['status_codes'] != oil_status.in_water
            sum_ = sc['mass'][unaccounted].sum()
            for key in sc.weathering_data:
                if 'avg_' != key[:4] and 'amount_released' != key:
                    sum_ += sc.weathering_data[key]
            assert abs(sum_ - sc.weathering_data['amount_released']) < 1.e-6

        print "completed step {0}".format(step)
        print sc.weathering_data


@pytest.mark.parametrize(("s0", "s1"), [("ALAMO", "ALAMO"),
                                        ("ALAMO", "AGUA DULCE")])
def test_two_substance_spills_weathering(sample_model_fcn, s0, s1):
    '''
    only tests data arrays are correct and we don't end up with stale data
    in substance_data structure of spill container. It models each substance
    independently
    '''
    model = sample_model_weathering(sample_model_fcn, s0)
    model.map = gnome.map.GnomeMap()    # make it all water
    model.uncertain = False
    rel_time = model.spills[0].get('release_time')
    model.duration = timedelta(days=1)

    et = floating(substance=s1)
    cs = point_line_release_spill(500, (0, 0, 0),
                                  rel_time,
                                  end_release_time=(rel_time +
                                                    timedelta(hours=1)),
                                  element_type=et,
                                  amount=1,
                                  units='tonnes')
    model.spills += cs
    model.water = Water()
    model.environment += constant_wind(1., 0)
    model.weatherers += Evaporation(model.water, model.environment[0])
    if s0 == s1:
        '''
        multiple substances will not work with Skimmer
        '''
        skimmer = make_skimmer(model.spills[0], 2)
        model.weatherers += [Dispersion(),
                             Burn(),
                             skimmer]

    # model.full_run()
    for step in model:
        for sc in model.spills.items():
            # If LEs are marked as 'skim', add them to sum_ since the mass must
            # be accounted for in the assertion
            unaccounted = sc['status_codes'] != oil_status.in_water
            sum_ = sc['mass'][unaccounted].sum()
            print 'starting sum_: ', sum_
            for key in sc.weathering_data:
                if 'avg_' != key[:4] and 'amount_released' != key:
                    sum_ += sc.weathering_data[key]
            assert abs(sum_ - sc.weathering_data['amount_released']) < 1.e-6

        print "completed step {0}".format(step)
        print sc.weathering_data


def test_weathering_data_attr():
    '''
    weathering_data is initialized/written if we have weatherers
    '''
    ts = 900
    s1_rel = datetime.now().replace(microsecond=0)
    s2_rel = s1_rel + timedelta(seconds=ts)
    model = Model(time_step=ts, start_time=s1_rel)
    s = [point_line_release_spill(10, (0, 0, 0), s1_rel),
         point_line_release_spill(10, (0, 0, 0), s2_rel)]
    model.spills += s
    model.step()

    for sc in model.spills.items():
        assert sc.weathering_data == {}

    model.water = Water()
    model.environment += constant_wind(0., 0)
    model.weatherers += [Evaporation(model.water,
                                     model.environment[0])]

    # use different element_type and initializers for both spills
    s[0].amount = 10.0
    s[0].units = 'kg'
    model.rewind()
    model.step()
    for sc in model.spills.items():
        assert sc.weathering_data['floating'] == sum(sc['mass'])
        assert sc.weathering_data['floating'] == s[0].amount

    s[1].amount = 5.0
    s[1].units = 'kg'
    model.rewind()
    exp_rel = 0.0
    for ix in range(2):
        model.step()
        exp_rel += s[ix].amount
        for sc in model.spills.items():
            assert sc.weathering_data['floating'] == sum(sc['mass'])
            assert sc.weathering_data['floating'] == exp_rel
    model.rewind()
    assert sc.weathering_data == {}

    # weathering data is now empty for all steps
    del model.weatherers[0]
    for ix in range(2):
        for sc in model.spills.items():
            assert not sc.weathering_data


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
            spill.set('initializers', [])
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

    def test_load_location_file(self):
        '''
        create a model
        load save file from script_boston which contains a spill. Then merge
        the created model into the model loaded from save file
        '''
        m = Model()
        m.water = Water()
        m.environment += constant_wind(1., 0.)
        m.weatherers += Evaporation(m.water, m.environment[-1])
        m.spills += point_line_release_spill(10, (0, 0, 0),
                                             datetime(2014, 1, 1, 12, 0))

        here = os.path.dirname(__file__)
        sample_save_file = \
            os.path.join(here,
                         '../../scripts/script_boston/save_model/Model.json')
        if os.path.exists(sample_save_file):
            model = load(sample_save_file)
            assert model.water is None

            model.merge(m)
            assert m.water is model.water
            for oc in m._oc_list:
                for item in getattr(m, oc):
                    model_oc = getattr(model, oc)
                    assert item is model_oc[item.id]

            for spill in m.spills:
                assert spill is model.spills[spill.id]

            # merge the other way and ensure model != m
            m.merge(model)
            assert model != m

if __name__ == '__main__':

    # test_all_movers()
    # test_release_at_right_time()
    # test_simple_run_with_image_output()

    test_simple_run_with_image_output_uncertainty()
