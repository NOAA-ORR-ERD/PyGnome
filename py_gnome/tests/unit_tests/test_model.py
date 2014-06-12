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

from gnome.basic_types import datetime_value_2d
from gnome.utilities import inf_datetime
from gnome.utilities.remote_data import get_datafile

import gnome.map
from gnome.environment import Wind, Tide
from gnome.model import Model

from gnome.spill import Spill, SpatialRelease, point_line_release_spill
from gnome.spill.elements import floating

from gnome.movers import SimpleMover, RandomMover, WindMover, CatsMover

from gnome.weatherers import Weatherer
from gnome.outputters import Renderer


basedir = os.path.dirname(__file__)
datadir = os.path.join(basedir, 'sample_data')
tides_dir = os.path.join(datadir, 'tides')
lis_dir = os.path.join(datadir, 'long_island_sound')

testmap = os.path.join(basedir, '../sample_data', 'MapBounds_Island.bna')


@pytest.fixture(scope='module')
def model(sample_model):
    '''
    Utility to setup up a simple, but complete model for tests
    '''
    images_dir = os.path.join(basedir, 'Test_images')

    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    model = sample_model['model']
    rel_start_pos = sample_model['release_start_pos']
    rel_end_pos = sample_model['release_end_pos']

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

    model.spills += point_line_release_spill(10, (0.0, 0.0, 0.0), model.start_time,
                        end_release_time=model.start_time + model.duration)

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

    model.map = gnome.map.MapFromBNA(testmap, refloat_halflife=6)  # hours
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


def test_simple_run_with_image_output():
    '''
    Pretty much all this tests is that the model will run and output images
    '''
    images_dir = os.path.join(basedir, 'Test_images')

    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    start_time = datetime(2012, 9, 15, 12, 0)

    # the land-water map
    gnome_map = gnome.map.MapFromBNA(testmap, refloat_halflife=6)  # hours
    renderer = gnome.outputters.Renderer(testmap, images_dir, size=(400, 300))

    model = Model(time_step=timedelta(minutes=15),
                  start_time=start_time, duration=timedelta(hours=1),
                  map=gnome_map,
                  uncertain=False, cache_enabled=False,
                  )

    model.outputters += renderer
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
        print 'calling step'
        try:
            image_info = model.step()
            num_steps_output += 1
            print image_info
        except StopIteration:
            print 'Done with the model run'
            break

    # There is the zeroth step, too.
    calculated_steps = (model.duration.total_seconds() / model.time_step) + 1
    assert num_steps_output == calculated_steps


def test_simple_run_with_image_output_uncertainty():
    '''
    Pretty much all this tests is that the model will run and output images
    '''
    images_dir = os.path.join(basedir, 'Test_images2')

    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    start_time = datetime(2012, 9, 15, 12, 0)

    # the land-water map
    gmap = gnome.map.MapFromBNA(testmap, refloat_halflife=6)  # hours
    renderer = gnome.outputters.Renderer(testmap, images_dir, size=(400, 300))

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
    c_data = get_datafile(os.path.join(lis_dir, 'tidesWAC.CUR'))
    model.movers += CatsMover(c_data)
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


def test_full_run(model):
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
    num_images = len(os.listdir(os.path.join(basedir, 'Test_images')))
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

    tide_file = get_datafile(os.path.join(tides_dir, 'CLISShio.txt'))
    tide_ = Tide(filename=tide_file)

    d_file = get_datafile(os.path.join(lis_dir, 'tidesWAC.CUR'))
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


def test_callback_add_mover_midrun(model):
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
        assert step['step_num'] == model.current_time_step

    assert True


def test_all_weatherers_in_model(model):
    '''
    test model run with weatherer
    '''
    weatherer = Weatherer()
    model.weatherers += weatherer
    print 'model.weatherers:', model.weatherers

    model.full_run()

    expected_keys = {'mass_components', 'half_lives'}
    assert expected_keys.issubset(model.spills.LE_data)


if __name__ == '__main__':

    # test_all_movers()
    # test_release_at_right_time()
    # test_simple_run_with_image_output()

    test_simple_run_with_image_output_uncertainty()
