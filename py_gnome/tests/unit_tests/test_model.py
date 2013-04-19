#!/usr/bin/env python

"""
test code for the model class

not a lot to test by itself, but a start

"""
import pytest
import os, shutil
from datetime import datetime, timedelta
import numpy as np
import gnome.model
import gnome.map
from gnome import movers, environment
import gnome.spill
from gnome.spill import SpatialReleaseSpill

datadir = os.path.join(os.path.dirname(__file__), r"SampleData")

def test_init():
    model = gnome.model.Model()
    
def test_start_time():
    model = gnome.model.Model()

    st = datetime.now()
    model.start_time = st
    assert model.start_time == st
    
    model.step()
    
    st = datetime(2012, 8, 12, 13)
    model.start_time = st
    
    assert model.current_time_step == -1
    assert model.start_time == st

def test_timestep():
    model = gnome.model.Model()

    ts = timedelta(hours=1)
    model.time_step = ts
    assert model.time_step == ts.total_seconds()
    
    dur = timedelta(days=3)
    model.duration = dur
    assert model._duration == dur

def test_end_time():
    """
    test if the duration is properly computed when the end_time property is set.
    """
    

def test_simple_run_rewind():
    """
    pretty much all this tests is that the model will run
    and the seed is set during first run, then set correctly 
    after it is rewound and run again
    """
    
    start_time = datetime(2012, 9, 15, 12, 0)
    
    model = gnome.model.Model()
    
    model.map = gnome.map.GnomeMap()
    a_mover = movers.simple_mover.SimpleMover(velocity=(1.0, 2.0, 0.0))
    
    model.movers += a_mover
    assert len(model.movers) == 1

    spill = gnome.spill.SurfaceReleaseSpill(num_elements=10,
                                            start_position = (0.0, 0.0, 0.0),
                                            release_time = start_time,
                                            )
    
    model.spills += spill
    assert len(model.spills) == 1
    #model.add_spill(spill) 

    model.start_time = spill.release_time

    # test iterator:
    for step in model:
        print "just ran time step: %s"%model.current_time_step

    pos = np.copy( model.spills.LE('positions'))
    
    # rewind and run again:
    print "rewinding"
    model.rewind()
    # test iterator:
    for step in model:
        print "just ran time step: %s"%model.current_time_step
        
    assert np.all( model.spills.LE('positions') == pos)
    
def test_simple_run_with_map():
    """
    pretty much all this tests is that the model will run
    """
    
    start_time = datetime(2012, 9, 15, 12, 0)
    
    model = gnome.model.Model()
    
    model.map = gnome.map.MapFromBNA(os.path.join(datadir, 'MapBounds_Island.bna'),
                                refloat_halflife=6*3600, #seconds
                                )
    a_mover = movers.simple_mover.SimpleMover(velocity=(1.0, 2.0, 0.0))
    
    model.movers += a_mover
    assert len(model.movers) == 1

    spill = gnome.spill.SurfaceReleaseSpill(num_elements=10,
                                            start_position = (0.0, 0.0, 0.0),
                                            release_time = start_time,
                                            )
    
    model.spills += spill
    #model.add_spill(spill)
    assert len(model.spills) == 1
    model.start_time = spill.release_time
    
    # test iterator:
    for step in model:
        print "just ran time step: %s"%step

    # reset and run again:
    model.reset()
    # test iterator:
    for step in model:
        print "just ran time step: %s"%step
        
    assert True

import gnome.utilities.map_canvas
from gnome.utilities.file_tools import haz_files

def test_simple_run_with_image_output():
    """
    pretty much all this tests is that the model will run and output images
    """
    
    # create a place for test images (cleaning out any old ones)
    images_dir = "Test_images"
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    start_time = datetime(2012, 9, 15, 12, 0)
    
    model = gnome.model.Model()
    model.duration = timedelta(hours=1)

    mapfile = os.path.join(datadir, 'MapBounds_Island.bna')

    # the land-water map
    model.map = gnome.map.MapFromBNA( mapfile,
                                      refloat_halflife=6*3600, #seconds
                                     )
    # the image output map
    polygons = haz_files.ReadBNA(mapfile, "PolygonSet")
    map = gnome.utilities.map_canvas.MapCanvas((400, 300), land_polygons=polygons)
    model.output_map = map
    
    a_mover = movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))
    model.movers += a_mover
    assert len(model.movers) == 1

    N = 10 # a line of ten points
    start_points = np.zeros((N, 3) , dtype=np.float64)
    start_points[:,0] = np.linspace(-127.1, -126.5, N)
    start_points[:,1] = np.linspace( 47.93, 48.1, N)
    #print start_points
    spill = gnome.spill.SpatialReleaseSpill(start_positions = start_points,
                                            release_time = start_time,
                                            )
    
    model.spills += spill
    #model.add_spill(spill)
    assert len(model.spills) == 1

    model.start_time = spill.release_time
    #image_info = model.next_image()

    num_steps_output = 0
    while True:
        print "calling next_image"
        try:
            image_info = model.next_image(images_dir)
            num_steps_output += 1
            print image_info
        except StopIteration:
            print "Done with the model run"
            break

    assert num_steps_output == (model.duration.total_seconds() / model.time_step) + 1 # there is the zeroth step, too.

def test_simple_run_with_image_output_uncertainty():
    """
    pretty much all this tests is that the model will run and output images
    """
    
    # create a place for test images (cleaning out any old ones)
    images_dir = "Test_images2"
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    start_time = datetime(2012, 9, 15, 12, 0)

    model = gnome.model.Model()
    model.duration = timedelta(hours=1)

    mapfile = os.path.join(datadir, 'MapBounds_Island.bna')

    # the land-water map
    model.map = gnome.map.MapFromBNA( mapfile,
                                      refloat_halflife=6*3600, #seconds
                                     )
    # the image output map
    polygons = haz_files.ReadBNA(mapfile, "PolygonSet")
    l__map = gnome.utilities.map_canvas.MapCanvas((400, 300), land_polygons=polygons)
    model.output_map = l__map

    a_mover = movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))
    model.movers += a_mover

    N = 10 # a line of ten points
    start_points = np.zeros((N, 3) , dtype=np.float64)
    start_points[:,0] = np.linspace(-127.1, -126.5, N)
    start_points[:,1] = np.linspace( 47.93, 48.1, N)
    #print start_points
    spill = gnome.spill.SpatialReleaseSpill(start_positions = start_points,
                                            release_time = start_time,
                                            )

    model.spills += spill
    #model.add_spill(spill)
    model.start_time = spill.release_time
    #image_info = model.next_image()

    model.uncertain = True

    num_steps_output = 0
    while True:
        try:
            image_info = model.next_image(images_dir)
            num_steps_output += 1
            print image_info
        except StopIteration:
            print "Done with the model run"
            break

    assert num_steps_output == (model.duration.total_seconds() / model.time_step) + 1 # there is the zeroth step, too.
    ## fixme -- do an assertionlooking for red in images?

def test_mover_api():
    """
    Test the API methods for adding and removing movers to the model.
    """
    start_time = datetime(2012, 1, 1, 0, 0)

    model = gnome.model.Model()
    model.duration = timedelta(hours=12)
    model.time_step = timedelta(hours = 1)
    model.start_time = start_time

    mover_1 = movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))
    mover_2 = movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))
    mover_3 = movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))
    mover_4 = movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))

    # test our add object methods
    model.movers += mover_1
    model.movers += mover_2

    # test our get object methods
    assert(model.movers[mover_1.id] == mover_1)
    assert(model.movers[mover_2.id] == mover_2)
    with pytest.raises(KeyError):
        l__temp = model.movers['Invalid']

    # test our iter and len object methods
    assert len(model.movers) == 2
    assert len([m for m in model.movers]) == 2
    for m1, m2 in zip(model.movers, [mover_1, mover_2]):
        assert m1 == m2

    # test our add objectlist methods
    model.movers += [mover_3, mover_4]
    assert [m for m in model.movers] == [mover_1, mover_2, mover_3, mover_4]

    # test our remove object methods
    del model.movers[mover_3.id]
    assert [m for m in model.movers] == [mover_1, mover_2, mover_4]
    with pytest.raises(KeyError):
        # our key should also be gone after the delete
        l__temp = model.movers[mover_3.id]

    # test our replace method
    model.movers[mover_2.id] = mover_3
    assert [m for m in model.movers] == [mover_1, mover_3, mover_4]
    assert model.movers[mover_3.id] == mover_3
    with pytest.raises(KeyError):
        # our key should also be gone after the delete
        l__temp = model.movers[mover_2.id]


test_cases = [(datetime(2012, 1, 1, 0, 0), 0, 12 ), # model start_time, No. of time_steps after which LEs release, duration as No. of timesteps
              (datetime(2012, 1, 1, 0, 0), 12, 12),
              (datetime(2012, 1, 1, 0, 0), 13, 12)]

#test_cases = [(datetime(2012, 1, 1, 0, 0), 13, 12)]

@pytest.mark.parametrize(("start_time", "release_delay", "duration"), test_cases)
def test_all_movers(start_time, release_delay, duration):
    """
    a test that tests that all the movers at least can be run

    add new ones as they come along!
    """
    model = gnome.model.Model()
    model.time_step = timedelta(hours=1)
    model.duration = timedelta(seconds=model.time_step*duration)
    model.start_time = start_time
    start_loc = (1.0, 2.0, 0.0) # random non-zero starting points
    
    # a spill - release after 5 timesteps
    model.spills += gnome.spill.SurfaceReleaseSpill(num_elements=10,
                                                    start_position=start_loc,
                                                    release_time  = start_time + timedelta(seconds=model.time_step*release_delay),
                                                    )
    print "release_delay: {0}".format(release_delay)
    print "LE positions:"
    print model.spills.LE('positions')
    # model.spills += gnome.spill.PointReleaseSpill(num_LEs=10,
    #                                               start_position = (0.0, 0.0, 0.0),
    #                                               release_time = start_time,
    #                                               )
    # assert len(model.spills) == 1

    # the land-water map
    model.map = gnome.map.GnomeMap() # the simpleset of maps
    
    # simplemover
    model.movers += movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))
    assert len(model.movers) == 1

    # random mover
    model.movers += gnome.movers.RandomMover(diffusion_coef=100000)
    assert len(model.movers) == 2

    # wind mover
    series = np.array( (start_time, ( 10,   45) ),  dtype=gnome.basic_types.datetime_value_2d).reshape((1,))
    model.movers += gnome.movers.WindMover(environment.Wind(timeseries=series, units='meter per second'))
    assert len(model.movers) == 3
    
    # add CATS mover
    model.movers += movers.CatsMover(os.path.join(datadir, r"long_island_sound/tidesWAC.CUR"))
    assert len(model.movers) == 4
    
    # run the model all the way...
    num_steps_output = 0
    for step in model:
        num_steps_output += 1
        #print "running step:", step
        
    # test release happens correctly for all cases
    if release_delay < duration:    # at least one get_move has been called after release
        assert np.all(model.spills.LE('positions')[:,:2] != start_loc[:2])
       
    elif release_delay == duration: # particles are released after last step so no motion, only initial state
        assert np.all(model.spills.LE('positions') == start_loc)
        
    else:                           # release_delay > duration so nothing released though model ran
        assert len(model.spills.LE('positions') ) == 0
        
    assert num_steps_output == (model.duration.total_seconds() / model.time_step) + 1 # there is the zeroth step, too.


@pytest.mark.parametrize("wind_persist", [0, 900, 5])   # 0 is infinite persistence
def test_linearity_of_wind_movers(wind_persist):
    """
    WindMover is defined as a linear operation - defining a model
    with a single WindMover with 15 knot wind is equivalent to defining
    a model with three WindMovers each with 5 knot wind. Or any number of
    WindMover's such that the sum of their magnitude is 15knots and the
    direction of wind is the same for both cases.
    
    Below is an example which defines two models and runs them. In model2, there
    are multiple winds defined so the windage parameter is reset 3 times for one timestep.
    Since windage range and persistance do not change, this only has the effect of doing the
    same computation 3 times. However, the results are the same.
    
    The mean and variance of the positions for both models are close. As windage_persist is decreased,
    the values become closer. Setting windage_persist=0 gives the larged difference between them.
    """
    start_time = datetime(2012, 1, 1, 0, 0)
    series1= np.array( (start_time, ( 15,   45) ),  dtype=gnome.basic_types.datetime_value_2d).reshape((1,))
    series2= np.array( (start_time, (  6,   45) ),  dtype=gnome.basic_types.datetime_value_2d).reshape((1,))
    series3= np.array( (start_time, (  3,   45) ),  dtype=gnome.basic_types.datetime_value_2d).reshape((1,))
    
    num_LEs=1000
    model1= gnome.model.Model()
    model1.duration = timedelta(hours=1)
    model1.time_step = timedelta(hours = 1)
    model1.start_time = start_time
    model1.spills += gnome.spill.SurfaceReleaseSpill(num_elements=num_LEs,
                                                     start_position=(1.,2.,0.),
                                                     release_time  = start_time,
                                                     windage_persist=wind_persist
                                                     )
    model1.movers += gnome.movers.WindMover(gnome.environment.Wind(timeseries=series1, units='meter per second'))
    
    model2= gnome.model.Model()
    model2.duration = timedelta(hours=10)
    model2.time_step = timedelta(hours = 1)
    model2.start_time = start_time
    model2.spills += gnome.spill.SurfaceReleaseSpill(num_elements=num_LEs,
                                                     start_position=(1.,2.,0.),
                                                     release_time  = start_time,
                                                     windage_persist=wind_persist
                                                     )
    
    # todo: CHECK RANDOM SEED
    #model2.movers += gnome.movers.WindMover(gnome.environment.Wind(timeseries=series1, units='meter per second'))
    
    model2.movers += gnome.movers.WindMover(gnome.environment.Wind(timeseries=series2, units='meter per second'))
    model2.movers += gnome.movers.WindMover(gnome.environment.Wind(timeseries=series2, units='meter per second'))
    model2.movers += gnome.movers.WindMover(gnome.environment.Wind(timeseries=series3, units='meter per second'))
    
    while True:
        try: 
            model1.next()
        except StopIteration:
            print "Done model1 .."
            break
            
    while True:
        try: 
            model2.next()
        except StopIteration:
            print "Done model2 .."
            break
    
    # mean and variance at the end should be fairly close
    # look at the mean of the position vector. Assume m1 is truth and m2 is approximation - look at the
    # absolute error between mean position of m2 in the 2 norm.
    #rel_mean_error=np.linalg.norm( np.mean( model2.spills.LE('positions'), 0)-np.mean( model1.spills.LE('positions'), 0) )
    #assert rel_mean_error <= 0.5
    # Similary look at absolute error in variance of position of m2 in the 2 norm.  
    rel_var_error=np.linalg.norm( np.var( model2.spills.LE('positions'), 0)-np.var( model1.spills.LE('positions'), 0) )
    assert rel_var_error <= 0.001
    

def test_model_release_after_start():
    """

    This runs the model for a simple spill, that starts after the model starts

    """
    start_time = datetime(2013, 2, 22, 0)

    model = gnome.model.Model(time_step=60*30, # 30 minutes in seconds
                              start_time=start_time, # default to now, rounded to the nearest hour
                              duration=timedelta(hours=3),
                              )

    # add a spill that starts after the run begins.
    model.spills += gnome.spill.SurfaceReleaseSpill(num_elements = 5,
                                                    start_position = (0, 0, 0),
                                                    release_time=start_time+timedelta(hours=1))

    # and another that starts later..
    model.spills += gnome.spill.SurfaceReleaseSpill(num_elements = 4,
                                                    start_position = (0, 0, 0),
                                                    release_time=start_time+timedelta(hours=2))

    # Add a Wind mover:
    series = np.array( (start_time, ( 10,   45) ),
                      dtype=gnome.basic_types.datetime_value_2d).reshape((1,))
    model.movers += gnome.movers.WindMover(environment.Wind(timeseries=series,
                                           units='meter per second'))

    for step in model:
        print "running a step"
        for sc in model.spills.items():
            print "num_LEs", len(sc['positions'])

def test_release_at_right_time():
    """
    Tests that the elements get released when they should
    
    There are issues in that we want the elements to show
    up in the output for a given time step if they were
    supposed to be released then. Particularly for the
    first time step of the model.

    """
    start_time = datetime(2013, 1, 1, 0)
    time_step = 2*60*60 # 2 hour in seconds

    model = gnome.model.Model(time_step=time_step,
                              start_time=start_time, # default to now, rounded to the nearest hour
                              duration=timedelta(hours=12),
                              )

    # add a spill that starts right when the run begins
    model.spills += gnome.spill.SurfaceReleaseSpill(num_elements=12,
                                                    start_position=(0, 0, 0),
                                                    release_time=datetime(2013, 1, 1, 0),
                                                    end_release_time=datetime(2013, 1, 1, 6),
                                                    )
    # before the run:
    assert model.spills.items()[0].num_elements == 0

    model.step()
    assert model.spills.items()[0].num_elements == 4

    model.step()
    assert model.spills.items()[0].num_elements == 8

    model.step()
    assert model.spills.items()[0].num_elements == 12

    model.step()
    assert model.spills.items()[0].num_elements == 12


def test_callback_add_mover():
    """ Test callback after add mover """
    model = gnome.model.Model()
    model.time_step = timedelta(hours=1)
    model.duration = timedelta(hours=10)
    model.start_time = datetime(2012, 1, 1, 0, 0)
    start_loc = (1.0, 2.0, 0.0) # random non-zero starting points
    
    # add Movers
    model.movers += movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))
    series = np.array( (model.start_time, ( 10,   45) ),  dtype=gnome.basic_types.datetime_value_2d).reshape((1,))
    model.movers += movers.WindMover(environment.Wind(timeseries=series, units='meter per second'))
    tide_ = environment.Tide(filename=os.path.join( os.path.dirname(__file__), r"SampleData","tides","CLISShio.txt"))
    model.movers += movers.CatsMover(os.path.join(datadir, r"long_island_sound/tidesWAC.CUR"), tide=tide_)
    model.movers += movers.CatsMover(os.path.join(datadir, r"long_island_sound/tidesWAC.CUR"))
    
    for mover in model.movers:
        assert mover.active_start == model.start_time
        assert mover.active_stop == model.start_time + model.duration
        
        if isinstance( mover, movers.WindMover):
            assert  mover.wind.id in model.environment
            
        if isinstance( mover, movers.CatsMover):
            if mover.tide is not None:
                assert mover.tide.id in model.environment
        
    
    # say wind object was added to environment collection, it should not be added again
    tide_ = environment.Tide(filename=os.path.join( os.path.dirname(__file__), r"SampleData","tides","CLISShio.txt"))
    model.environment += tide_
    model.movers += movers.CatsMover(os.path.join(datadir, r"long_island_sound/tidesWAC.CUR"), tide=tide_)
    
    assert model.environment[tide_.id] == tide_
    
    # Add a mover with user defined active_start / active_stop values - these should not be updated
    active_on = model.start_time+timedelta(hours=1)
    active_off = model.start_time+timedelta(hours=4)
    custom_mover =  movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0), 
                                                    active_start=active_on,
                                                    active_stop=active_off)
    model.movers += custom_mover
    
    assert model.movers[custom_mover.id].active_start == active_on
    assert model.movers[custom_mover.id].active_stop == active_off
    

if __name__ == "__main__":
    #test_all_movers()
    test_release_at_right_time()
    
    

    
