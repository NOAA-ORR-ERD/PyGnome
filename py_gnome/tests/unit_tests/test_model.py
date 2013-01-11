#!/usr/bin/env python

"""
test code for the model class

not a lot to test by itself, but a start

"""
import os, shutil
from datetime import datetime, timedelta
import numpy as np
import gnome.model
import gnome.map
from gnome import movers, weather
import gnome.spill
from gnome.spill import SpatialReleaseSpill

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

def test_simple_run():
    """
    pretty much all this tests is that the model will run
    """
    
    start_time = datetime(2012, 9, 15, 12, 0)
    
    model = gnome.model.Model()
    
    model.map = gnome.map.GnomeMap()
    a_mover = movers.simple_mover.SimpleMover(velocity=(1.0, 2.0, 0.0))
    
    model.add_mover(a_mover)

    spill = gnome.spill.SurfaceReleaseSpill(num_elements=10,
                                            start_position = (0.0, 0.0, 0.0),
                                            release_time = start_time,
                                            )
    
    model.add_spill(spill)
    model.start_time = spill.release_time
    
    # test iterator:
    for step in model:
        print "just ran time step: %s"%model.current_time_step

    # rewind and run again:
    print "rewinding"
    model.rewind()
    # test iterator:
    for step in model:
        print "just ran time step: %s"%model.current_time_step
        
    assert True
    
def test_simple_run_with_map():
    """
    pretty much all this tests is that the model will run
    """
    
    start_time = datetime(2012, 9, 15, 12, 0)
    
    model = gnome.model.Model()
    
    model.map = gnome.map.MapFromBNA( 'SampleData/MapBounds_Island.bna',
                                refloat_halflife=6*3600, #seconds
                                )
    a_mover = movers.simple_mover.SimpleMover(velocity=(1.0, 2.0, 0.0))
    
    model.add_mover(a_mover)

    spill = gnome.spill.SurfaceReleaseSpill(num_elements=10,
                                            start_position = (0.0, 0.0, 0.0),
                                            release_time = start_time,
                                            )
    
    model.add_spill(spill)
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

    mapfile = "SampleData/MapBounds_Island.bna"

    # the land-water map
    model.map = gnome.map.MapFromBNA( mapfile,
                                      refloat_halflife=6*3600, #seconds
                                     )
    # the image output map
    map = gnome.utilities.map_canvas.MapCanvas((400, 300))
    polygons = haz_files.ReadBNA(mapfile, "PolygonSet")
    map.set_land(polygons)
    model.output_map = map
    
    a_mover = movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))
    model.add_mover(a_mover)

    N = 10 # a line of ten points
    start_points = np.zeros((N, 3) , dtype=np.float64)
    start_points[:,0] = np.linspace(-127.1, -126.5, N)
    start_points[:,1] = np.linspace( 47.93, 48.1, N)
    #print start_points
    spill = gnome.spill.SpatialReleaseSpill(start_positions = start_points,
                                            release_time = start_time,
                                            )
    
    model.add_spill(spill)
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

    mapfile = "SampleData/MapBounds_Island.bna"

    # the land-water map
    model.map = gnome.map.MapFromBNA( mapfile,
                                      refloat_halflife=6*3600, #seconds
                                     )
    # the image output map
    map = gnome.utilities.map_canvas.MapCanvas((400, 300))
    polygons = haz_files.ReadBNA(mapfile, "PolygonSet")
    map.set_land(polygons)
    model.output_map = map
    
    a_mover = movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))
    model.add_mover(a_mover)

    N = 10 # a line of ten points
    start_points = np.zeros((N, 3) , dtype=np.float64)
    start_points[:,0] = np.linspace(-127.1, -126.5, N)
    start_points[:,1] = np.linspace( 47.93, 48.1, N)
    #print start_points
    spill = gnome.spill.SpatialReleaseSpill(start_positions = start_points,
                                            release_time = start_time,
                                            )
    
    model.add_spill(spill)
    model.start_time = spill.release_time
    #image_info = model.next_image()

    model.is_uncertain = True

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

    id_1 = model.add_mover(mover_1)
    id_2 = model.add_mover(mover_2)

    assert(model.get_mover(id_1) == mover_1)
    assert(model.get_mover(id_2) == mover_2)
    assert(model.get_mover('Invalid') is None)

    model.remove_mover(id_1)
    model.remove_mover(id_2)
    assert(model.remove_mover('Invalid') is None)

    assert(model.get_mover(id_1) is None)
    assert(model.get_mover(id_2) is None)

    id_1 = model.add_mover(mover_1)
    id_2 = model.add_mover(mover_2)

    # not there, and how should it work if it was?
    #model.replace_mover(id_1, mover_2)
    #model.replace_mover(id_2, mover_2)


def test_all_movers():
    """
    a test that tests that all the movers at least can be run

    add new ones as they come along!
    """

    start_time = datetime(2012, 1, 1, 0, 0)
    
    model = gnome.model.Model()
    model.duration = timedelta(hours=12)
    model.time_step = timedelta(hours = 1)
    model.start_time = start_time

    # a spill
    model.add_spill(gnome.spill.SurfaceReleaseSpill(num_elements=10,
                                                    start_position = (0.0, 0.0, 0.0),
                                                    release_time = start_time,
                                                    ) )

    # the land-water map
    model.map = gnome.map.GnomeMap() # the simpleset of maps
    
    # simplemover
    model.add_mover( movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0)) )

    # random mover
    model.add_mover( gnome.movers.RandomMover(diffusion_coef=100000) )

    # wind mover
    series = np.array( (start_time, ( 10,   45) ),  dtype=gnome.basic_types.datetime_value_2d).reshape((1,))
    model.add_mover( gnome.movers.WindMover(weather.Wind(timeseries=series, units='meter per second')) )
  
    
    # run the model all the way...
    num_steps_output = 0
    for step in model:
        num_steps_output += 1
        print "running step:", step

    assert num_steps_output == (model.duration.total_seconds() / model.time_step) + 1 # there is the zeroth step, too.
    
if __name__ == "__main__":
    test_all_movers()
    
    
    

    