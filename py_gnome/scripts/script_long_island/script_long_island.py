#!/usr/bin/env python

"""
Script to test GNOME with long island sound data

"""

import os
import shutil
from datetime import datetime, timedelta

import numpy as np

import gnome
from gnome.environment import Wind, Tide

from gnome.utilities import map_canvas
from gnome.utilities.file_tools import haz_files

# define base directory
base_dir = os.path.dirname(__file__)

global renderer

def make_model(images_dir=os.path.join(base_dir,"images") ):
    print "initializing the model"
    
    start_time = datetime(2012, 9, 15, 12, 0)

    mapfile = os.path.join( base_dir, './LongIslandSoundMap.BNA')

    gnome_map = gnome.map.MapFromBNA(mapfile,
                           refloat_halflife=6*3600, #seconds
                           )

    ## the image output renderer
    global renderer
    renderer = gnome.renderer.Renderer(mapfile,
                                       images_dir,
                                       size=(800, 600))

    model = gnome.model.Model(start_time = start_time,
                          duration = timedelta(hours=48),
                          time_step = 3600, # one hour in seconds
                          map = gnome_map,
                          uncertain = True,
                          cache_enabled = True,
                          )
    
    print "adding outputters"
    model.outputters += renderer
    

    print "adding a spill"    
    spill = gnome.spill.SurfaceReleaseSpill(num_elements=1000,
                                            start_position = (-72.419992, 41.202120, 0.0),
                                            release_time = start_time,
                                            )
        
    model.spills += spill
        
    print  "adding a RandomMover:"
    r_mover = gnome.movers.RandomMover(diffusion_coef=500000)
    model.movers += r_mover
    
    print "adding a wind mover:"
    series = np.zeros((5,), dtype=gnome.basic_types.datetime_value_2d)
    series[0] = (start_time,                      ( 10,   45) )
    series[1] = (start_time+timedelta(hours=18),  ( 10,   90) )
    series[2] = (start_time+timedelta(hours=30),  ( 10,  135) )
    series[3] = (start_time+timedelta(hours=42),  ( 10,  180) )
    series[4] = (start_time+timedelta(hours=54),  ( 10,  225) )
    
    
    wind = Wind(timeseries=series,units='m/s')
    w_mover = gnome.movers.WindMover(wind)
    model.movers += w_mover
    
    print "adding a cats mover:"
    curr_file=os.path.join( base_dir, r"./LI_tidesWAC.CUR")
    c_mover = gnome.movers.CatsMover(curr_file, tide=Tide(os.path.join( base_dir, r"./CLISShio.txt")))
    model.movers += c_mover
    model.environment += c_mover.tide 
    
    print "viewport is:", renderer.viewport
    
    return model

def post_run(model):
    # create a place for test images (cleaning out any old ones)
    images_dir = os.path.join( base_dir, "images_2")
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)
    renderer.images_dir = images_dir

    print "re-rendering images"
    # re-render images:
    renderer.viewport = ((-72.75, 41.1),(-72.34, 41.3))

    renderer.prepare_for_model_run()

    for step_num in range(model.num_time_steps):
        print "writing image:"
        image_info = renderer.write_output(step_num)
        print "image written:", image_info

    print "viewport is:", renderer.viewport



