#!/usr/bin/env python

"""
Script to test GNOME with long island sound data

Updated to use new version of code:
11/5/2012

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

def make_model():
    print "initializing the model"
    
    start_time = datetime(2012, 9, 15, 12, 0)
    model = gnome.model.Model(start_time = start_time,
                              duration = timedelta(days=2),
                              time_step = 3600, # one hour in seconds
                              uncertain = True
                              )
    
    print "adding the map"
    
    mapfile = os.path.join( base_dir, './LongIslandSoundMap.BNA')
    model.map = gnome.map.MapFromBNA(mapfile,
                                     refloat_halflife=6*3600, #seconds
                                     )
    
    ## the image output map
    map_ = map_canvas.MapCanvasFromBNA((800, 600), mapfile)
    model.output_map = map_
    
    
    print "adding a spill"
    
    spill = gnome.spill.SurfaceReleaseSpill(num_elements=1000,
                                            start_position = (-72.419992, 41.202120, 0.0),
                                            release_time = start_time,
                                            )
        
    model.spills += spill
    
    # second spill
    # spill = gnome.spill.PointReleaseSpill(num_LEs=10,
    #                                       start_position = (-72.419992,41.202120),
    #                                       release_time = start_time,
    #                                       )
        
    # model.add_spill(spill)
    
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
    
    return model

