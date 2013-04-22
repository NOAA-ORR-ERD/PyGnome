#!/usr/bin/env python

"""
Script to test GNOME with chesapeake bay data (netCDF 3D triangle grid)
Eventually update to use Grid Map rather than BNA

"""

import os
import shutil
from datetime import datetime, timedelta

import numpy as np

import gnome
from gnome.environment import Wind

from gnome.utilities import map_canvas
from gnome.utilities.file_tools import haz_files

# define base directory
base_dir = os.path.dirname(__file__)

def make_model(images_dir=os.path.join(base_dir,"images")):
    print "initializing the model"
    
    start_time = datetime(2004, 12, 31, 13, 0)
    model = gnome.model.Model(start_time = start_time,
                              duration = timedelta(days=1),	# 1 day of data in file
                              time_step = 30 * 60, # 1/2 hr in seconds
                              uncertain = False,
                              )
    
    print "adding the map"
    
    mapfile = os.path.join( base_dir, './ChesapeakeBay.bna')
    model.map = gnome.map.MapFromBNA(mapfile,
                                     refloat_halflife=1*3600, #seconds
                                     )
    
    ## the image output map
    ## fixme: need an easier way to do this!
    output_map = map_canvas.MapCanvasFromBNA((600, 800), mapfile)
    model.output_map = output_map
    
    ## reset the viewport of the ouput map
    ## a bit kludgy, should there be a model API to do this ?
    
    ## bounding  box of viewport
    output_map.viewport = ((-76.5, 37.25),(-75.8, 37.75))
    
    
    print "adding a spill"
    
    # for now subsurface spill stays on initial layer - will need diffusion and rise velocity - wind doesn't act
    spill = gnome.spill.SurfaceReleaseSpill(num_elements=1000,
                                            start_position = (-76.126872, 37.680952, 0.0),
                                            #start_position = (-76.126872, 37.680952, 5.0),
                                            release_time = start_time,
                                            )
        
    model.spills += spill
    
    print  "adding a RandomMover:"
    r_mover = gnome.movers.RandomMover(diffusion_coef=50000)
    model.movers += r_mover
    
    
    print "adding a wind mover:"
    
    series = np.zeros((2,), dtype=gnome.basic_types.datetime_value_2d)
    series[0] = (start_time, ( 30,   0) )
    series[1] = (start_time+timedelta(hours=23), ( 30,      0) )
    
    
    wind = Wind(timeseries=series,units='knot')
    w_mover = gnome.movers.WindMover(wind)
    model.movers += w_mover
    
    print "adding a current mover:"
    
    curr_file=os.path.join( base_dir, r"./ChesapeakeBay.nc")
    topology_file=os.path.join( base_dir, r"./ChesapeakeBay.DAT")
    c_mover = gnome.movers.GridCurrentMover(curr_file,topology_file)
    model.movers += c_mover

    return model

