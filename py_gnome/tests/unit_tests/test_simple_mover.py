#!/usr/bin/env python

"""
test code for the simple_mover class

designed to be run with py.test
"""

import numpy as np

from gnome import spill

from gnome import simple_mover

from gnome.utilities.projections import FlatEarthProjection as proj

def test_basic_move():
    sp = spill.Spill(num_LEs = 5) #initilizes to long, lat, z = 0.0, 0.0, 0.0
        
    mover = simple_mover.SimpleMover(velocity= (1.0, 10.0, 0.0) )

    delta = mover.get_move(sp, time_step = 100.0, model_time=None)

    expected = np.zeros_like(delta)
    expected = proj.meters_to_latlon((100.0, 1000.0, 0.0), (0.0, 0.0, 0.0))
    assert np.alltrue(delta == expected)
    
def test_north():
    sp = spill.Spill(num_LEs = 10,
                     initial_positions = (20, 0.0, 0.0),
                     )
        
    mover = simple_mover.SimpleMover(velocity= (0.0, 10, 0.0) )

    delta = mover.get_move(sp, time_step = 100.0, model_time=None)
    
    expected = np.zeros_like(delta)
    expected = proj.meters_to_latlon((0.0, 1000.0, 0.0), (0.0, 0.0, 0.0))
    assert np.alltrue(delta == expected)
    
    
    

    
