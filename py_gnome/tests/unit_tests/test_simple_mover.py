#!/usr/bin/env python

"""
test code for the simple_mover class

designed to be run with py.test
"""

import numpy as np

from gnome import spill

from gnome import simple_mover

def test_basic_move():
    sp = spill.Spill(num_LEs = 10)
        
    mover = simple_mover.simple_mover(velocity= (1.0, 10, 0.0) )

    delta = mover.get_move(sp, time_step = 100.0)
    
    assert np.alltrue(delta == (100, 1000, 0))
    
def test_north():
    sp = spill.Spill(num_LEs = 10,
                     initial_positions = (20, 0.0, 0.0),
                     )
        
    mover = simple_mover.simple_mover(velocity= (0.0, 10, 0.0) )

    delta = mover.get_move(sp, time_step = 100.0)
    
    assert np.alltrue(delta == (0.0, 1000, 0))
    
    
    

    