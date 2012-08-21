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

    mover.get_move(sp, time_step = 100.0)
    
    assert np.alltrue(sp['next_positions'] == (100, 1000, 0))
    
    
    

    