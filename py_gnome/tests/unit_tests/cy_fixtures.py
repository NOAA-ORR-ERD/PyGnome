import numpy as np
from datetime import datetime
from gnome.utilities import time_utils
from gnome import basic_types

class CyTestMove(object):
    """
    Test fixture for defining the inputs to for cython get_move
    These are the components of the spill class; however, a spill class
    is not used since cython only knows about these pieces not about the 
    spill object
    
    test setting up and moving four particles
    
    Base class that initializes stuff that is common for multiple cy_wind_mover objects
    """
    spill_size = np.zeros((1,), dtype=np.int) # number of LEs in 1 uncertainty spill - simple test
    def __init__(self, num_le=4):
        self.num_le = 4  # test on 4 LEs
        self.ref  =  np.zeros((self.num_le,), dtype=basic_types.world_point)   # LEs - initial locations

        self.time_step = 60
        time = datetime(2012, 8, 20, 13)
        self.model_time = time_utils.date_to_sec( time)
        
        self.status = np.empty((self.num_le,), dtype=basic_types.status_code_type)
        self.status[:] = basic_types.oil_status.in_water
        self.spill_size[0] = self.num_le  # for uncertainty spills
        
        self.windage = np.linspace(0.01, 0.04, self.num_le)
        
        # provide arrays for storing certain and uncertain get_move deltas
        self.delta  = np.zeros((self.num_le,), dtype=basic_types.world_point)
        self.u_delta  = np.zeros((self.num_le,), dtype=basic_types.world_point) 
        