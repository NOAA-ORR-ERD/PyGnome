"""
unit tests cython wrapper

designed to be run with py.test
"""

import numpy as np

from gnome import basic_types
from gnome.cy_gnome import cy_random_mover
from gnome import greenwich

class Common():
    """
    test setting up and moving four particles
    
    Base class that initializes stuff that is common for multiple cy_wind_mover objects
    """
    
    #################
    # create arrays #
    #################
    num_le = 4  # test on 4 LEs
    ref  =  np.zeros((num_le,), dtype=basic_types.world_point)   # LEs - initial locations
    status = np.empty((num_le,), dtype=basic_types.status_code_type)
    
    time_step = 60
    
    def __init__(self):
        self.model_time = greenwich.gwtm('01/01/1970 11:00:00').time_seconds
        ################
        # init. arrays #
        ################
        self.ref[:] = 1.
        self.ref[:]['z'] = 0 # on surface by default
        self.status[:] = basic_types.oil_status.in_water
        
def test_move(): 
    cm = Common()
    rm = cy_random_mover.CyRandomMover(diffusion_coef=100000)
    delta = np.zeros((cm.num_le,), dtype=basic_types.world_point)
    
    rm.prepare_for_model_run()
    
    rm.prepare_for_model_step(cm.model_time, cm.time_step, False)
    rm.get_move( cm.model_time,
                 cm.time_step, 
                 cm.ref,
                 delta,
                 cm.status,
                 basic_types.spill_type.forecast,
                 0)
    
    print delta
    assert False
    
if __name__ == "__main__":
    """
    This makes it easy to use this file to debug the lib_gnome DLL 
    through Visual Studio
    """
    test_move()