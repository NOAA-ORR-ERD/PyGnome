from gnome import movers

from gnome import basic_types, spill
from gnome import greenwich


import numpy as np

import datetime
import pytest

def test_exceptions():
    """
    Test ValueError exception thrown if improper input arguments
    """
    with pytest.raises(ValueError):
        movers.WindMover()
        wind_vel = np.zeros((1,), basic_types.velocity_rec)
        movers.WindMover(wind_vel=wind_vel)

class TestWindMover():
   """
   gnome.WindMover() test
   
   TODO: Move it to separate file
   """
   num_le = 10
   start_pos = np.zeros((num_le,3), dtype=basic_types.world_point_type)
   start_pos += (3.,3.,0.)
   rel_time = datetime.datetime(2012, 8, 20, 13)    # yyyy/month/day/hr/min/sec
   time_step = 15*60 # seconds
   
   pSpill = spill.PointReleaseSpill(num_le, start_pos, rel_time)
   
   time_val = np.zeros((1,), dtype=basic_types.time_value_pair)
   time_val['time'][0] = 0  # since it is just constant, just give it 0 time
   time_val['velocity_rec'][0] = (0., 100.)
   
   wm = movers.WindMover(wind_vel=time_val)
   
   #def test_prepare_for_model_step(self):
       
       
   
   
