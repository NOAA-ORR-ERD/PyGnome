from gnome import movers

from gnome import basic_types, spill
from gnome.utilities import time_utils 
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
   num_le = 5
   start_pos = np.zeros((num_le,3), dtype=basic_types.world_point_type)
   start_pos += (3.,3.,0.)
   rel_time = datetime.datetime(2012, 8, 20, 13)    # yyyy/month/day/hr/min/sec
   #model_time = basic_types.dt_to_epoch(rel_time)    # TODO: should this happen in mover or in model?
   model_time_sec = time_utils.date_to_sec(rel_time) + 1
   model_time = time_utils.sec_to_date(model_time_sec)
   time_step = 15*60 # seconds
   
   pSpill = spill.PointReleaseSpill(num_le, start_pos, rel_time, persist=-1)
   
   time_val = np.zeros((1,), dtype=basic_types.time_value_pair)
   time_val['time'][0] = 0  # since it is just constant, just give it 0 time
   time_val['value'][0] = (0., 100.)
   
   wm = movers.WindMover(wind_vel=time_val)
   
   def test_get_move(self):
       """
       """
       self.pSpill.prepare_for_model_step(self.model_time, self.time_step, self.pSpill.is_uncertain)
       self.wm.prepare_for_model_step(self.model_time_sec, self.time_step, self.pSpill.is_uncertain)
       
       delta = self.wm.get_move(self.pSpill, self.time_step, self.model_time_sec)
       print delta
       
       
if __name__=="__main__":
    tw = TestWindMover()
    tw.test_get_move()
    
   
   
