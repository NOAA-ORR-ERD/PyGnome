import numpy as np
from datetime import datetime
from gnome import basic_types
from gnome.cy_gnome import cy_cats_mover, cy_ossm_time, cy_shio_time
from gnome.utilities import time_utils
import cy_fixtures


file = r"SampleData/long_island_sound/CLISShio.txt"
shio = cy_shio_time.CyShioTime(file)
top_file=r"SampleData/long_island_sound/tidesWAC.CUR"


class TestCats():
    """
    Contains one test method to do one forecast move and one uncertainty move
    and verify that they are different
    
    Primarily just checks that CyCatsMover can be initialized correctly and all methods are invoked
    """
    cats = cy_cats_mover.CyCatsMover()
    cats.set_shio(shio)
    cats.prepare_for_model_run()    
    cats.read_topology(top_file)

    cm = cy_fixtures.CyTestMove()
    cm.ref[:] = (-72.5, 41.17, 0)
    
    def certain_move(self,delta):
        self.cats.prepare_for_model_step(self.cm.model_time, self.cm.time_step)
        self.cats.get_move(self.cm.model_time,
                           self.cm.time_step,
                           self.cm.ref, delta,
                           self.cm.status,basic_types.spill_type.forecast,
                           0)
        
    def uncertain_move(self, delta):
        self.cats.prepare_for_model_step(self.cm.model_time, self.cm.time_step,1,self.cm.spill_size)
        self.cats.get_move(self.cm.model_time,
                           self.cm.time_step,
                           self.cm.ref, delta,
                           self.cm.status,basic_types.spill_type.uncertainty,
                           0)
    
    def test_move(self):
        """
        call get_move for forcast and uncertainty spill and makes sure
        (a) there is a move, so the deltas are not all zero
        (b) the uncertain and forcast moves are different
        """
        self.certain_move(self.cm.delta)
        print 
        print self.cm.delta
        assert np.all(self.cm.delta != 0)
        
        self.uncertain_move(self.cm.u_delta)
        assert np.all( self.cm.u_delta != 0)
        
        assert np.all(self.cm.delta != self.cm.u_delta)
    
    def test_step_is_done(self):
        """
        call model_step_is_done() just to make sure it doesn't crash
        """
        self.cats.model_step_is_done()
        assert True