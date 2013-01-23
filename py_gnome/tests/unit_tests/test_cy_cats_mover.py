import numpy as np
from gnome import basic_types
from gnome.cy_gnome import cy_cats_mover, cy_ossm_time, cy_shio_time
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
        """
        test get_move for forecast LEs
        """
        self.cats.prepare_for_model_step(self.cm.model_time, self.cm.time_step)
        self.cats.get_move(self.cm.model_time,
                           self.cm.time_step,
                           self.cm.ref, delta,
                           self.cm.status,basic_types.spill_type.forecast,
                           0)
        self.cats.model_step_is_done()
        assert np.all(delta != 0)
        
    def uncertain_move(self, delta):
        """
        test get_move for uncertainty LEs
        """
        self.cats.prepare_for_model_step(self.cm.model_time, self.cm.time_step,1,self.cm.spill_size)
        self.cats.get_move(self.cm.model_time,
                           self.cm.time_step,
                           self.cm.ref, delta,
                           self.cm.status,basic_types.spill_type.uncertainty,
                           0)
        self.cats.model_step_is_done()
        assert np.all(delta != 0)
        
    # 
    # TODO: Figure out why tests fail if following tests are uncommented
    #
    #===========================================================================
    # def test_certain_move(self):
    #    delta1  = np.zeros((self.cm.num_le,), dtype=basic_types.world_point)
    #    self.certain_move(delta1)
    #    assert np.all(delta1 != 0)
    # 
    # def test_uncertain_move(self):
    #    delta2  = np.zeros((self.cm.num_le,), dtype=basic_types.world_point)
    #    self.uncertain_move(delta2)
    #    assert np.all( delta2 != 0)
    #===========================================================================
    
    def test_move(self):
        """
        call get_move for forcast and uncertainty spill and makes sure
        (a) there is a move, so the deltas are not all zero
        (b) the uncertain and forcast moves are different
        """
        self.certain_move(self.cm.delta)
        print 
        print self.cm.delta
        
        self.uncertain_move(self.cm.u_delta)
        print self.cm.u_delta
        assert np.all(self.cm.delta != self.cm.u_delta)
    