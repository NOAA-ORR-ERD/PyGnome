from gnome import movers

from gnome import basic_types, spill
from gnome.utilities import time_utils 
from gnome import greenwich
from gnome.utilities import projections

import numpy as np

import datetime
from datetime import timedelta
import pytest


def test_exceptions():
    """
    Test ValueError exception thrown if improper input arguments
    """
    with pytest.raises(ValueError):
        movers.RandomMover(diffusion_coef=0)


class TestWindMover():
    """
    gnome.WindMover() test

    TODO: Move it to separate file
    """
    num_le = 5
    start_pos = np.zeros((num_le,3), dtype=basic_types.world_point_type)
    start_pos += (3.,6.,0.)
    rel_time = datetime.datetime(2012, 8, 20, 13)    # yyyy/month/day/hr/min/sec
    model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time) + 1)
    time_step = 15*60 # seconds

    pSpill = spill.PointReleaseSpill(num_le, start_pos, rel_time, persist=-1)

    mover = movers.RandomMover()

    def test_string_representation_matches_repr_method(self):
        assert repr(self.mover) == 'Random Mover'
        assert str(self.mover) == 'Random Mover'

    def test_id_matches_builtin_id(self):
        assert id(self.mover) == self.mover.id

    def test_get_move(self):
        """
        Test the get_move(...) results in WindMover match the expected delta
        """
        self.pSpill.prepare_for_model_step(self.model_time, self.time_step)
        self.mover.prepare_for_model_step(self.model_time, self.time_step)

        # make sure clean up is happening fine
        num_steps = 10
        delta = np.zeros((num_steps,self.pSpill.num_LEs), dtype=basic_types.world_point)
        delta = np.zeros((num_steps,self.pSpill.num_LEs), dtype=basic_types.world_point) 
        for ix in range(0,num_steps):
            curr_time = time_utils.sec_to_date(time_utils.date_to_sec(self.model_time)+(self.time_step*ix))
            print "Time step [sec]: " + str( time_utils.date_to_sec(curr_time)-time_utils.date_to_sec(self.model_time))
            delta[ix] = self.mover.get_move(self.pSpill, self.time_step, curr_time)
       
    def test_change_diffusion_coef(self):
        self.mover.diffusion_coef = 200000
        print self.mover.diffusion_coef
        assert self.mover.diffusion_coef == 200000 
       
if __name__=="__main__":
    tw = TestWindMover()
    tw.test_get_move()
    tw.test_change_diffusion_coef()