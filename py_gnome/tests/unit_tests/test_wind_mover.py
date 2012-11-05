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
    start_pos += (3.,6.,0.)
    rel_time = datetime.datetime(2012, 8, 20, 13)    # yyyy/month/day/hr/min/sec
    #model_time = basic_types.dt_to_epoch(rel_time)    # TODO: should this happen in mover or in model?
    model_time = time_utils.sec_to_date(time_utils.date_to_sec(rel_time) + 1)
    time_step = 15*60 # seconds

    pSpill = spill.PointReleaseSpill(num_le, start_pos, rel_time, persist=-1)

    time_val = np.zeros((1,), dtype=basic_types.time_value_pair)
    time_val['time'][0] = 0  # since it is just constant, just give it 0 time
    time_val['value'][0] = (0., 100.)

    wm = movers.WindMover(wind_vel=time_val)

    def test_string_representation_matches_repr_method(self):
        assert repr(self.wm) == 'Wind Mover'
        assert str(self.wm) == 'Wind Mover'

    def test_id_matches_builtin_id(self):
        assert id(self.wm) == self.wm.id

    def test_get_move(self):
        """
        Test the get_move(...) results in WindMover match the expected delta
        """
        self.pSpill.prepare_for_model_step(self.model_time, self.time_step)
        self.wm.prepare_for_model_step(self.model_time, self.time_step)

        # make sure clean up is happening fine
        delta = np.zeros((2,self.pSpill.num_LEs), dtype=basic_types.world_point)
        delta = np.zeros((2,self.pSpill.num_LEs), dtype=basic_types.world_point)
        actual =np.zeros((2,self.pSpill.num_LEs), dtype=basic_types.world_point) 
        for ix in range(0,2):
            curr_time = time_utils.sec_to_date(time_utils.date_to_sec(self.model_time)+(self.time_step*ix))
            print "Time step [sec]: " + str( time_utils.date_to_sec(curr_time)-time_utils.date_to_sec(self.model_time))
            delta[ix] = self.wm.get_move(self.pSpill, self.time_step, curr_time)
            
            actual[ix] = self._expected_move()
            
        # the results should be independent of model time
        tol = 1e-8
        print "C++ delta-lat: ";   print str(delta['lat'])
        print "Expected delta-lat: "; print str(actual['lat'])
        print "C++ delta-long: ";  print str(delta['long'])
        print "Expected delta-long: ";print str(actual['long'])
        np.testing.assert_allclose(delta['lat'], actual['lat'], tol, tol,
                                   "get_time_value is not within a tolerance of " + str(tol), 0)
        np.testing.assert_allclose(delta['long'], actual['long'], tol, tol,
                               "get_time_value is not within a tolerance of "+str(tol), 0)
       
    def _expected_move(self):
        """
        Put the expected move logic in separate private method
        """
        # expected move
        exp = np.zeros( (self.pSpill.num_LEs, 3) )
        exp[:,0] = self.pSpill['windages']*self.time_val[0]['value']['u']*self.time_step # 'u'
        exp[:,1] = self.pSpill['windages']*self.time_val[0]['value']['v']*self.time_step # 'v'

        xform = projections.FlatEarthProjection.meters_to_latlon(exp, self.pSpill['positions'])

        actual = np.zeros((len(exp),), dtype=basic_types.world_point)
        actual ['lat'] = xform[:, 1]
        actual ['long'] = xform[:, 0]
        
        return actual
        
        
       
if __name__=="__main__":
    tw = TestWindMover()
    tw.test_get_move()
