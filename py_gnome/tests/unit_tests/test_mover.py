'''
Test some of the base class functionality independent of derived clases.
Just simpler to do the testing here

'''

from datetime import datetime,timedelta
from gnome import movers
from gnome.spill_container import TestSpillContainer

import pytest 

def test_exceptions():
    with pytest.raises(ValueError):
        now = datetime.now()
        m = movers.Mover(active_start=now, active_stop=now)
        

def test_properties():
    """
    Test default props
    """
    m = movers.Mover()
    assert m.on == True
    
    m.on = False
    assert m.on == False

def test_active():
    """
    tests that is active is toggled correctly based on timespan
    """
    time_step = 15 * 60 # seconds
    model_time = datetime(2012, 8, 20, 13)
    sc = TestSpillContainer(1, (0,0,0))   # no used for anything
    
    mv = movers.Mover()
    mv.prepare_for_model_step( sc, time_step, model_time)
    assert mv.active == True    # model_time = active_start
    
    mv = movers.Mover(active_start=model_time )
    mv.prepare_for_model_step( sc, time_step, model_time)
    assert mv.active == True    # model_time = active_start
    
    mv = movers.Mover(active_start=model_time+timedelta(seconds=time_step) )
    mv.prepare_for_model_step( sc, time_step, model_time)
    assert mv.active == False    # model_time + time_step = active_start
    
    mv.active_start = model_time - timedelta(seconds=time_step/2)
    mv.prepare_for_model_step( sc, time_step, model_time)
    assert mv.active == True    # model_time - time_step/2 = active_start
    
    # No need to test get_move again, above tests it is working per active flag
    # Next test just some more borderline cases that active is being set correctly
    mv.active_stop = model_time + timedelta(seconds=1.5*time_step)
    mv.prepare_for_model_step( sc, time_step, model_time)
    assert mv.active == True
    
    mv.active_stop = model_time + timedelta(seconds=2*time_step)
    mv.prepare_for_model_step( sc, time_step, model_time+2*timedelta(seconds=time_step))
    assert mv.active == False
