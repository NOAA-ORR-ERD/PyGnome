"""
Tests the SpillContainer class
"""

from datetime import datetime, timedelta

import pytest

import numpy as np

from gnome import basic_types
from gnome.spill_container import SpillContainer
from gnome.spill import Spill, SurfaceReleaseSpill

def test_simple_init():
    sc = SpillContainer()


## real tesing involves adding spills!
def test_one_simple_spill():
    start_time = datetime(2012, 1, 1, 12)
    start_position = (23.0, -78.5, 0.0)
    num_LEs = 100
    sc = SpillContainer()
    spill = SurfaceReleaseSpill(num_LEs,
                                start_position,
                                start_time)
    sc.add_spill(spill)
    sc.prepare_for_model_step(start_time)

    assert sc['positions'].shape == (num_LEs, 3)
    assert sc['last_water_positions'].shape == (num_LEs, 3)

    sc.prepare_for_model_step(start_time + timedelta(hours=24) )

    assert sc['positions'].shape == (num_LEs, 3)
    assert sc['last_water_positions'].shape == (num_LEs, 3)

    assert np.array_equal( sc['positions'][0], start_position )

## multiple spills with different release times:
def test_multiple_spills():
    start_time = datetime(2012, 1, 1, 12)
    start_time2 = datetime(2012, 1, 2, 12)
    start_position = (23.0, -78.5, 0.0)
    num_LEs = 100
    sc = SpillContainer()
    spill = SurfaceReleaseSpill(num_LEs,
                                start_position,
                                start_time)

    spill2 = SurfaceReleaseSpill(num_LEs,
                                start_position,
                                start_time2)

    sc.add_spill(spill)
    sc.add_spill(spill2)
    sc.prepare_for_model_step(start_time)

    assert sc['positions'].shape == (num_LEs, 3)
    assert sc['last_water_positions'].shape == (num_LEs, 3)

    sc.prepare_for_model_step(start_time + timedelta(hours=24) )

    assert sc['positions'].shape == (num_LEs*2, 3)
    assert sc['last_water_positions'].shape == (num_LEs*2, 3)


def test_reset():
    start_time = datetime(2012, 1, 1, 12)
    start_time2 = datetime(2012, 1, 2, 12)
    start_position = (23.0, -78.5, 0.0)
    num_LEs = 100
    sc = SpillContainer()
    spill = SurfaceReleaseSpill(num_LEs,
                                start_position,
                                start_time)

    spill2 = SurfaceReleaseSpill(num_LEs,
                                start_position,
                                start_time2)

    sc.add_spill(spill)
    sc.add_spill(spill2)
    sc.prepare_for_model_step(start_time)
    sc.prepare_for_model_step(start_time + timedelta(hours=24) )

    sc.reset()
    assert spill.num_released == 0
    assert spill2.num_released == 0

def test_reset2():
    """
    test that extra arrays are removed on a reset

    # not much of a test, really -- add more?
    """
    start_time = datetime(2012, 1, 1, 12)
    start_time2 = datetime(2012, 1, 2, 12)
    start_position = (23.0, -78.5, 0.0)
    num_LEs = 100
    sc = SpillContainer()
    spill = Spill(num_LEs)

    spill2 = SurfaceReleaseSpill(num_LEs,
                                start_position,
                                start_time2)

    sc.add_spill(spill)
    sc.add_spill(spill2)

    sc.prepare_for_model_step(start_time)
    sc.prepare_for_model_step(start_time + timedelta(hours=24) )

    sc.remove_spill(spill)

    sc.reset()
    print "id of spill 2", spill2.id
    assert spill2.num_released == 0





# def test_data_access():
#     sp = spill.Spill(num_LEs = 10)
    
#     sp['positions'] += (3.0, 3.0, 3.0)

#     assert np.array_equal(sp['positions'],
#                           np.ones( (10, 3), dtype=basic_types.world_point_type ) * 3.0
#                           )
    
    
# def test_data_setting():
#     sp = spill.Spill(num_LEs = 10)
    
#     new_pos = np.ones( (10, 3), dtype=basic_types.world_point_type ) * 3.0
    
#     sp['positions'] = new_pos

#     assert np.array_equal(sp['positions'],
#                           new_pos
#                           )
    
# def test_data_setting_error1():
#     """
#     Should get an error when trying to set the data to a different size array
#     """
#     sp = spill.Spill(num_LEs = 10)
    
#     new_pos = np.ones( (12, 3), dtype=basic_types.world_point_type ) * 3.0
#     with pytest.raises(ValueError):
#         sp['positions'] = new_pos


# def test_data_setting_error2():
#     """
#     Should get an error when trying to set the data to a different type array
#     """
#     sp = spill.Spill(num_LEs = 10)
    
#     new_pos = np.ones( (10, 3), dtype=np.int32 )
    
#     with pytest.raises(ValueError):
#         sp['positions'] = new_pos

# def test_data_setting_error3():
#     """
#     Should get an error when trying to set the data to a different shape array
#     """
#     sp = spill.Spill(num_LEs = 10)
    
#     new_pos = np.ones( (10, 4), dtype=basic_types.world_point_type ) * 3.0
    
#     with pytest.raises(ValueError):
#         sp['positions'] = new_pos

# def test_data_setting_new():
#     """
#     Should be able to add a new data array
#     """
#     sp = spill.Spill(num_LEs = 10)
    
#     new_arr = np.ones( (10, 3), dtype=np.float64 )
    
#     sp['new_name'] = new_arr

#     assert sp['new_name'] is new_arr 

# def test_data_setting_new_list():
#     """
#     Should be able to add a new data that's not a numpy array
#     """
#     sp = spill.Spill(num_LEs = 10)
    
#     new_arr = range(10)
    
#     sp['new_name'] = new_arr

#     assert np.array_equal(sp['new_name'],  new_arr) 
    
# def test_data_setting_new_error():
#     """
#     Should get an error adding a new data array of the wrong size
#     """
#     sp = spill.Spill(num_LEs = 10)
    
#     new_arr = np.ones( (12, 3), dtype=np.float64 )
    
#     with pytest.raises(ValueError):
#         sp['new_name'] = new_arr

# ## PointReleaseTests
    
# def test_point_init():
#     sp = spill.PointReleaseSpill(num_LEs = 10,
#                                  start_position = (28.5, -128.3, 0),
#                                  release_time=datetime.datetime(2012, 8, 20, 13),
#                                  )
    
#     assert sp['status_codes'].shape == (10,)
#     assert sp['positions'].shape == (10,3)
#     assert np.alltrue( sp['status_codes'] == basic_types.oil_status.not_released)

# def test_point_release():
#     rel_time = datetime.datetime(2012, 8, 20, 13)
#     sp = spill.PointReleaseSpill(num_LEs = 10,
#                                  start_position = (28.5, -128.3, 0),
#                                  release_time=rel_time,
#                                  )
#     time_step = 15*60 # seconds
#     assert np.alltrue( sp['status_codes'] == basic_types.oil_status.not_released)

#     sp.prepare_for_model_step(rel_time - datetime.timedelta(seconds=1), time_step)# one second before release time
#     assert np.alltrue( sp['status_codes'] == basic_types.oil_status.not_released )
    
#     sp.prepare_for_model_step(rel_time, time_step)
#     assert np.alltrue( sp['status_codes'] == basic_types.oil_status.in_water )


# def test_point_reset():
#     rel_time = datetime.datetime(2012, 8, 20, 13)
#     time_step = 15*60 # seconds
#     sp = spill.PointReleaseSpill(num_LEs = 10,
#                                  start_position = (28.5, -128.3, 0),
#                                  release_time=rel_time,
#                                  )
    
#     sp.prepare_for_model_step(rel_time, time_step)
#     assert np.alltrue( sp['status_codes'] == basic_types.oil_status.in_water )

#     sp.reset()
#     assert np.alltrue( sp['status_codes'] == basic_types.oil_status.not_released)

# def test_multi_points():

#     start_position = [ (28.5, -128.3, 0),
#                        (28.6, -128.4, 0),
#                        (28.7, -128.5, 0)]

#     rel_time = datetime.datetime(2012, 8, 20, 13)
#     time_step = 15*60 # seconds
#     sp = spill.PointReleaseSpill(num_LEs = len(start_position),
#                                  start_position = start_position,
#                                  release_time=rel_time,
#                                  )
    
#     assert np.alltrue( sp['status_codes'] == basic_types.oil_status.not_released)
#     assert np.array_equal(start_position, sp['positions'])

#     sp.prepare_for_model_step(rel_time, time_step)
#     assert np.alltrue( sp['status_codes'] == basic_types.oil_status.in_water )
#     assert np.array_equal(start_position, sp['positions'])

#     sp.reset()
#     assert np.alltrue( sp['status_codes'] == basic_types.oil_status.not_released)
#     assert np.array_equal(start_position, sp['positions'])

# def test_stay_moved():

#     start_position = np.array( [ (28.5, -128.3, 0),
#                                  (28.6, -128.4, 0),
#                                  (28.7, -128.5, 0),
#                                 ]
#                               )

#     rel_time = datetime.datetime(2012, 8, 20, 13)
#     time_step = 15*60 # seconds
#     sp = spill.PointReleaseSpill(num_LEs = len(start_position),
#                                  start_position = start_position,
#                                  release_time=rel_time,
#                                  )
    
#     assert np.alltrue( sp['status_codes'] == basic_types.oil_status.not_released)
#     assert np.array_equal(start_position, sp['positions'])

#     sp.prepare_for_model_step(rel_time, time_step)
#     assert np.alltrue( sp['status_codes'] == basic_types.oil_status.in_water )
#     assert np.array_equal(start_position, sp['positions'])

#     # now move them:
#     sp['positions'] += 0.2
#     assert np.array_equal(start_position + 0.2, sp['positions'])

#     # make sure they stay moved:
#     sp.prepare_for_model_step(rel_time + datetime.timedelta(seconds=time_step), time_step)
#     assert np.array_equal(start_position + 0.2, sp['positions'])

    


