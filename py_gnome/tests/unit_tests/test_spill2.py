"""
Tests the new spill code.
"""
# import datetime

# import pytest

# import numpy as np

from gnome import basic_types
from gnome.sources import Spill, FloatingSpill, SurfaceReleaseSpill


def test_init_Spill():
    """
    the base class does not do much
    """
    sp = Spill()

    assert  sp.id == 1


def test_set_id():
    """
    ids should get set, and stay unique as you delete and create spills
    """

    spills = [Spill() for i in range(10)]

    # the ids are unique
    assert len( set([spill.id for spill in spills]) ) == len(spills)

    #delete and create a few:
    del spills[3]
    del spills[5]

    spills.extend(  [FloatingSpill() for i in range(5)] ) 

    # ids still unique
    assert len( set([spill.id for spill in spills]) ) == len(spills)

    del spills[10]
    del spills[4]

    spills.extend(  [SurfaceReleaseSpill() for i in range(5)] ) 

    # ids still unique
    assert len( set([spill.id for spill in spills]) ) == len(spills)

    #print [spill.id for spill in spills]
    #assert False

def test_new_elements():
    """
    see if creating new elements works
    """
    sp = Spill()
    




#     assert sp['status_codes'].shape == (10,)
#     assert sp['positions'].shape == (10,3)
#     assert np.alltrue( sp['status_codes'] == basic_types.oil_status.in_water )


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

if __name__ == "__main__":
    test_init_simple()


