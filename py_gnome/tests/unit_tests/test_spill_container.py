#!/usr/bin/env python

"""
Tests the SpillContainer class
"""

from datetime import datetime, timedelta
import copy

import numpy as np
import pytest

from gnome.basic_types import oil_status, id_type, world_point_type
from gnome.spill_container import SpillContainer, SpillContainerPair
from gnome.spill import PointLineSource

from conftest import sample_sc_release

# only required to setup data arrays correctly
from gnome import array_types


# Sample data for creating spill
windage_at = dict(array_types.WindMover)
num_elements = 100
start_position = (23.0, -78.5, 0.0)
release_time = datetime(2012, 1, 1, 12)
end_position = (24.0, -79.5, 1.0)
end_release_time = datetime(2012, 1, 1, 12) + timedelta(hours=4)


def test_simple_init():
    sc = SpillContainer()
    assert sc


### Helper functions ###
def assert_dataarray_shapes(sc):
    for key, val in sc.array_types.iteritems():
        assert sc[key].shape == (sc.num_released,) + val.shape

    assert np.all(sc['status_codes']
                      == sc.array_types['status_codes'].initial_value)
    assert np.all(sc['id'] == range(0, sc.num_released))


def assert_sc_single_spill(sc):
    """ standard asserts for a SpillContainer with a single spill """
    assert_dataarray_shapes(sc)
    assert np.all(sc['spill_num'] == sc['spill_num'][0])

    # only one spill in SpillContainer
    for spill in sc.spills:
        assert np.array_equal(sc['positions'][0], spill.start_position)
        assert np.array_equal(sc['positions'][-1], spill.end_position)


def test_test_spill_container():
    sc = sample_sc_release()
    assert_sc_single_spill(sc)


## real testing involves adding spills!

@pytest.mark.parametrize("spill",
                         [PointLineSource(num_elements,
                                          start_position,
                                          release_time),
                          PointLineSource(num_elements,
                                          start_position,
                                          release_time,
                                          end_position,
                                          end_release_time)])
def test_one_simple_spill(spill):
    """ checks data_arrays correctly populated for a single spill in
    SpillContainer """
    sc = SpillContainer()
    sc.spills.add(spill)
    time_step = 3600

    sc.prepare_for_model_run(windage_at)
    num_steps = ((spill.end_release_time -
                  spill.release_time).seconds / time_step + 1)

    for step in range(num_steps):
        current_time = spill.release_time + timedelta(seconds=time_step * step)
        sc.prepare_for_model_step(current_time)
        sc.release_elements(current_time, time_step)
        assert sc.current_time_stamp == current_time

    assert sc.num_released == spill.num_elements

    assert_sc_single_spill(sc)


@pytest.mark.parametrize("uncertain", [False, True])
def test_multiple_spills(uncertain):
    """
    SpillContainer initializes correct number of elements in data_arrays.
    Use Multiple spills with different release times.
    Also, deleting a spill shouldn't change data_arrays for particles
    already released.
    """
    sc = SpillContainer(uncertain)
    spills = [PointLineSource(num_elements, start_position, release_time),
              PointLineSource(num_elements, start_position,
                    release_time + timedelta(hours=1),
                    end_position, end_release_time)]

    sc.spills.add(spills)

    for spill in spills:
        assert sc.spills[spill.id] == spill

    assert sc.uncertain == uncertain

    time_step = 3600
    num_steps = ((spills[-1].end_release_time -
                  spills[-1].release_time).seconds / time_step + 1)

    sc.prepare_for_model_run(windage_at)

    for step in range(num_steps):
        current_time = release_time + timedelta(seconds=time_step * step)
        sc.prepare_for_model_step(current_time)
        sc.release_elements(current_time, time_step)
        assert sc.current_time_stamp == current_time

    assert sc.num_released == spills[0].num_elements * len(spills)
    assert_dataarray_shapes(sc)

    sc.spills.remove(spills[0].id)

    with pytest.raises(KeyError):
        # it shouldn't be there anymore.
        assert sc.spills[spills[0].id] is None

    # however, the data arrays of released particles should be unchanged
    assert sc.num_released == spill.num_elements * len(spills)
    assert_dataarray_shapes(sc)


def test_rewind():
    """
    Test rewinding spill containter rewinds the spills.
    - SpillContainer should reset its the data_arrays to empty and num_released
      to 0
    - it should reset num_released = 0 for all spills and reset
      start_time_invalid flag to True. Basically all spills are rewound
    """
    num_elements = 100
    release_time = datetime(2012, 1, 1, 12)
    release_time2 = release_time + timedelta(hours=24)
    start_position = (23.0, -78.5, 0.0)
    sc = SpillContainer()

    spills = [PointLineSource(num_elements, start_position, release_time),
              PointLineSource(num_elements, start_position, release_time2)]
    sc.spills.add(spills)

    sc.prepare_for_model_run(dict(array_types.WindMover))

    for time in [release_time, release_time2]:
        sc.prepare_for_model_step(time)
        sc.release_elements(time, 3600)

    assert sc.num_released == num_elements * len(spills)
    for spill in spills:
        assert spill.num_released == spill.num_elements

    sc.rewind()
    assert sc.num_released == 0
    assert_dataarray_shapes(sc)
    for spill in spills:
        assert spill.num_released == 0
        assert spill.start_time_invalid


def test_data_access():
    sp = sample_sc_release()

    sp['positions'] += (3.0, 3.0, 3.0)

    assert np.array_equal(sp['positions'], np.ones((sp.num_released, 3),
                          dtype=world_point_type) * 3.0)


def test_set_data_array():
    """
    add data to a data array in the spill container
    """
    sc = sample_sc_release()

    sc['spill_num'] = np.ones(len(sc['spill_num']), dtype=id_type) * 4
    assert np.all(sc['spill_num'] == 4)

    new_pos = np.ones((sc.num_released, 3), dtype=world_point_type) * 3.0
    sc['positions'] = new_pos

    assert np.array_equal(sc['positions'], new_pos)


def test_data_setting_wrong_size_error():
    """
    Should get an error when trying to set the data to a different size array
    """

    sc = sample_sc_release()

    new_pos = np.ones((sc.num_released + 2, 3), dtype=world_point_type) * 3.0

    with pytest.raises(ValueError):
        sc['positions'] = new_pos


def test_data_setting_wrong_dtype_error():
    """
    Should get an error when trying to set the data to a different type array
    """

    sc = sample_sc_release()

    new_pos = np.ones((sc.num_released, 3), dtype=np.int32)

    with pytest.raises(ValueError):
        sc['positions'] = new_pos


def test_data_setting_wrong_shape_error():
    """
    Should get an error when trying to set the data to a different shape array
    """

    sc = sample_sc_release()

    new_pos = np.ones((sc.num_released, 4), dtype=world_point_type) * 3.0

    with pytest.raises(ValueError):
        sc['positions'] = new_pos


@pytest.mark.xfail
def test_addto_array_types():
    """
    Can add a new ArrayType to all_array_types; however, must rewind model
    to get the 'new_name' array in data_arrays
    todo: revisit this!
    """

    sc = sample_sc_release()
    sc.array_types['new_name'] = array_types.ArrayType((3,), np.float64, 0)

    # MUST rewind and release elements again to get new_name in data_arrays

    sc.rewind()
    sc.release_elements(sc.current_time_stamp, time_step=100)

    new_arr = np.ones((10, 3), dtype=np.float64)
    sc['new_name'] = new_arr

    assert 'new_name' in sc.data_arrays_dict
    assert sc['new_name'] is new_arr
    assert False


def test_data_setting_new():
    """
    Can add a new item to data_arrays. This will automatically update
    SpillContainer's array_types dict

    No rewind necessary. Subsequent releases will initialize the newly added
    numpy array for newly released particles
    """
    spill = PointLineSource(20, start_position, release_time,
                            end_release_time=end_release_time)
    # release 10 particles
    time_step = (end_release_time - release_time) / 2
    sc = sample_sc_release(time_step=time_step.seconds, spill=spill)

    new_arr = np.ones((sc.num_released, 3), dtype=np.float64)
    sc['new_name'] = new_arr

    assert 'new_name' in sc.data_arrays_dict
    assert 'new_name' in sc.array_types
    assert sc['new_name'] is new_arr
    assert_dataarray_shapes(sc)

    # now release remaining particles and check to see new_name is populated
    # with zeros - default initial_value
    sc.prepare_for_model_step(spill.release_time + time_step)
    released = sc.num_released

    sc.release_elements(spill.release_time + time_step, time_step.seconds)
    new_released = sc.num_released - released

    assert_dataarray_shapes(sc)     # check shape is consistent for all arrays
    assert sc.num_released == spill.num_elements     # release all elements
    assert np.all(sc['new_name'][-new_released:] ==  # initialized to 0!
                  (0.0, 0.0, 0.0))


def test_data_setting_new_list():
    """
    Should be able to add a new data that's not a numpy array
    """

    sp = sample_sc_release()

    new_arr = range(sp.num_released)

    sp['new_name'] = new_arr

    assert np.array_equal(sp['new_name'], new_arr)


@pytest.mark.xfail
def test_data_arrays():
    """
    SpillContainer manages a number of numpy arrays that represent the
    properties of the LEs that have been released by the contained spills.
    Here we test that the data arrays are behaving as expected.
    """

    start_time1 = datetime(2012, 1, 1, 12)
    start_time2 = datetime(2012, 1, 2, 12)
    start_time3 = datetime(2012, 1, 3, 12)
    start_time4 = datetime(2012, 1, 4, 12)
    start_time5 = datetime(2012, 1, 5, 12)
    start_position = (23.0, -78.5, 0.0)
    num_elements = 5
    sc = SpillContainer()
    sp1 = PointLineSource(num_elements, start_position,
                                    start_time1)

    sp2 = PointLineSource(num_elements, start_position,
                                    start_time2)

    sp3 = PointLineSource(num_elements, start_position,
                                    start_time3)

    sp4 = SubsurfaceRelease(num_elements, start_position, start_time4)

    sp5 = PointLineSource(num_elements, start_position,
                                    start_time5)

    sc.spills += [sp1, sp2, sp3]

    print sc.spills

    # as we move forward in time, the spills will release LEs
    # in an expected way

    sc.prepare_for_model_run(windage_at)
    sc.release_elements(start_time1, time_step=100)

    assert sc['positions'].shape == (num_elements, 3)
    assert sc['last_water_positions'].shape == (num_elements, 3)
    assert sc['windages'].shape == (num_elements, )  # it should be there.
    with pytest.raises(KeyError):

        # it shouldn't be there.

        assert sc['water_currents'].shape == (num_elements, 3)

    sc.release_elements(start_time1 + timedelta(hours=24),
                        time_step=100)

    assert sc['positions'].shape == (num_elements * 2, 3)
    assert sc['last_water_positions'].shape == (num_elements * 2, 3)
    assert sc['windages'].shape == (num_elements * 2, )  # it should be there.

    with pytest.raises(KeyError):

        # it shouldn't be there.

        assert sc['water_currents'].shape == (num_elements * 2, 3)

    sc.release_elements(start_time2 + timedelta(hours=24),
                        time_step=100)

    assert sc['positions'].shape == (num_elements * 3, 3)
    assert sc['last_water_positions'].shape == (num_elements * 3, 3)
    assert sc['windages'].shape == (num_elements * 3, )  # it should be there.

    with pytest.raises(KeyError):

        # it shouldn't be there.

        assert sc['water_currents'].shape == (num_elements * 3, 3)

    # - When we delete a spill, the previously released LEs from that spill
    #   will stay in the data arrays
    # - All LEs, including from the deleted spill, will maintain their
    #   spill_num property.
    # - When a spill is added with new properties, new items representing
    #   those properties will be created in the data arrays and back-filled
    #   to accommodate the previously released LEs

    del sc.spills[sp2.id]
    sc.spills += sp4
    sc.release_elements(start_time3 + timedelta(hours=24),
                        time_step=100)

    assert sc['positions'].shape == (num_elements * 4, 3)
    assert sc['last_water_positions'].shape == (num_elements * 4, 3)
    assert sc['windages'].shape == (num_elements * 4, )

    # new property should be there with the right shape.

    assert sc['water_currents'].shape == (num_elements * 4, 3)

    # All spill_nums, even the ones that were deleted

    assert set(sc['spill_num']) == set([0, 1, 2, 3])

    # - When we delete a spill, any properties that are not needed by the still
    #   existing spills will be purged.  This purging will happen on the next
    #   release after the delete.

    del sc.spills[sp4.id]
    sc.spills += sp5
    sc.release_elements(start_time4 + timedelta(hours=24),
                        time_step=100)

    assert sc['positions'].shape == (num_elements * 5, 3)
    assert sc['last_water_positions'].shape == (num_elements * 5, 3)
    assert sc['windages'].shape == (num_elements * 5, )

    with pytest.raises(KeyError):

        # extra property from deleted spill should go away

        assert sc['water_currents'].shape == (num_elements * 5, 3)

    # All spill_nums, even the ones that were deleted

    assert set(sc['spill_num']) == set([0, 1, 2, 3, 4])


def test_uncertain_copy():
    """
    test whether creating an uncertain copy of a spill_container works
    """

    release_time = datetime(2012, 1, 1, 12)
    start_time2 = datetime(2012, 1, 2, 12)

    start_position = (23.0, -78.5, 0.0)
    start_position2 = (45.0, 75.0, 0.0)

    num_elements = 100

    sc = SpillContainer()
    spill = PointLineSource(num_elements, start_position,
            release_time)

    sp2 = PointLineSource(num_elements, start_position2,
                                    start_time2)

    sc.spills.add(spill)
    sc.spills.add(sp2)

    u_sc = sc.uncertain_copy()

    assert u_sc.uncertain
    assert len(sc.spills) == len(u_sc.spills)

    # make sure they aren't references to the same spills

    assert sc.spills[spill.id] not in u_sc.spills
    assert sc.spills[sp2.id] not in u_sc.spills

    # make sure they have unique ids:

    for id1 in [s.id for s in sc.spills]:
        for id2 in [s.id for s in u_sc.spills]:
            print id1, id2
            assert not id1 == id2

    # do the spills work?

    sc.prepare_for_model_run(windage_at)
    sc.release_elements(release_time, time_step=100)

    assert sc['positions'].shape == (num_elements, 3)
    assert sc['last_water_positions'].shape == (num_elements, 3)

    # now release second set:

    u_sc.prepare_for_model_run(windage_at)

    assert u_sc['positions'].shape[0] == 0  # nothing released yet.

    u_sc.release_elements(release_time, time_step=100)

    # elements should be there.

    assert u_sc['positions'].shape == (num_elements, 3)
    assert u_sc['last_water_positions'].shape == (num_elements, 3)

    # next release:

    sc.release_elements(release_time + timedelta(hours=24), time_step=100)

    assert sc['positions'].shape == (num_elements * 2, 3)
    assert sc['last_water_positions'].shape == (num_elements * 2, 3)

    # second set should not have changed

    assert u_sc['positions'].shape == (num_elements, 3)
    assert u_sc['last_water_positions'].shape == (num_elements, 3)

    # release second set

    u_sc.release_elements(release_time + timedelta(hours=24),
                          time_step=100)
    assert u_sc['positions'].shape == (num_elements * 2, 3)
    assert u_sc['last_water_positions'].shape == (num_elements * 2, 3)


def test_ordered_collection_api():
    release_time = datetime(2012, 1, 1, 12)
    start_position = (23.0, -78.5, 0.0)
    num_elements = 100

    sc = SpillContainer()
    sc.spills += PointLineSource(num_elements,
            start_position, release_time)
    assert len(sc.spills) == 1


""" SpillContainerPairData tests """


def test_init_SpillContainerPair():
    """
    all this does is test that it can be initilized
    """

    SpillContainerPair()
    SpillContainerPair(True)

    assert True


def test_SpillContainerPair_uncertainty():
    """ test uncertainty property """

    u_scp = SpillContainerPair(True)
    u_scp.uncertain = False
    assert not u_scp.uncertain
    assert not hasattr(u_scp, '_u_spill_container')

    u_scp.uncertain = True
    assert u_scp.uncertain
    assert hasattr(u_scp, '_u_spill_container')


class TestAddSpillContainerPair:

    start_time = datetime(2012, 1, 1, 12)
    start_time2 = datetime(2012, 1, 2, 12)

    start_position = (23.0, -78.5, 0.0)
    start_position2 = (45.0, 75.0, 0.0)

    num_elements = 100

    def test_exception_tuple(self):
        """
        tests that spills can be added to SpillContainerPair object
        """

        spill = PointLineSource(self.num_elements,
                self.start_position, self.start_time)
        sp2 = PointLineSource(self.num_elements,
                self.start_position2, self.start_time2)
        scp = SpillContainerPair(True)

        with pytest.raises(ValueError):
            scp += (spill, sp2, spill)

    def test_exception_uncertainty(self):
        spill = PointLineSource(self.num_elements,
                self.start_position, self.start_time)
        sp2 = PointLineSource(self.num_elements,
                self.start_position2, self.start_time2)
        scp = SpillContainerPair(False)

        with pytest.raises(ValueError):
            scp += (spill, sp2)

    def test_add_spill(self):
        spill = [PointLineSource(self.num_elements,
                 self.start_position, self.start_time) for i in
                 range(2)]

        scp = SpillContainerPair(False)
        scp += (spill[0], )
        scp += spill[1]
        for sp_ix in zip(scp._spill_container.spills,
                         range(len(spill))):
            spill_ = sp_ix[0]
            index = sp_ix[1]
            assert spill_.id == spill[index].id

    def test_add_spillpair(self):
        c_spill = [PointLineSource(self.num_elements,
                   self.start_position, self.start_time) for i in
                   range(2)]

        u_spill = [PointLineSource(self.num_elements,
                   self.start_position2, self.start_time2) for i in
                   range(2)]

        scp = SpillContainerPair(True)

        for sp_tuple in zip(c_spill, u_spill):
            scp += sp_tuple

        for sp_ix in zip(scp._spill_container.spills,
                         range(len(c_spill))):
            spill = sp_ix[0]
            index = sp_ix[1]
            assert spill.id == c_spill[index].id

        for sp_ix in zip(scp._u_spill_container.spills,
                         range(len(c_spill))):
            spill = sp_ix[0]
            index = sp_ix[1]
            assert spill.id == u_spill[index].id

    def test_to_dict(self):
        c_spill = [PointLineSource(self.num_elements,
                   self.start_position, self.start_time) for i in
                   range(2)]

        u_spill = [PointLineSource(self.num_elements,
                   self.start_position2, self.start_time2) for i in
                   range(2)]

        scp = SpillContainerPair(True)

        for sp_tuple in zip(c_spill, u_spill):
            scp += sp_tuple

        dict_ = scp.to_dict()

        for key in dict_.keys():
            if key == 'certain_spills':
                enum_spill = c_spill
            elif key == 'uncertain_spills':
                enum_spill = u_spill

            for (i, spill) in enumerate(enum_spill):
                assert dict_[key]['id_list'][i][0] \
                    == '{0}.{1}'.format(spill.__module__,
                        spill.__class__.__name__)
                assert dict_[key]['id_list'][i][1] == spill.id


def test_get_spill_mask():
    """
    Simple tests for get_spill_mask
    """

    start_time0 = datetime(2012, 1, 1, 12)
    start_time1 = datetime(2012, 1, 2, 12)
    start_time2 = start_time1 + timedelta(hours=1)
    start_position = (23.0, -78.5, 0.0)
    num_elements = 5
    sc = SpillContainer()
    sp0 = PointLineSource(num_elements, start_position,
                                    start_time0)

    sp1 = PointLineSource(num_elements, start_position,
                                    start_time1,
                                    end_position=(start_position[0]
                                    + 0.2, start_position[1] + 0.2,
                                    0.0), end_release_time=start_time1
                                    + timedelta(hours=3))

    sp2 = PointLineSource(num_elements, start_position,
                                    start_time2)

    sc.spills += [sp0, sp1, sp2]

    # as we move forward in time, the spills will release LEs in an
    # expected way

    sc.prepare_for_model_run(windage_at)
    sc.release_elements(start_time0, time_step=100)
    sc.release_elements(start_time0 + timedelta(hours=24),
                        time_step=100)
    sc.release_elements(start_time1 + timedelta(hours=1), time_step=100)
    sc.release_elements(start_time1 + timedelta(hours=3), time_step=100)

    assert all(sc['spill_num'][sc.get_spill_mask(sp2)] == 2)
    assert all(sc['spill_num'][sc.get_spill_mask(sp0)] == 0)
    assert all(sc['spill_num'][sc.get_spill_mask(sp1)] == 1)


def test_eq_spill_container():
    """ test if two spill containers are equal """

    (sp1, sp2) = get_eq_spills()
    sc1 = SpillContainer()
    sc2 = SpillContainer()

    sc1.spills.add(sp1)
    sc2.spills.add(sp2)

    sc1.prepare_for_model_run(windage_at)
    sc1.release_elements(sp1.release_time, 360)

    sc2.prepare_for_model_run(windage_at)
    sc2.release_elements(sp2.release_time, 360)

    assert sc1 == sc2


def test_eq_allclose_spill_container():
    """
    test if two spill containers are equal within 1e-5 tolerance
    adjust the spill_container._array_allclose_atol = 1e-5
    and vary start_positions by 1e-8
    """

    (sp1, sp2) = get_eq_spills()

    # just move one data array a bit

    sp2.start_position = sp2.start_position + (1e-8, 1e-8, 0)

    sc1 = SpillContainer()
    sc2 = SpillContainer()

    sc1.spills.add(sp1)
    sc2.spills.add(sp2)

    sc1.prepare_for_model_run(windage_at)
    sc1.release_elements(sp1.release_time, 360)

    sc2.prepare_for_model_run(windage_at)
    sc2.release_elements(sp2.release_time, 360)

    # need to change both atol

    sc1._array_allclose_atol = 1e-5
    sc2._array_allclose_atol = 1e-5

    assert sc1 == sc2
    assert sc2 == sc1


@pytest.mark.xfail
def test_eq_spill_container_pair():
    """
    SpillContainerPair inherits from SpillContainer so it should
    compute __eq__ and __ne__ in the same way - test it here

    Incomplete - this doesn't currently work!
    Test fails if uncertainty is on whether particles are released or not

    """

    (sp1, sp2) = get_eq_spills()
    scp1 = SpillContainerPair(True)  # uncertainty is on
    scp1.add(sp1)

    scp2 = SpillContainerPair(True)
    scp2.add(sp2)

    assert False


# =============================================================================
#     for sc in zip( scp1.items(), scp2.items()):
#         sc[0].prepare_for_model_run(windage_at)
#         sc[0].release_elements(sp1.release_time, 360)
#         sc[1].prepare_for_model_run(windage_at)
#         sc[1].release_elements(sp2.release_time, 360)
#
#     assert scp1 == scp2
# =============================================================================

def test_ne_spill_container():
    """ test two spill containers are not equal """

    (sp1, sp2) = get_eq_spills()

    # just move one data array a bit

    sp2.start_position = sp2.start_position + (1e-8, 1e-8, 0)

    sc1 = SpillContainer()
    sc2 = SpillContainer()

    sc1.spills.add(sp1)
    sc2.spills.add(sp2)

    sc1.prepare_for_model_run(windage_at)
    sc1.release_elements(sp1.release_time, 360)

    sc2.prepare_for_model_run(windage_at)
    sc2.release_elements(sp2.release_time, 360)

    assert sc1 != sc2


def test_model_step_is_done():
    """
    tests that correct elements are released when their status_codes is toggled
    to basic_types.oil_status.to_be_removed
    """

    release_time = datetime(2012, 1, 1, 12)
    start_time2 = datetime(2012, 1, 2, 12)
    start_position = (23.0, -78.5, 0.0)
    num_elements = 10
    sc = SpillContainer()
    spill = PointLineSource(num_elements, start_position,
            release_time)

    sp2 = PointLineSource(num_elements, start_position,
                                    start_time2)

    sc.spills += [spill, sp2]

    sc.prepare_for_model_run(windage_at)

    sc.release_elements(release_time, time_step=100)
    sc.release_elements(start_time2, time_step=100)

    (sc['status_codes'])[5:8] = oil_status.to_be_removed
    (sc['status_codes'])[14:17] = oil_status.to_be_removed
    sc['status_codes'][19] = oil_status.to_be_removed

    # also make corresponding positions 0 as a way to test

    sc['positions'][5:8, :] = (0, 0, 0)
    sc['positions'][14:17, :] = (0, 0, 0)
    sc['positions'][19, :] = (0, 0, 0)
    sc.model_step_is_done()

    assert sc.num_released == 2 * num_elements - 7

    assert np.all(sc['status_codes'] != oil_status.to_be_removed)
    assert np.all(sc['positions'] == start_position)

    assert np.count_nonzero(sc['spill_num'] == 0) == num_elements - 3
    assert np.count_nonzero(sc['spill_num'] == 1) == num_elements - 4


def get_eq_spills():
    """ returns a tuple of identical PointLineSource objects """

    num_elements = 10
    release_time = datetime(2000, 1, 1, 1)

    spill = PointLineSource(num_elements, (28, -75, 0), release_time)
    spill2 = PointLineSource.new_from_dict(spill.to_dict('create'))

    return (spill, spill2)


if __name__ == '__main__':
    test_rewind()
