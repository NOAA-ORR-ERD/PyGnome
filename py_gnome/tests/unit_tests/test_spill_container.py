#!/usr/bin/env python

"""
Tests the SpillContainer class
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from gnome.basic_types import oil_status, id_type, world_point_type

from gnome.spill_container import (SpillContainer,
                                   TestSpillContainer,
                                   SpillContainerPair)
from gnome.spill import Spill, PointSourceRelease, SubsurfaceRelease

# only required to setup data arrays correctly
from gnome.element_types import ArrayType, basic, windage

basic_at = dict(basic)
windage_at = dict(basic_at.items() + dict(windage).items())


def test_simple_init():
    sc = SpillContainer()
    assert sc


def test_test_spill_container():
    pos = (28.0, -75.0, 0.0)
    num_elements = 10
    sc = TestSpillContainer(num_elements, (28, -75, 0))

    assert sc['positions'].shape == (10, 3)

    assert np.array_equal(sc['positions'][0], pos)
    assert np.array_equal(sc['positions'][-1], pos)
    assert sc.all_array_types['status_codes'].initial_value == oil_status.in_water
    assert np.all(sc['status_codes'] == sc.all_array_types['status_codes'].initial_value)


## real testing involves adding spills!
def test_one_simple_spill():
    start_time = datetime(2012, 1, 1, 12)
    start_position = (23.0, -78.5, 0.0)
    num_elements = 100
    sc = SpillContainer()
    spill = PointSourceRelease(num_elements,
                                start_position,
                                start_time)

    sc.spills.add(spill)
    sc.prepare_for_model_run(spill.release_time, windage_at)
    sc.release_elements(start_time, 1)

    assert sc.num_elements == num_elements

    assert sc['positions'].shape == (num_elements, 3)
    assert sc['last_water_positions'].shape == (num_elements, 3)

    sc.prepare_for_model_step(start_time + timedelta(hours=24))

    assert sc['positions'].shape == (num_elements, 3)
    assert sc['last_water_positions'].shape == (num_elements, 3)

    assert np.array_equal(sc['positions'][0], start_position)


## multiple spills with different release times:
def test_multiple_spills():
    start_time = datetime(2012, 1, 1, 12)
    start_time2 = datetime(2012, 1, 2, 12)
    start_position = (23.0, -78.5, 0.0)
    num_elements = 100
    sc = SpillContainer()

    spill = PointSourceRelease(num_elements,
                               start_position,
                               start_time)

    sp2 = PointSourceRelease(num_elements,
                             start_position,
                             start_time2)

    sc.spills += [spill, sp2]
    print sc.spills

    sc.prepare_for_model_run(start_time, windage_at)
    sc.release_elements(start_time, time_step=100)

    assert sc['positions'].shape == (num_elements, 3)
    assert sc['last_water_positions'].shape == (num_elements, 3)

    sc.release_elements(start_time + timedelta(hours=24), time_step=100)

    assert sc['positions'].shape == (num_elements * 2, 3)
    assert sc['last_water_positions'].shape == (num_elements * 2, 3)

    ## check the get_spill method
    assert sc.spills[spill.id] == spill
    assert sc.spills[sp2.id] == sp2

    ## check remove
    sc.spills.remove(spill.id)
    with pytest.raises(KeyError):
        # it shouldn't be there anymore.
        assert sc.spills[spill.id] is None


def test_add_data_array_spill_container():
    """ add a custom data array to spill container and see that it works"""
    spill = PointSourceRelease(num_elements=10,
                               start_position=(23.0, -78.5, 0.0),
                               release_time=datetime(2012, 1, 1, 12))
    sc = SpillContainer()
    sc.spills.add(spill)
    sc.prepare_for_model_run(spill.release_time, windage_at)
    sc.release_elements(spill.release_time, 1)

    with pytest.raises(ValueError):
        sc['positions'] = np.zeros(len(sc['spill_num']) - 1, dtype=np.int)

    with pytest.raises(ValueError):
        sc['spill_num'] = np.zeros(len(sc['spill_num']), dtype=np.int)

    sc['spill_num'] = np.zeros(len(sc['spill_num']), dtype=id_type)


def test_rewind():
    start_time = datetime(2012, 1, 1, 12)
    start_time2 = datetime(2012, 1, 2, 12)
    start_position = (23.0, -78.5, 0.0)
    num_elements = 100
    sc = SpillContainer()

    spill = PointSourceRelease(num_elements,
                               start_position,
                               start_time)

    sp2 = PointSourceRelease(num_elements,
                             start_position,
                             start_time2)

    sc.spills.add(spill)
    sc.spills.add(sp2)
    sc.prepare_for_model_step(start_time)
    sc.prepare_for_model_step(start_time + timedelta(hours=24))

    sc.rewind()
    assert spill.num_released == 0
    assert sp2.num_released == 0


def test_rewind2():
    """
    test that extra arrays are removed on a rewind

    # not much of a test, really -- add more?
    """
    start_time = datetime(2012, 1, 1, 12)
    start_time2 = datetime(2012, 1, 2, 12)
    start_position = (23.0, -78.5, 0.0)
    num_elements = 100
    sc = SpillContainer()

    spill = Spill(num_elements)

    sp2 = PointSourceRelease(num_elements,
                             start_position,
                             start_time2)

    sc.spills.add(spill)
    sc.spills.add(sp2)

    sc.prepare_for_model_step(start_time)
    sc.prepare_for_model_step(start_time + timedelta(hours=24))

    sc.spills.remove(spill.id)

    sc.rewind()
    print "id of spill 2", sp2.id
    assert sp2.num_released == 0


def test_data_access():
    sp = TestSpillContainer(10, (0, 0, 0),)

    sp['positions'] += (3.0, 3.0, 3.0)

    assert np.array_equal(sp['positions'],
                          np.ones((10, 3), dtype=world_point_type) * 3.0
                          )


def test_data_setting():
    sp = TestSpillContainer(num_elements=10)

    new_pos = np.ones((10, 3), dtype=world_point_type) * 3.0

    sp['positions'] = new_pos

    assert np.array_equal(sp['positions'],
                          new_pos
                          )


def test_data_setting_error1():
    """
    Should get an error when trying to set the data to a different size array
    """
    sp = TestSpillContainer(num_elements=10)

    new_pos = np.ones((12, 3), dtype=world_point_type) * 3.0

    with pytest.raises(ValueError):
        sp['positions'] = new_pos


def test_data_setting_error2():
    """
    Should get an error when trying to set the data to a different type array
    """
    sp = TestSpillContainer(num_elements=10)

    new_pos = np.ones((10, 3), dtype=np.int32)

    with pytest.raises(ValueError):
        sp['positions'] = new_pos


def test_data_setting_error3():
    """
    Should get an error when trying to set the data to a different shape array
    """
    sp = TestSpillContainer(num_elements=10)

    new_pos = np.ones((10, 4), dtype=world_point_type) * 3.0

    with pytest.raises(ValueError):
        sp['positions'] = new_pos


def test_addto_all_array_types():
    """
    Can add a new ArrayType to all_array_types; however, must rewind model
    to get the 'new_name' array in data_arrays
    """
    sc = TestSpillContainer(num_elements=10)
    sc.all_array_types['new_name'] = ArrayType((3,), np.float64, 0)

    # MUST rewind and release elements again to get new_name in data_arrays
    sc.rewind()
    sc.release_elements(sc.current_time_stamp, time_step=100)

    new_arr = np.ones((10, 3), dtype=np.float64)
    sc['new_name'] = new_arr

    assert 'new_name' in sc.data_arrays_dict
    assert sc['new_name'] is new_arr


def test_data_setting_new():
    """
    Can add a new item to data_arrays. This will automatically update
    SpillContainer's all_array_types dict

    No rewind necessary
    """
    sc = TestSpillContainer(num_elements=10)

    new_arr = np.ones((10, 3), dtype=np.float64)
    sc['new_name'] = new_arr

    assert 'new_name' in sc.data_arrays_dict
    assert sc['new_name'] is new_arr


def test_data_setting_new_list():
    """
    Should be able to add a new data that's not a numpy array
    """
    sp = TestSpillContainer(num_elements=10)

    new_arr = range(10)

    sp['new_name'] = new_arr

    assert np.array_equal(sp['new_name'],  new_arr)


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

    sp1 = PointSourceRelease(num_elements,
                             start_position,
                             start_time1)

    sp2 = PointSourceRelease(num_elements,
                             start_position,
                             start_time2)

    sp3 = PointSourceRelease(num_elements,
                             start_position,
                             start_time3)

    sp4 = SubsurfaceRelease(num_elements,
                            start_position,
                            start_time4)

    sp5 = PointSourceRelease(num_elements,
                             start_position,
                             start_time5)

    sc.spills += [sp1, sp2, sp3]

    print sc.spills

    # as we move forward in time, the spills will release LEs
    # in an expected way
    sc.prepare_for_model_run(start_time1, windage_at)
    sc.release_elements(start_time1, time_step=100)

    assert sc['positions'].shape == (num_elements, 3)
    assert sc['last_water_positions'].shape == (num_elements, 3)
    assert sc['windages'].shape == (num_elements, )  # it should be there.
    with pytest.raises(KeyError):
        # it shouldn't be there.
        assert sc['water_currents'].shape == (num_elements, 3)

    sc.release_elements(start_time1 + timedelta(hours=24), time_step=100)

    assert sc['positions'].shape == (num_elements * 2, 3)
    assert sc['last_water_positions'].shape == (num_elements * 2, 3)
    assert sc['windages'].shape == (num_elements * 2, )  # it should be there.

    with pytest.raises(KeyError):
        # it shouldn't be there.
        assert sc['water_currents'].shape == (num_elements * 2, 3)

    sc.release_elements(start_time2 + timedelta(hours=24), time_step=100)

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
    sc.release_elements(start_time3 + timedelta(hours=24), time_step=100)

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
    sc.release_elements(start_time4 + timedelta(hours=24), time_step=100)

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
    start_time = datetime(2012, 1, 1, 12)
    start_time2 = datetime(2012, 1, 2, 12)

    start_position = (23.0, -78.5, 0.0)
    start_position2 = (45.0, 75.0, 0.0)

    num_elements = 100

    sc = SpillContainer()
    spill = PointSourceRelease(num_elements,
                               start_position,
                               start_time)

    sp2 = PointSourceRelease(num_elements,
                             start_position2,
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
    sc.prepare_for_model_run(start_time, windage_at)
    sc.release_elements(start_time, time_step=100)

    assert sc['positions'].shape == (num_elements, 3)
    assert sc['last_water_positions'].shape == (num_elements, 3)

    # now release second set:
    u_sc.prepare_for_model_run(start_time, windage_at)

    assert u_sc['positions'].shape[0] == 0  # nothing released yet.

    u_sc.release_elements(start_time, time_step=100)
    # elements should be there.
    assert u_sc['positions'].shape == (num_elements, 3)
    assert u_sc['last_water_positions'].shape == (num_elements, 3)

    # next release:
    sc.release_elements(start_time + timedelta(hours=24),
                        time_step=100)

    assert sc['positions'].shape == (num_elements * 2, 3)
    assert sc['last_water_positions'].shape == (num_elements * 2, 3)

    # second set should not have changed
    assert u_sc['positions'].shape == (num_elements, 3)
    assert u_sc['last_water_positions'].shape == (num_elements, 3)

    # release second set
    u_sc.release_elements(start_time + timedelta(hours=24), time_step=100)
    assert u_sc['positions'].shape == (num_elements * 2, 3)
    assert u_sc['last_water_positions'].shape == (num_elements * 2, 3)


def test_ordered_collection_api():
    start_time = datetime(2012, 1, 1, 12)
    start_position = (23.0, -78.5, 0.0)
    num_elements = 100

    sc = SpillContainer()
    sc.spills += PointSourceRelease(num_elements,
                                    start_position,
                                    start_time)
    assert len(sc.spills) == 1

## SpillContainerPairData tests.


def test_init_SpillContainerPair():
    """
    all this does is test that it can be initilized
    """
    scp = SpillContainerPair()
    u_scp = SpillContainerPair(True)

    print scp, u_scp


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
        spill = PointSourceRelease(self.num_elements,
                                   self.start_position, self.start_time)
        sp2 = PointSourceRelease(self.num_elements,
                                 self.start_position2, self.start_time2)
        scp = SpillContainerPair(True)

        with pytest.raises(ValueError):
            scp += (spill, sp2, spill)

    def test_exception_uncertainty(self):
        spill = PointSourceRelease(self.num_elements,
                                   self.start_position, self.start_time)
        sp2 = PointSourceRelease(self.num_elements,
                                 self.start_position2, self.start_time2)
        scp = SpillContainerPair(False)

        with pytest.raises(ValueError):
            scp += (spill, sp2)

    def test_add_spill(self):
        spill = [PointSourceRelease(self.num_elements,
                                    self.start_position, self.start_time)
                 for i in range(2)]

        scp = SpillContainerPair(False)
        scp += (spill[0],)
        scp += spill[1]
        for sp_ix in zip(scp._spill_container.spills, range(len(spill))):
            spill_ = sp_ix[0]
            index = sp_ix[1]
            assert spill_.id == spill[index].id

    def test_add_spillpair(self):
        c_spill = [PointSourceRelease(self.num_elements,
                                      self.start_position, self.start_time)
                   for i in range(2)]

        u_spill = [PointSourceRelease(self.num_elements,
                                      self.start_position2, self.start_time2)
                   for i in range(2)]

        scp = SpillContainerPair(True)

        for sp_tuple in zip(c_spill, u_spill):
            scp += sp_tuple

        for sp_ix in zip(scp._spill_container.spills, range(len(c_spill))):
            spill = sp_ix[0]
            index = sp_ix[1]
            assert spill.id == c_spill[index].id

        for sp_ix in zip(scp._u_spill_container.spills, range(len(c_spill))):
            spill = sp_ix[0]
            index = sp_ix[1]
            assert spill.id == u_spill[index].id

    def test_to_dict(self):
        c_spill = [PointSourceRelease(self.num_elements,
                                      self.start_position, self.start_time)
                   for i in range(2)]

        u_spill = [PointSourceRelease(self.num_elements,
                                      self.start_position2, self.start_time2)
                   for i in range(2)]

        scp = SpillContainerPair(True)

        for sp_tuple in zip(c_spill, u_spill):
            scp += sp_tuple

        dict_ = scp.to_dict()

        for key in dict_.keys():
            if key == 'certain_spills':
                enum_spill = c_spill
            elif key == 'uncertain_spills':
                enum_spill = u_spill

            for i, spill in enumerate(enum_spill):
                assert dict_[key]['id_list'][i][0] == "{0}.{1}".format(spill.__module__,
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

    sp0 = PointSourceRelease(num_elements,
                             start_position,
                             start_time0)

    sp1 = PointSourceRelease(num_elements,
                             start_position,
                             start_time1,
                             end_position=(start_position[0] + 0.2,
                                           start_position[1] + 0.2, 0.0),
                             end_release_time=start_time1 + timedelta(hours=3)
                             )

    sp2 = PointSourceRelease(num_elements,
                             start_position,
                             start_time2)

    sc.spills += [sp0, sp1, sp2]

    # as we move forward in time, the spills will release LEs in an
    # expected way
    sc.prepare_for_model_run(start_time0, windage_at)
    sc.release_elements(start_time0, time_step=100)
    sc.release_elements(start_time0 + timedelta(hours=24), time_step=100)
    sc.release_elements(start_time1 + timedelta(hours=1), time_step=100)
    sc.release_elements(start_time1 + timedelta(hours=3), time_step=100)

    assert all(sc['spill_num'][sc.get_spill_mask(sp2)] == 2)
    assert all(sc['spill_num'][sc.get_spill_mask(sp0)] == 0)
    assert all(sc['spill_num'][sc.get_spill_mask(sp1)] == 1)


def test_eq_spill_container1():
    """ test if two spill containers are equal """
    (sp1, sp2) = get_eq_spills()
    sc1 = SpillContainer()
    sc2 = SpillContainer()

    sc1.spills.add(sp1)
    sc2.spills.add(sp2)

    sc1.prepare_for_model_run(sp1.release_time, windage_at)
    sc1.release_elements(sp1.release_time, 360)

    sc2.prepare_for_model_run(sp2.release_time, windage_at)
    sc2.release_elements(sp2.release_time, 360)

    assert sc1 == sc2


def test_eq_spill_container2():
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

    sc1.prepare_for_model_run(sp1.release_time, windage_at)
    sc1.release_elements(sp1.release_time, 360)

    sc2.prepare_for_model_run(sp2.release_time, windage_at)
    sc2.release_elements(sp2.release_time, 360)

    # need to change both atol
    sc1._array_allclose_atol = 1e-5
    sc2._array_allclose_atol = 1e-5

    assert sc1 == sc2
    assert sc2 == sc1


def test_eq_spill_container_pair():
    """
    SpillContainerPair inherits from SpillContainer so it should
    compute __eq__ and __ne__ in the same way - test it here
    """
    pass


def test_ne_spill_container():
    """ test two spill containers are not equal """
    (sp1, sp2) = get_eq_spills()

    # just move one data array a bit
    sp2.start_position = sp2.start_position + (1e-8, 1e-8, 0)

    sc1 = SpillContainer()
    sc2 = SpillContainer()

    sc1.spills.add(sp1)
    sc2.spills.add(sp2)

    sc1.prepare_for_model_run(sp1.release_time, windage_at)
    sc1.release_elements(sp1.release_time, 360)

    sc2.prepare_for_model_run(sp2.release_time, windage_at)
    sc2.release_elements(sp2.release_time, 360)

    assert sc1 != sc2


def test_model_step_is_done():
    """
    tests that correct elements are released when their status_codes is toggled
    to basic_types.oil_status.to_be_removed
    """
    start_time = datetime(2012, 1, 1, 12)
    start_time2 = datetime(2012, 1, 2, 12)
    start_position = (23.0, -78.5, 0.0)
    num_elements = 10
    sc = SpillContainer()

    spill = PointSourceRelease(num_elements,
                               start_position,
                               start_time)

    sp2 = PointSourceRelease(num_elements,
                             start_position,
                             start_time2)

    sc.spills += [spill, sp2]

    sc.prepare_for_model_run(start_time, windage_at)

    sc.release_elements(start_time, time_step=100)
    sc.release_elements(start_time2, time_step=100)

    sc['status_codes'][5:8] = oil_status.to_be_removed
    sc['status_codes'][14:17] = oil_status.to_be_removed
    sc['status_codes'][19] = oil_status.to_be_removed

    # also make corresponding positions 0 as a way to test
    sc['positions'][5:8, :] = (0, 0, 0)
    sc['positions'][14:17, :] = (0, 0, 0)
    sc['positions'][19, :] = (0, 0, 0)
    sc.model_step_is_done()

    assert sc.num_elements == 2 * num_elements - 7

    assert np.all(sc['status_codes'] != oil_status.to_be_removed)
    assert np.all(sc['positions'] == start_position)

    assert np.count_nonzero(sc['spill_num'] == 0) == num_elements - 3
    assert np.count_nonzero(sc['spill_num'] == 1) == num_elements - 4


""" Helper function """


def get_eq_spills():
    """ returns a tuple of identical PointSourceRelease objects """
    num_elements = 10
    release_time = datetime(2000, 1, 1, 1)

    spill = PointSourceRelease(num_elements, (28, -75, 0), release_time)
    spill2 = PointSourceRelease.new_from_dict(spill.to_dict('create'))

    return (spill, spill2)


if __name__ == "__main__":
    test_rewind2()
