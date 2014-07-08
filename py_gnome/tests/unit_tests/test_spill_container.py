#!/usr/bin/env python
'''
Tests the SpillContainer class
'''

from datetime import datetime, timedelta
import copy

import pytest
from pytest import raises
from conftest import sample_sc_release

import numpy
np = numpy

from gnome.basic_types import (oil_status,
                               world_point_type,
                               id_type)
from gnome import array_types
from gnome.spill.elements import (ElementType,
                            InitMassFromTotalMass,
                            InitWindages,
                            InitRiseVelFromDist,
                            InitRiseVelFromDropletSizeFromDist,
                            floating)

from gnome.utilities.distributions import UniformDistribution

from gnome.spill_container import SpillContainer, SpillContainerPair
from gnome.spill import point_line_release_spill, Spill, Release


# additional array_type for testing spill_container functionality
windage_at = {'windages': array_types.windages,
              'windage_range': array_types.windage_range,
              'windage_persist': array_types.windage_persist}

# Sample data for creating spill
num_elements = 100
start_position = (23.0, -78.5, 0.0)
release_time = datetime(2012, 1, 1, 12)
end_position = (24.0, -79.5, 1.0)
end_release_time = datetime(2012, 1, 1, 12) + timedelta(hours=4)


def test_simple_init():
    sc = SpillContainer()
    assert sc != None


def test_length_zero():
    sc = SpillContainer()
    assert len(sc) == 0


def test_length():
    sp = sample_sc_release()

    assert len(sp) == 10


### Helper functions ###
def assert_dataarray_shape_size(sc):
    for key, val in sc.array_types.iteritems():
        assert sc[key].shape == (sc.num_released,) + val.shape
        assert sc[key].dtype == val.dtype

    assert np.all(sc['status_codes']
                      == sc.array_types['status_codes'].initial_value)
    assert np.all(sc['id'] == range(0, sc.num_released))


def assert_sc_single_spill(sc):
    """ standard asserts for a SpillContainer with a single spill """
    assert_dataarray_shape_size(sc)
    assert np.all(sc['spill_num'] == sc['spill_num'][0])

    # only one spill in SpillContainer
    for spill in sc.spills:
        assert np.array_equal(sc['positions'][0], spill.release.start_position)
        assert np.array_equal(sc['positions'][-1], spill.release.end_position)


def test_test_spill_container():
    sc = sample_sc_release()
    assert_sc_single_spill(sc)


@pytest.mark.parametrize("spill",
                         [point_line_release_spill(num_elements,
                                          start_position,
                                          release_time),
                          point_line_release_spill(num_elements,
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
    num_steps = ((spill.release.end_release_time -
                  spill.release.release_time).seconds / time_step + 1)
    for step in range(num_steps):
        current_time = (spill.release.release_time +
                        timedelta(seconds=time_step * step))
        sc.release_elements(time_step, current_time)

    assert sc.num_released == spill.release.num_elements

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
    spills = [point_line_release_spill(num_elements, start_position,
                                       release_time),
              point_line_release_spill(num_elements, start_position,
                    release_time + timedelta(hours=1),
                    end_position, end_release_time)]

    sc.spills.add(spills)

    for spill in spills:
        assert sc.spills[spill.id] == spill

    assert sc.uncertain == uncertain

    time_step = 3600
    num_steps = ((spills[-1].release.end_release_time -
                  spills[-1].release.release_time).seconds / time_step + 1)

    sc.prepare_for_model_run(windage_at)

    for step in range(num_steps):
        current_time = release_time + timedelta(seconds=time_step * step)
        sc.release_elements(time_step, current_time)

    assert sc.num_released == spills[0].release.num_elements * len(spills)
    assert_dataarray_shape_size(sc)

    sc.spills.remove(spills[0].id)

    with raises(KeyError):
        # it shouldn't be there anymore.
        assert sc.spills[spills[0].id] is None

    # however, the data arrays of released particles should be unchanged
    assert sc.num_released == spill.release.num_elements * len(spills)
    assert_dataarray_shape_size(sc)


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

    spills = [point_line_release_spill(num_elements, start_position, release_time),
              point_line_release_spill(num_elements, start_position, release_time2)]
    sc.spills.add(spills)

    sc.prepare_for_model_run(windage_at)

    for time in [release_time, release_time2]:
        sc.release_elements(3600, time)

    assert sc.num_released == num_elements * len(spills)
    for spill in spills:
        assert spill.get('num_released') == spill.release.num_elements

    sc.rewind()
    assert sc.num_released == 0
    assert_dataarray_shape_size(sc)
    for spill in spills:
        assert spill.get('num_released') == 0
        assert spill.release.start_time_invalid


def test_data_access():
    sc = sample_sc_release()

    sc['positions'] += (3.0, 3.0, 3.0)

    assert np.array_equal(sc['positions'], np.ones((sc.num_released, 3),
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

    with raises(ValueError):
        sc['positions'] = new_pos


def test_data_setting_wrong_dtype_error():
    """
    Should get an error when trying to set the data to a different type array
    """

    sc = sample_sc_release()

    new_pos = np.ones((sc.num_released, 3), dtype=np.int32)

    with raises(ValueError):
        sc['positions'] = new_pos


def test_data_setting_wrong_shape_error():
    """
    Should get an error when trying to set the data to a different shape array
    """

    sc = sample_sc_release()

    new_pos = np.ones((sc.num_released, 4), dtype=world_point_type) * 3.0

    with raises(ValueError):
        sc['positions'] = new_pos


def test_data_setting_new():
    """
    Can add a new item to data_arrays. This will automatically update
    SpillContainer's array_types dict

    No rewind necessary. Subsequent releases will initialize the newly added
    numpy array for newly released particles
    """
    spill = point_line_release_spill(20, start_position, release_time,
                            end_release_time=end_release_time)
    # release 10 particles
    time_step = (end_release_time - release_time) / 2
    sc = sample_sc_release(time_step=time_step.seconds, spill=spill)
    released = sc.num_released

    new_arr = np.ones((sc.num_released, 3), dtype=np.float64)
    sc['new_name'] = new_arr

    assert 'new_name' in sc.data_arrays
    assert 'new_name' in sc.array_types
    assert sc['new_name'] is new_arr
    assert_dataarray_shape_size(sc)

    # now release remaining particles and check to see new_name is populated
    # with zeros - default initial_value
    sc.release_elements(time_step.seconds,
                        spill.release.release_time + time_step)
    new_released = sc.num_released - released

    assert_dataarray_shape_size(sc)  # shape is consistent for all arrays
    assert sc.num_released == spill.release.num_elements  # release all elems
    assert np.all(sc['new_name'][-new_released:] ==  # initialized to 0!
                  (0.0, 0.0, 0.0))


def test_data_setting_new_list():
    """
    Should be able to add a new data that's not a numpy array
    """
    sc = sample_sc_release()

    new_arr = range(sc.num_released)

    sc['new_name'] = new_arr

    assert np.array_equal(sc['new_name'], new_arr)
    assert_dataarray_shape_size(sc)


sc_default_array_types = SpillContainer().array_types


class TestAddArrayTypes:
    """
    Cannot add to array_types dict directly.
    - Must either add key, value in prepare_for_model_run()
    - or add a new data_array and an associated ArrayType object will be
      inferred and created
    """
    sc = SpillContainer()
    new_at = array_types.ArrayType((3,), np.float64, 0)

    def default_arraytypes(self):
        """ return array_types back to baseline for SpillContainer """
        self.sc.rewind()
        self.sc.prepare_for_model_run()  # set to anything
        assert self.sc.array_types == sc_default_array_types

    def test_no_addto_array_types(self):
        """
        Cannot add a new ArrayType directly via SpillContainer's array_types
        property
        """
        self.default_arraytypes()
        self.sc.array_types['new_name'] = self.new_at

        assert 'new_name' not in self.sc.array_types

    def test_addto_array_types_prepare_for_model_run(self):
        """
        Can add a new array type via prepare_for_model_run()
        at the beginning of the run
        """
        self.default_arraytypes()
        self.sc.prepare_for_model_run(array_types={'new_name': self.new_at})
        assert 'new_name' in self.sc.array_types

    def test_addto_array_types_via_data_array(self):
        """
        can also add a new data array in the middle of the run, which will
        infer the array_type and add it to array_types dict. This is so in the
        next release, the data is correctly initialized for new array.
        The user is basically responsible for 'backfilling' the data for newly
        added array.
        """
        self.default_arraytypes()

        # this will add a new array type
        new_arr = np.ones((len(self.sc['positions']), 3),
                          dtype=self.new_at.dtype)
        self.sc['new_name'] = new_arr
        assert 'new_name' in self.sc.array_types


def test_array_types_reset():
    """
    check the array_types are reset on rewind() only
    """
    sc = SpillContainer()
    sc.prepare_for_model_run(array_types=windage_at)

    assert 'windages' in sc.array_types

    sc.rewind()
    assert 'windages' not in sc.array_types
    assert sc.array_types == sc_default_array_types

    sc.prepare_for_model_run(array_types=windage_at)
    assert 'windages' in sc.array_types

    # now if we invoke prepare_for_model_run without giving it any array_types
    # it should not reset the dict to default
    sc.prepare_for_model_run()   # set to any datetime
    assert 'windages' in sc.array_types


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
    spill = point_line_release_spill(num_elements, start_position,
            release_time)

    sp2 = point_line_release_spill(num_elements, start_position2,
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
    sc.release_elements(100, release_time)

    assert sc['positions'].shape == (num_elements, 3)
    assert sc['last_water_positions'].shape == (num_elements, 3)

    # now release second set:

    u_sc.prepare_for_model_run(windage_at)

    assert u_sc['positions'].shape[0] == 0  # nothing released yet.

    u_sc.release_elements(100, release_time)

    # elements should be there.

    assert u_sc['positions'].shape == (num_elements, 3)
    assert u_sc['last_water_positions'].shape == (num_elements, 3)

    # next release:

    sc.release_elements(100, release_time + timedelta(hours=24))

    assert sc['positions'].shape == (num_elements * 2, 3)
    assert sc['last_water_positions'].shape == (num_elements * 2, 3)

    # second set should not have changed

    assert u_sc['positions'].shape == (num_elements, 3)
    assert u_sc['last_water_positions'].shape == (num_elements, 3)

    # release second set

    u_sc.release_elements(100, release_time + timedelta(hours=24))
    assert u_sc['positions'].shape == (num_elements * 2, 3)
    assert u_sc['last_water_positions'].shape == (num_elements * 2, 3)


def test_ordered_collection_api():
    release_time = datetime(2012, 1, 1, 12)
    start_position = (23.0, -78.5, 0.0)
    num_elements = 100

    sc = SpillContainer()
    sc.spills += point_line_release_spill(num_elements,
            start_position, release_time)
    assert len(sc.spills) == 1


""" tests w/ element types set for two spills """
el0 = ElementType({'windages': InitWindages((0.02, 0.02), -1),
                   'mass': InitMassFromTotalMass(),
                   'rise_vel': InitRiseVelFromDist(distribution=UniformDistribution(low=1, high=10))
                   })

el1 = ElementType({'windages': InitWindages(),
                   'mass': InitMassFromTotalMass(),
                   'rise_vel': InitRiseVelFromDist()})

arr_types = {'windages': array_types.windages,
             'windage_range': array_types.windage_range,
             'windage_persist': array_types.windage_persist,
             'rise_vel': array_types.rise_vel}


@pytest.mark.parametrize(("elem_type", "arr_types"),
        [((el0, el1), arr_types)])
def test_element_types(elem_type, arr_types, sample_sc_no_uncertainty):
    """
    Tests that the spill_container's data_arrays associated with initializers
    are correctly setup for each spill
    uses sample_sc_no_uncertainty fixture defined in conftest.py
    """
    sc = sample_sc_no_uncertainty
    release_t = None
    for idx, spill in enumerate(sc.spills):
        spill.element_type = elem_type[idx]

        if release_t is None:
            release_t = spill.release.release_time

        # set release time based on earliest release spill
        if spill.release.release_time < release_t:
            release_t = spill.release.release_time

    time_step = 3600
    num_steps = 4   # just run for 4 steps
    sc.prepare_for_model_run(arr_types)

    for step in range(num_steps):
        current_time = release_t + timedelta(seconds=time_step * step)
        sc.release_elements(time_step, current_time)

    # after all steps, check that the element_type parameters were initialized
    # correctly
    for spill in sc.spills:
        spill.element_type
        spill_mask = sc.get_spill_mask(spill)

        for key in spill.element_type.initializers:
            if key in sc.data_arrays:
                if key == 'windage_range':
                    assert (np.all(sc[key][spill_mask] ==
                        spill.element_type.initializers[key].windage_range))
                elif key == 'windage_persist':
                    assert (np.all(sc[key][spill_mask] ==
                        spill.element_type.initializers[key].windage_persist))
                elif key == 'rise_vel':
                    if (isinstance(spill.element_type.initializers[key].distribution,
                                   UniformDistribution)):
                        low = spill.element_type.initializers[key].distribution.low
                        high = spill.element_type.initializers[key].distribution.high

                        assert (np.all(sc[key][spill_mask] >= low))
                        assert (np.all(sc[key][spill_mask] <= high))


def test_init_SpillContainerPair():
    'All this does is test that it can be initialized'
    SpillContainerPair()
    SpillContainerPair(True)

    assert True


def test_SpillContainerPair_uncertainty():
    'test uncertainty property'

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

        spill = point_line_release_spill(self.num_elements,
                self.start_position, self.start_time)
        sp2 = point_line_release_spill(self.num_elements,
                self.start_position2, self.start_time2)
        scp = SpillContainerPair(True)

        with raises(ValueError):
            scp += (spill, sp2, spill)

    def test_exception_uncertainty(self):
        spill = point_line_release_spill(self.num_elements,
                self.start_position, self.start_time)
        sp2 = point_line_release_spill(self.num_elements,
                self.start_position2, self.start_time2)
        scp = SpillContainerPair(False)

        with raises(ValueError):
            scp += (spill, sp2)

    def test_add_spill(self):
        spill = [point_line_release_spill(self.num_elements,
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
        c_spill = [point_line_release_spill(self.num_elements,
                   self.start_position, self.start_time) for i in
                   range(2)]

        u_spill = [point_line_release_spill(self.num_elements,
                   self.start_position2, self.start_time2) for i in
                   range(2)]

        scp = SpillContainerPair(True)

        for sp_tuple in zip(c_spill, u_spill):
            scp += sp_tuple

        for sp, idx in zip(scp._spill_container.spills, range(len(c_spill))):
            assert sp.id == c_spill[idx].id

        for sp, idx in zip(scp._u_spill_container.spills, range(len(c_spill))):
            assert sp.id == u_spill[idx].id

    def test_release_particles(self):
        '''test that the 'id' for uncertain spill container's data is
        starting from 0'''
        spill = [point_line_release_spill(self.num_elements,
                 self.start_position, self.start_time) for i in
                 range(2)]

        scp = SpillContainerPair(True)
        scp += spill[0]
        scp += spill[1]
        for sc in scp.items():
            sc.prepare_for_model_run(windage_at)
            # model sets this for each step
            sc.current_time_stamp = self.start_time
            sc.release_elements(100, self.start_time)

        for key in ['id', 'spill_num', 'age']:
            c_val = scp.LE(key)
            u_val = scp.LE(key, 'uncertain')
            assert np.all(c_val == u_val)

    def test_spill_by_index(self):
        'test spill_by_index returns the correct spill object'
        spill = [point_line_release_spill(self.num_elements,
                                          self.start_position, self.start_time),
                 point_line_release_spill(self.num_elements,
                                          self.start_position, self.start_time)]

        scp = SpillContainerPair(True)
        scp += spill[0]
        scp += spill[1]
        for ix in range(2):
            assert scp.spill_by_index(ix) is spill[ix]
            u_spill = scp.spill_by_index(ix, uncertain=True)
            assert u_spill is not spill[ix]
            assert scp.items()[1].spills[ix] is u_spill

    @pytest.mark.parametrize('json_', ['save', 'webapi'])
    def test_to_dict(self, json_):
        c_spill = [point_line_release_spill(self.num_elements,
                   self.start_position, self.start_time) for i in
                   range(2)]

        u_spill = [point_line_release_spill(self.num_elements,
                   self.start_position2, self.start_time2) for i in
                   range(2)]

        scp = SpillContainerPair(True)

        for sp_tuple in zip(c_spill, u_spill):
            scp += sp_tuple

        toserial = scp.to_dict()
        assert 'spills' in toserial
        assert 'uncertain_spills' in toserial

        for key in ('spills', 'uncertain_spills'):
            if key == 'spills':
                check = c_spill
            else:
                check = u_spill

            alltrue = [check[ix].id == spill['id'] \
                                for ix, spill in enumerate(toserial[key])]
            assert all(alltrue)
            alltrue = [check[ix].obj_type_to_dict() == spill['obj_type'] \
                                for ix, spill in enumerate(toserial[key])]
            assert all(alltrue)


def test_get_spill_mask():
    'Simple tests for get_spill_mask'

    start_time0 = datetime(2012, 1, 1, 12)
    start_time1 = datetime(2012, 1, 2, 12)
    start_time2 = start_time1 + timedelta(hours=1)
    start_position = (23.0, -78.5, 0.0)
    num_elements = 5
    sc = SpillContainer()
    sp0 = point_line_release_spill(num_elements, start_position,
                                    start_time0)

    sp1 = point_line_release_spill(num_elements, start_position,
                                    start_time1,
                                    end_position=(start_position[0]
                                    + 0.2, start_position[1] + 0.2,
                                    0.0), end_release_time=start_time1
                                    + timedelta(hours=3))

    sp2 = point_line_release_spill(num_elements, start_position,
                                    start_time2)

    sc.spills += [sp0, sp1, sp2]

    # as we move forward in time, the spills will release LEs in an
    # expected way

    sc.prepare_for_model_run(windage_at)
    sc.release_elements(100, start_time0)
    sc.release_elements(100, start_time0 + timedelta(hours=24))
    sc.release_elements(100, start_time1 + timedelta(hours=1))
    sc.release_elements(100, start_time1 + timedelta(hours=3))

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
    sc1.release_elements(360, sp1.release.release_time)

    sc2.prepare_for_model_run(windage_at)
    sc2.release_elements(360, sp2.release.release_time)

    assert sc1 == sc2
    assert sc2 == sc1
    assert not (sc1 != sc2)
    assert not (sc2 != sc1)


def test_eq_allclose_spill_container():
    """
    test if two spill containers are equal within 1e-5 tolerance
    adjust the spill_container._array_allclose_atol = 1e-5
    and vary start_positions by 1e-8
    """

    (sp1, sp2) = get_eq_spills()

    # just move one data array a bit

    sp2.start_position = sp2.release.start_position + (1e-8, 1e-8, 0)

    sc1 = SpillContainer()
    sc2 = SpillContainer()

    sc1.spills.add(sp1)
    sc2.spills.add(sp2)

    sc1.prepare_for_model_run(windage_at)
    sc1.release_elements(360, sp1.release.release_time)

    sc2.prepare_for_model_run(windage_at)
    sc2.release_elements(360, sp2.release.release_time)

    # need to change both atol

    sc1._array_allclose_atol = 1e-5
    sc2._array_allclose_atol = 1e-5

    assert sc1 == sc2
    assert sc2 == sc1
    assert not (sc1 != sc2)
    assert not (sc2 != sc1)


@pytest.mark.parametrize("uncertain", [False, True])
def test_eq_spill_container_pair(uncertain):
    """
    SpillContainerPair inherits from SpillContainer so it should
    compute __eq__ and __ne__ in the same way - test it here

    Incomplete - this doesn't currently work!
    Test fails if uncertainty is on whether particles are released or not
    This is because 'id' of uncertain spills do not match and one should not
    expect them to match.

    todo: remove 'id' property as a check for equality. This requires changes
          in persisting logic. Update persistence then revisit this test
          and simplify it
    """
    (sp1, sp2) = get_eq_spills()

    # windages array will not match after elements are released so lets not
    # add any more types to data_arrays for this test. Just look at base
    # array_types for SpillContainer's and ensure the data matches for them
    #sp1.element_type = ElementType()
    #sp2.element_type = ElementType()

    scp1 = SpillContainerPair(uncertain)  # uncertainty is on
    scp1.add(sp1)

    scp2 = SpillContainerPair(uncertain)
    if uncertain:
        u_sp1 = [scp1.items()[1].spills[spill.id] for spill in
                 scp1.items()[1].spills][0]

        u_sp2 = copy.deepcopy(u_sp1)
        # deepcopy does not match ids!
        # for test, we need these to match so force them to be equal here
        u_sp2._id = u_sp1.id

        scp2.add((sp2, u_sp2))
    else:
        scp2.add(sp2)

    for sc in zip(scp1.items(), scp2.items()):
        sc[0].prepare_for_model_run()
        sc[0].release_elements(360, sp1.release.release_time)
        sc[1].prepare_for_model_run()
        sc[1].release_elements(360, sp2.release.release_time)

    assert scp1 == scp2
    assert scp2 == scp1
    assert not (scp1 != scp2)
    assert not (scp2 != scp1)


def test_ne_spill_container():
    """ test two spill containers are not equal """

    (sp1, sp2) = get_eq_spills()

    # just move one data array a bit

    sp2.release.start_position = sp2.release.start_position + (1e-8, 1e-8, 0)

    sc1 = SpillContainer()
    sc2 = SpillContainer()

    sc1.spills.add(sp1)
    sc2.spills.add(sp2)

    sc1.prepare_for_model_run(windage_at)
    sc1.release_elements(360, sp1.release.release_time)

    sc2.prepare_for_model_run(windage_at)
    sc2.release_elements(360, sp2.release.release_time)

    assert sc1 != sc2
    assert sc2 != sc1
    assert not (sc1 == sc2)
    assert not (sc2 == sc1)


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
    sp1 = point_line_release_spill(num_elements, start_position,
            release_time)

    sp2 = point_line_release_spill(num_elements, start_position,
                                    start_time2)

    sc.spills += [sp1, sp2]

    sc.prepare_for_model_run(windage_at)

    sc.release_elements(100, release_time)
    sc.release_elements(100, start_time2)

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


def test_SpillContainer_add_array_types():
    '''
    Test an array_type is dynamically added/subtracted from SpillContainer if
    it is contained in Initailizer's array_types property.

    For example:

        Add 'rise_vel' initializer, InitRiseVelFromDropletSizeFromDist()) is
        added to Spill's element_type object. Now, the array_types for this
        initailizer are 'rise_vel' and 'droplet_diameter'. Only if a
        RiseVelocityMover is added to the model in which case the Model
        provides 'rise_vel' as an array_type to the SpillContainer to append
        it to its own list, then the SpillContainer will also add the
        'droplet_diameter' array_type that is additionally set by the
        Initializer but is not explicitly required by the Mover.
    '''
    sc = SpillContainer()
    s = Spill(Release(datetime(2014, 1, 1, 12, 0), 0))
    s.set_initializer('rise_vel', InitRiseVelFromDropletSizeFromDist())
    sc.spills += s
    assert 'rise_vel' not in sc.array_types
    assert 'droplet_diameter' not in sc.array_types

    # Now say you added RiseVelocityMover and the Model collects ArrayTypes
    # from all movers and passes it into SpillContainer's prepare_for_model_run
    #
    sc.prepare_for_model_run(array_types={'rise_vel': array_types.rise_vel})
    assert 'rise_vel' in sc.array_types
    assert 'droplet_diameter' in sc.array_types

    # calling prepare_for_model_run without different array_types keeps the
    # previously added 'rise_vel' array_types - always rewind if you want to
    # clear out the state and reset array_types to original data
    sc.prepare_for_model_run()
    assert 'rise_vel' in sc.array_types
    assert 'droplet_diameter' in sc.array_types

    # Now let's rewind array_types and these extra properties should disappear
    # they are only added after the prepare_for_model_run step
    sc.rewind()
    sc.prepare_for_model_run()
    assert 'rise_vel' not in sc.array_types
    assert 'droplet_diameter' not in sc.array_types


def get_eq_spills():
    """
    returns a tuple of identical point_line_release_spill objects

    Set the spill's element_type is to floating(windage_range=(0, 0))
    since the default, floating(), uses randomly generated values for initial
    data array values and these will not match for the two spills.

    TODO: Currently does not persist the element_type object.
    spill.to_dict('save') does not persist this attribute - Fix this.
    """
    num_elements = 10
    release_time = datetime(2000, 1, 1, 1)

    spill = point_line_release_spill(num_elements,
                            (28, -75, 0),
                            release_time,
                            element_type=floating(windage_range=(0, 0)))
    #dict_ = spill.to_dict('save')
    #spill2 = spill.new_from_dict(dict_)
    spill2 = copy.deepcopy(spill)

    # IDs will not match! force this so our tests work
    spill2._id = spill.id

    # check here if equal spills didn't get created - fail this function
    assert spill == spill2

    return (spill, spill2)


if __name__ == '__main__':
    test_rewind()
