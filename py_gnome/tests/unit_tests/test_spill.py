#!/usr/bin/env python

"""
Tests the spill code.
"""

from datetime import datetime, timedelta
import copy

import pytest

import numpy as np

from gnome.spill import (Spill,
                         PointLineSource)
    #SpatialRelease, SubsurfaceSpill, \
    #SubsurfaceRelease, SpatialRelease
from gnome import array_types

from conftest import mock_append_data_arrays


# Used to mock SpillContainer functionality of creating/appending data_arrays
arr_types = dict(array_types.SpillContiner)


def initialize_arrays(arr_types):
    """
    Serve the function of the SpillContainer since that will initialize
    the data_arrays and pass that onto the Spill
    """
    return


def test_deepcopy():
    """
    only tests that the spill_nums work -- not sure about anything else...

    test_spill_container does test some other issues.
    """

    sp1 = Spill()
    sp2 = copy.deepcopy(sp1)
    assert sp1 is not sp2

    # try deleting the copy, and see if any errors result

    del sp2
    del sp1


def test_copy():
    """
    only tests that the spill_nums work -- not sure about anything else...
    """

    sp1 = Spill()
    sp2 = copy.copy(sp1)
    assert sp1 is not sp2

    # try deleting the copy, and see if any errors result

    del sp1
    del sp2


def test_uncertain_copy():
    """
    only tests a few things...
    """

    spill = PointLineSource(
        num_elements=100,
        start_position=(28, -78, 0.),
        release_time=datetime.now(),
        end_position=(29, -79, 0.),
        end_release_time=datetime.now() + timedelta(hours=24),
        windage_range=(.02, .03),
        windage_persist=0,
        )

    u_spill = spill.uncertain_copy()

    assert u_spill is not spill
    assert np.array_equal(u_spill.start_position, spill.start_position)
    del spill
    del u_spill


class Test_PointLineSource(object):

    num_elements = 10
    start_position = (-128.3, 28.5, 0)
    release_time = datetime(2012, 8, 20, 13)
    timestep = 3600  # one hour in seconds

    def release_and_assert(self,
                           sp, release_time, timestep, data_arrays,
                           expected_num_released):
        """
        Helper function.
        For each release test in this function, group the common actions
        in this function.

        :param sp: spill object
        :param release_time: release time for particles
        :param timestep: timestep to use for releasing particles
        :param data_arrays: data_arrays to which new elements are appended.
            dict containing numpy arrays for values. Serves the same
            function as gnome.spill_container.SpillContainer().data_arrays
        :param expected_num_released: number of particles that we expect
            to release for this timestep. This is used for assertions.

        It returns a copy of the data_arrays after appending the newly
        released particles to it. This is so the caller can do more
        assertions against it.
        Also so we can keep appending to data_arrays since that is what the
        SpillContainer will work until a rewind.
        """
        prev_num_rel = sp.num_released
        num = sp.num_elements_to_release(release_time, timestep)
        assert num == expected_num_released

        # updated after set_newparticle_values is called
        assert prev_num_rel == sp.num_released

        data_arrays = mock_append_data_arrays(arr_types, num, data_arrays)

        # todo: maybe better to return 0 in num_elements_to_release
        # revisit this functionality
        if num is not None:
            # only invoked if particles are released
            sp.set_newparticle_values(num, release_time, timestep, data_arrays)
            assert sp.num_released == prev_num_rel + expected_num_released

        assert data_arrays['positions'].shape == (sp.num_released, 3)

        return data_arrays

    def test_init(self):
        sp = PointLineSource(num_elements=self.num_elements,
                start_position=self.start_position,
                release_time=self.release_time)

        assert sp.num_elements == self.num_elements
        assert np.all(sp.start_position == self.start_position)
        assert np.all(sp.end_position == self.start_position)
        assert sp.release_time == self.release_time
        assert np.all(sp.end_position == self.start_position)

    def test_noparticles_model_run_after_release_time(self):
        """
        Tests that the spill doesn't release anything if the first call
        to release elements is after the release time.
        This so that if the user sets the model start time after the spill,
        they don't get anything.
        """
        sp = PointLineSource(num_elements=self.num_elements,
                start_position=self.start_position,
                release_time=self.release_time)

        # Test no particles released for following conditions
        #     current_time > spill's release_time
        #     current_time + timedelta > spill's release_time
        for rel_delay in range(1, 3):
            num = sp.num_elements_to_release(self.release_time
                                   + timedelta(hours=rel_delay),
                                   time_step=30 * 60)
            assert num is None

        # rewind and it should work
        sp.rewind()
        data_arrays = self.release_and_assert(sp, self.release_time, 30 * 60,
                                {}, self.num_elements)
        assert np.alltrue(data_arrays['positions'] == self.start_position)

    def test_noparticles_model_run_before_release_time(self):
        """
        Tests that the spill doesn't release anything if the first call
        to num_elements_to_release is before the release_time + timestep.
        """

        sp = PointLineSource(num_elements=self.num_elements,
                start_position=self.start_position,
                release_time=self.release_time)
        print 'release_time:', self.release_time
        timestep = 360  # seconds

        # right before the release
        num = sp.num_elements_to_release(self.release_time -
                                         timedelta(seconds=360), timestep)
        assert num is None

        # right after the release
        data_arrays = self.release_and_assert(sp, self.release_time -
                                timedelta(seconds=1),
                                timestep, {}, self.num_elements)
        assert np.alltrue(data_arrays['positions'] == self.start_position)

    def test_inst_point_release(self):
        """
        Test all particles for an instantaneous point release are released
        correctly.
        """
        sp = PointLineSource(num_elements=self.num_elements,
                start_position=self.start_position,
                release_time=self.release_time)
        timestep = 3600  # seconds

        # release all particles
        data_arrays = self.release_and_assert(sp, self.release_time,
                                timestep, {}, self.num_elements)
        assert np.alltrue(data_arrays['positions'] == self.start_position)

        # no more particles to release since all particles have been released
        num = sp.num_elements_to_release(self.release_time + timedelta(10),
                timestep)
        assert num is None

        # reset and try again
        sp.rewind()
        assert sp.num_released == 0
        num = sp.num_elements_to_release(self.release_time - timedelta(10),
                                         timestep)
        assert num is None
        assert sp.num_released == 0

        # release all particles
        data_arrays = self.release_and_assert(sp, self.release_time,
                                timestep, {}, self.num_elements)
        assert np.alltrue(data_arrays['positions'] == self.start_position)

    def test_cont_point_release(self):
        """
        Time varying release so release_time < end_release_time. It releases
        particles over 10 hours. start_position == end_position so it is still
        a point source

        It simulates how particles could be released by a Model with a variable
        timestep
        """
        sp = PointLineSource(num_elements=100,
                start_position=self.start_position,
                release_time=self.release_time,
                end_release_time=self.release_time
                + timedelta(hours=10))
        timestep = 3600  # one hour in seconds

        """
        Release elements incrementally to test continuous release

        4 times and timesteps over which elements are released. The timesteps
        are variable
        """
        # at exactly the release time -- ten get released at start_position
        # one hour into release -- ten more released
        # keep appending to data_arrays in same manner as SpillContainer would
        # 1-1/2 hours into release - 5 more
        # at end -- rest (75 particles) should be released
        data_arrays = {}
        delay_after_rel_time = [timedelta(hours=0),
                                timedelta(hours=1),
                                timedelta(hours=2),
                                timedelta(hours=10)]
        ts = [timestep, timestep, timestep / 2, timestep]
        exp_num_released = [10, 10, 5, 75]

        for ix in range(4):
            data_arrays = self.release_and_assert(sp,
                                self.release_time + delay_after_rel_time[ix],
                                ts[ix], data_arrays, exp_num_released[ix])
            assert np.alltrue(data_arrays['positions'] == self.start_position)

        assert sp.num_released == sp.num_elements

        # rewind and reset data arrays for new release
        sp.rewind()
        data_arrays = {}

        # 360 second time step: should release first LE
        # In 3600 sec, 10 particles are released so one particle every 360sec
        # release one particle each over (360, 720) seconds
        for ix in range(2):
            ts = ix * 360 + 360
            data_arrays = self.release_and_assert(sp, self.release_time, ts,
                                                  data_arrays, 1)
            assert np.alltrue(data_arrays['positions'] == self.start_position)

    def test_inst_line_release(self):
        """
        release all elements instantaneously but
        start_position != end_position so they are released along a line
        """
        sp = PointLineSource(num_elements=11,
                start_position=(-128.0, 28.0, 0),
                release_time=self.release_time, end_position=(-129.0,
                29.0, 0))
        data_arrays = self.release_and_assert(sp, self.release_time,
                                              600, {}, sp.num_elements)

        assert data_arrays['positions'].shape == (11, 3)
        assert np.array_equal(data_arrays['positions'][:, 0],
                              np.linspace(-128, -129, 11))
        assert np.array_equal(data_arrays['positions'][:, 1],
                              np.linspace(28, 29, 11))

        assert sp.num_released == 11

    def test_cont_line_release1(self):
        """
        testing a release that is releasing while moving over time

        In this one it all gets released in the first time step.
        """
        sp = PointLineSource(num_elements=11,
                start_position=(-128.0, 28.0, 0),
                release_time=self.release_time,
                end_position=(-129.0, 29.0, 0),
                end_release_time=self.release_time + timedelta(minutes=100))
        timestep = 100 * 60

        # the full release over one time step
        # (plus a tiny bit to get the last one)
        data_arrays = self.release_and_assert(sp, self.release_time,
                                            timestep + 1, {}, sp.num_elements)

        assert data_arrays['positions'].shape == (11, 3)
        assert np.array_equal(data_arrays['positions'][:, 0],
                              np.linspace(-128, -129, 11))
        assert np.array_equal(data_arrays['positions'][:, 1],
                              np.linspace(28, 29, 11))

        assert sp.num_released == 11

    def test_cont_line_release2(self):
        """
        testing a release that is releasing while moving over time

        Release 1/10 of particles (10 or 100) over two steps. Then release
        the remaining particles in the last step
        """
        num_elems = 100
        sp = PointLineSource(num_elems,
                start_position=(-128.0, 28.0, 0),
                release_time=self.release_time,
                end_position=(-129.0, 29.0, 0),
                end_release_time=self.release_time + timedelta(minutes=100))
        lats = np.linspace(-128, -129, 100)
        lons = np.linspace(28, 29, 100)

        # at release time with time step of 1/10 of release_time
        # 1/10th of total particles are expected to be released
        # release 10 particles over two steps. Then release remaining particles
        # over the last timestep
        timestep = 600
        data_arrays = {}
        delay_after_rel_time = [timedelta(0),
                                timedelta(seconds=timestep),
                                #timedelta(minutes=100)    # releases 0!
                                timedelta(minutes=90)]
        ts = [timestep, timestep, timestep]
        exp_elems = [10, 10, 80]

        for ix in range(len(ts)):
            data_arrays = self.release_and_assert(sp,
                            self.release_time + delay_after_rel_time[ix],
                            ts[ix], data_arrays, exp_elems[ix])
            assert np.array_equal(
                    data_arrays['positions'][:, 0], lats[:sp.num_released])
            assert np.array_equal(
                    data_arrays['positions'][:, 1], lons[:sp.num_released])

    def test_cont_line_release3(self):
        """
        testing a release that is releasing while moving over time

        making sure it's right for the full release
        - multiple elements per step
        """
        sp = PointLineSource(num_elements=50,
                start_position=(-128.0, 28.0, 0),
                release_time=self.release_time, end_position=(-129.0,
                30.0, 0), end_release_time=self.release_time
                + timedelta(minutes=50))

        # start before release
        time = self.release_time - timedelta(minutes=10)
        delta_t = timedelta(minutes=10)
        num_rel_per_step = 10
        data_arrays = {}

        # end after release - release 10 particles at every step
        while time < self.release_time + timedelta(minutes=100):
            if (time + delta_t <= self.release_time or
                sp.num_released == sp.num_elements):
                exp_num_rel = None
            else:
                exp_num_rel = num_rel_per_step

            data_arrays = self.release_and_assert(sp,
                                                  time,
                                                  delta_t.total_seconds(),
                                                  data_arrays,
                                                  exp_num_rel)
            time += delta_t

        # all particles have been released
        assert data_arrays['positions'].shape == (sp.num_elements, 3)
        assert np.array_equal(data_arrays['positions'][0],
                              sp.start_position)
        assert np.array_equal(data_arrays['positions'][-1],
                              sp.end_position)

        # check if they are all close to the same line (constant slope)
        diff = np.diff(data_arrays['positions'][:, :2], axis=0)
        assert np.alltrue(np.abs(np.diff(diff[:, 0] / diff[:, 1])) < 1e-10)

#==============================================================================
#     def test_cont_line_release4(self):
#         """
#         testing a release that is releasing while moving over time
# 
#         making sure it's right for the full release
#         - less than one elements per step
#         """
# 
#         sp = PointSourceSurfaceRelease(num_elements=10,
#                 start_position=(-128.0, 28.0, 0),
#                 release_time=self.release_time, end_position=(-129.0,
#                 31.0, 0), end_release_time=self.release_time
#                 + timedelta(minutes=50))
# 
#         # start before release
# 
#         time = self.release_time - timedelta(minutes=10)
#         delta_t = timedelta(minutes=2)
#         timestep = delta_t.total_seconds()
#         positions = np.zeros((0, 3), dtype=np.float64)
# 
#         # end after release
# 
#         while time < self.release_time + timedelta(minutes=100):
#             arrays = sp.release_elements(time, timestep)
#             if arrays is not None:
#                 positions = np.r_[positions, arrays['positions']]
#             time += delta_t
#         assert positions.shape == (10, 3)
#         assert np.array_equal(positions[0], (-128.0, 28.0, 0))
#         assert np.array_equal(positions[-1], (-129.0, 31.0, 0))
# 
#         # check for monotonic
# 
#         assert np.alltrue(np.sign(np.diff(positions[:, :2], axis=0))
#                           == (-1, 1))
# 
#         # check if they are all close to the same line (constant slope)
# 
#         diff = np.diff(positions[:, :2], axis=0)
#         assert np.alltrue(np.abs(np.diff(diff[:, 0] / diff[:, 1]))
#                           < 1e-10)
# 
#     positions = [((128.0, 2.0, 0.), (128.0, -2.0, 0.)), ((128.0, 2.0,
#                  0.), (128.0, 4.0, 0.)), ((128.0, 2.0, 0.), (125.0,
#                  2.0, 0.)), ((-128.0, 2.0, 0.), (-120.0, 2.0, 0.)),
#                  ((-128.0, 2.0, 0.), (-120.0, 2.01, 0.))]  # south
#                                                            # north
#                                                            # west
#                                                            # east
#                                                            # almost east
# 
#     @pytest.mark.parametrize(('start_position', 'end_position'),
#                              positions)
#     def test_south_line(self, start_position, end_position):
#         """
#         testing a line release to the north
#         making sure it's right for the full release
#         - multiple elements per step
#         """
# 
#         sp = PointSourceSurfaceRelease(num_elements=50,
#                 start_position=start_position,
#                 release_time=self.release_time,
#                 end_position=end_position,
#                 end_release_time=self.release_time
#                 + timedelta(minutes=50))
# 
#         # start before release
# 
#         time = self.release_time - timedelta(minutes=10)
#         delta_t = timedelta(minutes=10)
#         timestep = delta_t.total_seconds()
#         positions = np.zeros((0, 3), dtype=np.float64)
# 
#         # end after release
# 
#         while time < self.release_time + timedelta(minutes=100):
#             arrays = sp.release_elements(time, timestep)
#             if arrays is not None:
#                 positions = np.r_[positions, arrays['positions']]
#             time += delta_t
#         assert positions.shape == (50, 3)
#         assert np.array_equal(positions[0], start_position)
#         assert np.allclose(positions[-1], end_position)
# 
#         # #check if they are all close to the same line (constant slope)
# 
#         diff = np.diff(positions[:, :2], axis=0)
#         if start_position[1] == end_position[1]:
# 
#             # horizontal line -- infinite slope
# 
#             assert np.alltrue(diff[:, 1] == 0)
#         else:
#             assert np.alltrue(np.abs(np.diff(diff[:, 0] / diff[:, 1]))
#                               < 1e-8)
# 
#     def test_cont_not_valid_times(self):
#         with pytest.raises(ValueError):
#             sp = PointSourceSurfaceRelease(num_elements=100,
#                     start_position=self.start_position,
#                     release_time=self.release_time,
#                     end_release_time=self.release_time
#                     - timedelta(seconds=1))
# 
#     def test_end_position(self):
#         """
#         if end_position = None, then automatically set it to start_position
#         """
# 
#         sp = PointSourceSurfaceRelease(num_elements=self.num_elements,
#                 start_position=self.start_position,
#                 release_time=self.release_time)
# 
#         sp.start_position = (0, 0, 0)
#         assert np.any(sp.start_position != sp.end_position)
# 
#         sp.end_position = None
#         assert np.all(sp.start_position == sp.end_position)
# 
#     def test_end_release_time(self):
#         """
#         if end_release_time = None, then automatically set it to release_time
#         """
# 
#         sp = PointSourceSurfaceRelease(num_elements=self.num_elements,
#                 start_position=self.start_position,
#                 release_time=self.release_time)
# 
#         sp.release_time = self.release_time + timedelta(hours=20)
#         assert sp.release_time != sp.end_release_time
# 
#         sp.end_release_time = None
#         assert sp.release_time == sp.end_release_time
# 
# 
# num_elements = (
#     (998, ),
#     (100, ),
#     (11, ),
#     (10, ),
#     (5, ),
#     (4, ),
#     (3, ),
#     (2, ),
#     )
# 
# 
# @pytest.mark.parametrize(('num_elements', ), num_elements)
# def test_single_line(num_elements):
#     """
#     various numbers of elemenets over ten time steps, so release
#     is less than one, one and more than one per time step.
#     """
# 
#     print 'using num_elements:', num_elements
#     start_time = datetime(2012, 1, 1)
#     end_time = start_time + timedelta(seconds=100)
#     time_step = timedelta(seconds=10)
#     start_pos = np.array((0., 0., 0.))
#     end_pos = np.array((1.0, 2.0, 0.))
# 
#     sp = PointSourceSurfaceRelease(num_elements=num_elements,
#                                    start_position=start_pos,
#                                    release_time=start_time,
#                                    end_position=end_pos,
#                                    end_release_time=end_time)
# 
#     time = start_time
#     positions = []
#     while time <= end_time + time_step * 2:
#         data = sp.release_elements(time, time_step.total_seconds())
#         if data is not None:
#             positions.extend(data['positions'])
#         time += time_step
# 
#     positions = np.array(positions)
# 
#     assert len(positions) == num_elements
#     assert np.allclose(positions[0], start_pos)
#     assert np.allclose(positions[-1], end_pos)
#     assert np.allclose(positions[:, 0], np.linspace(start_pos[0],
#                        end_pos[0], num_elements))
# 
# 
# def test_line_release_with_one_element():
#     """
#     one element with a line release
#     -- doesn't really make sense, but it shouldn't crash
#     """
# 
#     start_time = datetime(2012, 1, 1)
#     end_time = start_time + timedelta(seconds=100)
#     time_step = timedelta(seconds=10)
#     start_pos = np.array((0., 0., 0.))
#     end_pos = np.array((1.0, 2.0, 0.))
# 
#     sp = PointSourceSurfaceRelease(num_elements=1,
#                                    start_position=start_pos,
#                                    release_time=start_time,
#                                    end_position=end_pos,
#                                    end_release_time=end_time)
# 
#     time = start_time - time_step
#     assert sp.release_elements(time, time_step.total_seconds()) is None
#     time += time_step
#     data = sp.release_elements(time, time_step.total_seconds())
# 
#     assert np.array_equal(data['positions'], [start_pos])
# 
# 
# def test_line_release_with_big_timestep():
#     """
#     a line release: where the timestpe spans before to after the release time
#     """
# 
#     start_time = datetime(2012, 1, 1)
#     end_time = start_time + timedelta(seconds=100)
#     time_step = timedelta(seconds=300)
#     start_pos = np.array((0., 0., 0.))
#     end_pos = np.array((1.0, 2.0, 0.))
# 
#     sp = PointSourceSurfaceRelease(num_elements=10,
#                                    start_position=start_pos,
#                                    release_time=start_time,
#                                    end_position=end_pos,
#                                    end_release_time=end_time)
# 
#     data = sp.release_elements(start_time - timedelta(seconds=100),
#                                time_step.total_seconds())
# 
#     assert np.array_equal(data['positions'][:, 0], np.linspace(0., 1.0,
#                           10))
#     assert np.array_equal(data['positions'][:, 1], np.linspace(0., 2.0,
#                           10))
# 
# 
# def test_SpatialRelease():
#     """
#     see if the right arrays get created
#     """
# 
#     start_positions = ((0., 0., 0.), (28.0, -75.0, 0.), (-15, 12, 4.0),
#                        (80, -80, 100.0))
# 
#     release_time = datetime(2012, 1, 1, 1)
#     sp = SpatialRelease(start_positions, release_time,
#                         windage_range=(0.01, 0.04), windage_persist=900)
#     data = sp.release_elements(release_time, time_step=600)
# 
#     assert data['positions'].shape == (4, 3)
# 
# 
# def test_SpatialRelease2():
#     """
#     make sure they don't release elements twice
#     """
# 
#     start_positions = ((0., 0., 0.), (28.0, -75.0, 0.), (-15, 12, 4.0),
#                        (80, -80, 100.0))
# 
#     release_time = datetime(2012, 1, 1, 1)
# 
#     sp = SpatialRelease(start_positions, release_time,
#                         windage_range=(0.01, 0.04), windage_persist=900)
# 
#     data = sp.release_elements(release_time, time_step=600)
# 
#     assert data['positions'].shape == (4, 3)
#     data = sp.release_elements(release_time + timedelta(hours=1),
#                                time_step=600)
# 
# 
# def test_SpatialRelease3():
#     """
#     make sure it doesn't release if the first call is too late
#     """
# 
#     start_positions = ((0., 0., 0.), (28.0, -75.0, 0.), (-15, 12, 4.0),
#                        (80, -80, 100.0))
# 
#     release_time = datetime(2012, 1, 1, 1)
# 
#     sp = SpatialRelease(start_positions, release_time,
#                         windage_range=(0.01, 0.04), windage_persist=900)
# 
#     # first call after release_time
# 
#     data = sp.release_elements(release_time + timedelta(seconds=1),
#                                time_step=600)
#     assert data is None
# 
#     # still shouldn't release
# 
#     data = sp.release_elements(release_time + timedelta(hours=1),
#                                time_step=600)
#     assert data is None
# 
#     sp.rewind()
# 
#     # now it should:
# 
#     data = sp.release_elements(release_time, time_step=600)
#     assert data['positions'].shape == (4, 3)
# 
# 
# def test_PointSourceSurfaceRelease_new_from_dict():
#     """
#     test to_dict function for Wind object
#     create a new wind object and make sure it has same properties
#     """
# 
#     spill = PointSourceSurfaceRelease(num_elements=1000,
#             start_position=(144.664166, 13.441944, 0.),
#             release_time=datetime(2013, 2, 13, 9, 0),
#             end_release_time=datetime(2013, 2, 13, 9, 0)
#             + timedelta(hours=6))
# 
#     sp_state = spill.to_dict('create')
#     print sp_state
# 
#     # this does not catch two objects with same ID
# 
#     sp2 = PointSourceSurfaceRelease.new_from_dict(sp_state)
# 
#     assert spill == sp2
# 
# 
# def test_PointSourceSurfaceRelease_from_dict():
#     """
#     test from_dict function for Wind object
#     update existing wind object from_dict
#     """
# 
#     spill = PointSourceSurfaceRelease(num_elements=1000,
#             start_position=(144.664166, 13.441944, 0.),
#             release_time=datetime(2013, 2, 13, 9, 0),
#             end_release_time=datetime(2013, 2, 13, 9, 0)
#             + timedelta(hours=6))
# 
#     sp_dict = spill.to_dict()
#     sp_dict['windage_range'] = [.02, .03]
#     spill.from_dict(sp_dict)
# 
#     for key in sp_dict.keys():
#         if isinstance(spill.__getattribute__(key), np.ndarray):
#             np.testing.assert_equal(sp_dict.__getitem__(key),
#                                     spill.__getattribute__(key))
#         else:
#             assert spill.__getattribute__(key) \
#                 == sp_dict.__getitem__(key)
#==============================================================================

if __name__ == '__main__':

    # TC = Test_PointSourceSurfaceRelease()
    # TC.test_model_skips_over_release_time()

    test_SpatialRelease3()
