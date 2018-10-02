"""
tests for classes in release module
Tests for release are in test_spill.py and test_release.py
test_spill.py was getting too big so moved the tests that do not use a Spill
object here - more comprehensive tests of release objects within a Spill are
in test_spill.py
"""

import os
from datetime import datetime, timedelta

import pytest

import numpy as np

from gnome.spill import (Release,
                         PointLineRelease,
                         ContinuousRelease,
                         GridRelease,
                         Spill)
from gnome.spill.release import release_from_splot_data
from gnome.spill.le import LEData


def test_init():
    Release(release_time=datetime.now())


class TestRelease:
    rel_time = datetime.now().replace(microsecond=0)

    def test_init(self):
        rel = Release(self.rel_time, 0)
        assert rel.num_elements == 0
        assert rel.release_time == self.rel_time
        assert rel.start_time_invalid is None
        assert rel.release_duration == 0


# def test_grid_release():
#     """
#     fixme: test something more here? like actually have the release do its thing?
#     """
#     bounds = ((0, 10), (2, 12))
#     release = GridRelease(datetime.now(), bounds, 3)
#
#     assert np.array_equal(release.start_position, [[0., 10., 0.],
#                                                    [1., 10., 0.],
#                                                    [2., 10., 0.],
#                                                    [0., 11., 0.],
#                                                    [1., 11., 0.],
#                                                    [2., 11., 0.],
#                                                    [0., 12., 0.],
#                                                    [1., 12., 0.],
#                                                    [2., 12., 0.]])

# todo: add other release to this test - need schemas for all

rel_time = datetime(2012, 8, 20, 13)
rel_type = [PointLineRelease(release_time=rel_time,
                             num_elements=5,
                             start_position=(0, 0, 0)),
            PointLineRelease(release_time=rel_time,
                             num_per_timestep=5,
                             start_position=(0, 0, 0)),
            ContinuousRelease(initial_elements=5,
                              release_time=rel_time,
                              num_per_timestep=5,
                              start_position=(0, 0, 0))]
# SpatialRelease(rel_time, np.zeros((4, 3), dtype=np.float64))]


@pytest.mark.parametrize("rel_type", rel_type)
def test_release_serialization_deserialization(rel_type):
    '''
    test for a mid run state.
    'save' preserves midrun parameters
    'webapi' does not
    '''
    cls = rel_type.__class__
    n_rel = cls.deserialize(rel_type.serialize())
    assert n_rel == rel_type


class TestPointLineRelease:
    rel_time = datetime(2014, 1, 1, 0, 0)
    pos = (0, 10, 20)

    def test_LE_timestep_ratio(self):
        r1 = PointLineRelease(release_time=self.rel_time,
                              start_position=self.pos,
                              end_position= (10, 20, 40),
                              end_release_time=self.rel_time + timedelta(seconds=1000)*10,
                              num_elements=1000)
        #timestep of 10 seconds. 10,000 second release, min 1000 elements exactly
        assert r1.LE_timestep_ratio(10) == 1
        assert r1.LE_timestep_ratio(20) == 2

    def test_release_timeseries(self):
        r1 = PointLineRelease(release_time=self.rel_time,
                              start_position=self.pos,
                              end_position= (10, 20, 30),
                              end_release_time=self.rel_time + timedelta(seconds=900)*10,
                              num_elements=1000)
        r1.prepare_for_model_run(900, 5000)
        assert len(r1._release_ts.data) == 11
        assert r1._release_ts.at(None, r1.release_time) == 0
        assert r1._release_ts.at(None, r1.end_release_time) == 1000
        assert np.all(r1._release_ts.data == np.linspace(0,1000, len(r1._release_ts.data)))
        assert r1._mass_per_le == 5
        assert len(r1._pos_ts.time) == 11
        assert np.all(r1._pos_ts.at(None, r1.release_time + timedelta(seconds=900)*5) == np.array([(5.,15.,25.)]))

        r1.rewind()
        r1.prepare_for_model_run(450,2500)
        assert len(r1._release_ts.data) == 21
        assert r1._release_ts.at(None, r1.release_time) == 0
        assert r1._release_ts.at(None, r1.end_release_time) == 1000
        assert np.all(r1._release_ts.data == np.linspace(0,1000, len(r1._release_ts.data)))
        assert r1._mass_per_le == 2.5
        assert len(r1._pos_ts.time) == 21
        assert np.all(r1._pos_ts.at(None, r1.release_time + timedelta(seconds=450)*10) == np.array([(5.,15.,25.)]))

        #No end_release time. Timeseries must be 2 entries, 1 second apart
        r2 = PointLineRelease(release_time=self.rel_time,
                              start_position=self.pos,
                              end_position= (10, 20, 30),
                              num_elements=1000)
        r2.prepare_for_model_run(900, 5000)
        assert len(r2._release_ts.data) == 2
        assert r2._release_ts.at(None, r2.release_time) == 0
        assert r2._release_ts.at(None, r2.release_time - timedelta(seconds=1)) == 0
        assert r2._release_ts.at(None, r2.release_time + timedelta(seconds=1)) == 1000
        assert r2._release_ts.at(None, r2.release_time + timedelta(seconds=2)) == 1000
        assert np.all(r2._release_ts.data == np.linspace(0,1000, len(r2._release_ts.data)))
        assert r2._mass_per_le == 5
        assert len(r2._pos_ts.time) == 2
        import pdb
        pdb.set_trace()


    def test_property_num_per_timestep_elements(self):
        '''
        test either num_elements or num_per_timestep is set but not both
        also test the num_elements_to_release references correct method
        '''
        r = PointLineRelease(self.rel_time,
                             self.pos,
                             num_per_timestep=100)
        r.num_elements = 10
        assert r.num_per_timestep is None

        r.num_per_timestep = 100
        assert r.num_elements is None

    def test_num_per_timestep(self):
        'test PointLineRelease when a fixed rate per timestep is given'
        r = PointLineRelease(self.rel_time,
                             self.pos,
                             num_per_timestep=100)
        assert r.num_elements is None
        for ts in (100, 400):
            num = r.num_elements_to_release(self.rel_time, ts)
            assert num == 100

    def test_num_per_timestep_release_elements(self):
        """release elements in the context of a Spill object"""
        # fixme: A "proper" unit test shouldn't need to put it in a spill
        # todo: need a test for a line release where rate is given - to check
        # positions are being initialized correctly
        end_time = self.rel_time + timedelta(hours=1)
        release = PointLineRelease(self.rel_time,
                                   self.pos,
                                   num_per_timestep=100,
                                   end_release_time=end_time)
        s = Spill(release)
        sc = SpillContainer()
        sc.spills += s
        sc.prepare_for_model_run()
        for ix in range(5):
            time = self.rel_time + timedelta(seconds=900 * ix)
            num_les = sc.release_elements(900, time)
            if time <= s.end_release_time:
                assert num_les == 100
                assert sc.num_released == 100 + ix * 100
            else:
                assert num_les == 0

    def test_rewind(self):
        '''
        test rewind resets all parameters of interest
        '''
        r = PointLineRelease(self.rel_time,
                             self.pos,
                             end_position=(1, 2, 3),
                             num_per_timestep=100,
                             end_release_time=self.rel_time + timedelta(hours=2))
        num = r.num_elements_to_release(self.rel_time, 900)
        assert not r.start_time_invalid

        # updated only after set_newparticle_positions is called
        assert r.num_released == 0
        pos = {'positions': positions.initialize(num)}
        r.set_newparticle_positions(num,
                                    self.rel_time,
                                    900,
                                    pos)
        assert r.num_released == num
        assert r._delta_pos is not None
        assert np.any(r._next_release_pos != r.start_position)

        r.rewind()
        assert r.start_time_invalid is None
        assert r._delta_pos is None
        assert np.all(r._next_release_pos == r.start_position)

    def test__eq__(self):
        r = PointLineRelease(self.rel_time,
                             self.pos,
                             end_position=(1, 2, 3),
                             num_per_timestep=100,
                             end_release_time=self.rel_time + timedelta(hours=2))
        r1 = PointLineRelease(self.rel_time,
                              (0, 0, 0),
                              end_position=(1, 2, 3),
                              num_per_timestep=100,
                              end_release_time=self.rel_time + timedelta(hours=2))
        assert r != r1


class TestContinuousRelease:
    rel_time = datetime(2014, 1, 1, 0, 0)
    pos = (0, 1, 2)

    def test_property_num_per_timestep_elements(self):
        '''
        test either num_elements or num_per_timestep is set but not both
        also test the num_elements_to_release references correct method
        '''
        r = ContinuousRelease(self.rel_time,
                              self.pos,
                              initial_elements=100,
                              num_per_timestep=100)
        r.num_elements = 10
        assert r.num_per_timestep is None
        assert r.num_elements_to_release(self.rel_time, 900) == 110

        r.num_per_timestep = 100
        assert r.num_elements is None
        assert r.num_elements_to_release(self.rel_time, 900) == 200

    def test_num_per_timestep(self):
        'test ContinuousRelease when a fixed rate per timestep is given'
        r = ContinuousRelease(self.rel_time,
                              self.pos,
                              initial_elements=1000,
                              num_per_timestep=100)
        assert r.num_elements is None
        assert r.num_elements_to_release(self.rel_time, 100) == 1100
        r.initial_done = True
        for ts in (200, 400):
            num = r.num_elements_to_release(self.rel_time, ts)
            assert num == 100

    def test_num_per_timestep_release_elements(self):
        'release elements in the context of a spill container'
        # todo: need a test for a line release where rate is given - to check
        # positions are being initialized correctly
        end_time = self.rel_time + timedelta(hours=1)
        release = ContinuousRelease(self.rel_time,
                                    self.pos,
                                    num_per_timestep=100,
                                    initial_elements=1000,
                                    end_release_time=end_time)
        s = Spill(release)
        sc = s.data
        s.prepare_for_model_run()
        for ix in range(5):
            time = self.rel_time + timedelta(seconds=900 * ix)
            num_les = s.release_elements(900, time)
            if time <= s.end_release_time:
                if ix == 0:
                    assert num_les == 1100
                else:
                    assert num_les == 100
                assert s.num_released == 100 + ix * 100 + 1000
            else:
                assert num_les == 0

    def test_rewind(self):
        '''
        test rewind resets all parameters of interest
        '''
        r = PointLineRelease(release_time=self.rel_time,
                             start_position=self.pos,
                             end_position=(1, 2, 3),
                             num_per_timestep=100,
                             end_release_time=self.rel_time + timedelta(hours=2))
        assert not r.start_time_invalid

        # updated only after set_newparticle_positions is called

        assert r._delta_pos is not None
        assert np.any(r._next_release_pos != r.start_position)

        r.rewind()
        assert r.start_time_invalid is None
        assert r._delta_pos is None
        assert np.all(r._next_release_pos == r.start_position)

    def test__eq__(self):
        r = PointLineRelease(self.rel_time,
                             self.pos,
                             end_position=(1, 2, 3),
                             num_per_timestep=100,
                             end_release_time=self.rel_time + timedelta(hours=2))
        r1 = PointLineRelease(self.rel_time,
                              (0, 0, 0),
                              end_position=(1, 2, 3),
                              num_per_timestep=100,
                              end_release_time=self.rel_time + timedelta(hours=2))
        assert r != r1


def test_release_from_splot_data():
    '''
    test release_from_splot_data by creating file with fake data
    '''
    test_data = \
        ('-7.885776000000000E+01    4.280546000000000E+01   4.4909252E+01\n'
         '-7.885776000000000E+01    4.279556000000000E+01   4.4909252E+01\n'
         '-8.324346000000000E+01    4.196396000000001E+01   3.0546749E+01\n')
    here = os.path.dirname(__file__)
    td_file = os.path.join(here, 'test_data.txt')
    with open(td_file, 'w') as td:
        td.write(test_data)

    exp = np.asarray((44.909252, 44.909252, 30.546749),
                     dtype=int)
    exp_num_elems = exp.sum()
    rel = release_from_splot_data(datetime(2015, 1, 1), td_file)
    assert rel.num_elements == exp_num_elems
    assert len(rel.start_position) == exp_num_elems
    cumsum = np.cumsum(exp)
    for ix in xrange(len(cumsum) - 1):
        assert np.all(rel.start_position[cumsum[ix]] ==
                      rel.start_position[cumsum[ix]:cumsum[ix + 1]])
    assert np.all(rel.start_position[0] == rel.start_position[:cumsum[0]])

    os.remove(td_file)

if __name__ == "__main__":
    ct = TestContinuousRelease()
    ct.test_num_per_timestep_release_elements()
    pass
