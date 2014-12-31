"""
tests for classes in release module
Tests for release are in test_spill.py and test_release.py
test_spill.py was getting too big so moved the tests that do not use a Spill
object here - more comprehensive tests of release objects within a Spill are
in test_spill.py
"""

from datetime import datetime, timedelta

import pytest

import numpy as np

from gnome.model import Model
from gnome.movers import RandomMover
from gnome.array_types import windages
from gnome.spill import (Release,
                         PointLineRelease,
                         GridRelease,
                         InitElemsFromFile,
                         Spill)
from gnome.spill_container import SpillContainer
from ..conftest import testdata


def test_init():
    Release(release_time=datetime.now())


class TestRelease:
    rel_time = datetime.now().replace(microsecond=0)

    def test_init(self):
        rel = Release(self.rel_time, 0)
        assert rel.num_elements == 0
        assert rel.release_time == self.rel_time
        assert rel.start_time_invalid is None
        assert rel.release_duration == timedelta(0)

    @pytest.mark.parametrize("curr_time", [rel_time,
                                           rel_time - timedelta(seconds=1),
                                           rel_time + timedelta(seconds=1)])
    def test_num_elements_to_release(self, curr_time):
        rel = Release(self.rel_time, 0)
        rel.num_elements_to_release(curr_time, 900)

        if curr_time <= rel.release_time:
            assert not rel.start_time_invalid
        else:
            assert rel.start_time_invalid

    def test_rewind(self):
        rel = Release(self.rel_time, 10)
        rel.num_elements_to_release(self.rel_time, 900)
        assert not rel.start_time_invalid

        # change attribute manually for test
        rel.num_released = rel.num_elements

        rel.rewind()
        assert rel.start_time_invalid is None
        assert rel.num_released == 0


def test_grid_release():
    bounds = ((0, 10), (2, 12))
    release = GridRelease(datetime.now(), bounds, 3)

    assert np.array_equal(release.start_position, [[0.,  10.,  0.],
                                                   [1.,  10.,  0.],
                                                   [2.,  10.,  0.],
                                                   [0.,  11.,  0.],
                                                   [1.,  11.,  0.],
                                                   [2.,  11.,  0.],
                                                   [0.,  12.,  0.],
                                                   [1.,  12.,  0.],
                                                   [2.,  12.,  0.]])


'''  todo: add other release to this test - need schemas for all '''

rel_time = datetime(2012, 8, 20, 13)
rel_type = [PointLineRelease(release_time=rel_time,
                             num_elements=5,
                             start_position=(0, 0, 0))]
            #SpatialRelease(rel_time, np.zeros((4, 3), dtype=np.float64))]


@pytest.mark.parametrize("rel_type", rel_type)
def test_release_serialization_deserialization(rel_type):
    '''
    test for a mid run state.
    'save' preserves midrun parameters
    'webapi' does not
    '''
    cls = rel_type.__class__
    rel_type.num_released = 100  # read only parameter for releases
    for json_ in ('save', 'webapi'):
        dict_ = cls.deserialize(rel_type.serialize(json_))
        n_rel = cls.new_from_dict(dict_)
        if json_ == 'save':
            assert n_rel == rel_type
        else:
            assert n_rel != rel_type


class TestInitElementsFromFile():
    nc_start_time = datetime(2011, 3, 11, 7, 0)
    time_step = timedelta(hours=24)

    @pytest.mark.parametrize("index", [None, 0, 2])
    def test_init(self, index):
        release = InitElemsFromFile(testdata['nc']['nc_output'], index=index)
        assert release.num_elements == 4000
        if index is None:
            # file contains initial condition plus 4 timesteps
            exp_rel_time = self.nc_start_time + self.time_step * 4
            assert np.all(release._init_data['age'] ==
                          self.time_step.total_seconds() * 4)
        else:
            exp_rel_time = self.nc_start_time + self.time_step * index

            assert np.all(release._init_data['age'] ==
                          self.time_step.total_seconds() * index)
        assert release.release_time == exp_rel_time

    def test_init_with_releasetime(self):
        'test release time gets set correctly'
        reltime = datetime(2014, 1, 1, 0, 0)
        release = InitElemsFromFile(testdata['nc']['nc_output'], reltime)
        assert release.num_elements == 4000
        assert release.release_time == reltime

    @pytest.mark.parametrize("at", [{},
                                    {'windages': windages}])
    def test_release_elements(self, at):
        'release elements in the context of a spill container'
        s = Spill(InitElemsFromFile(testdata['nc']['nc_output']))
        sc = SpillContainer()
        sc.spills += s
        sc.prepare_for_model_run(array_types=at)
        num_les = sc.release_elements(self.time_step, self.nc_start_time)
        assert sc.num_released == s.release.num_elements
        assert num_les == s.release.num_elements
        for array, val in s.release._init_data.iteritems():
            if array in sc:
                assert np.all(val == sc[array])
                assert val is not sc[array]
            else:
                assert array not in at

    def test_full_run(self):
        'just check that all data arrays work correctly'
        s = Spill(InitElemsFromFile(testdata['nc']['nc_output']))
        model = Model(start_time=s.get('release_time'),
                      time_step=self.time_step.total_seconds(),
                      duration=timedelta(days=2))
        model.spills += s
        model.movers += RandomMover()

        # setup model run
        for step in model:
            if step['step_num'] == 0:
                continue
            for sc in model.spills.items():
                for key in sc.data_arrays.keys():
                    # following keys will not change with run
                    if key in ('status_codes',
                               'mass',
                               'id',
                               'spill_num',
                               'last_water_positions'): # all water map
                        assert np.all(sc[key] == s.release._init_data[key])
                    else:
                        assert np.any(sc[key] != s.release._init_data[key])
