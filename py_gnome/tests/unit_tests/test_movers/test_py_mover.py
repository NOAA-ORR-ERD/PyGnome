'''
Test some of the base class functionality independent of derived classes.
Just simpler to do the testing here
Tests the Process methods and the Mover's get_move
'''
from datetime import datetime, timedelta

import numpy as np

from pytest import raises
from ..conftest import sample_sc_release

from gnome.utilities.inf_datetime import InfDateTime
from gnome.movers import PyMover


def test_exceptions():
    with raises(ValueError):
        now = datetime.now()
        _mover = PyMover(active_range=(now, now))


def test_default_properties():
    mover = PyMover()

    assert mover.name == 'PyMover'
    assert mover.on is True

    assert mover.active_range == (InfDateTime('-inf'), InfDateTime('inf'))

    assert mover.array_types == set()
    assert mover.make_default_refs is True

    assert mover.default_num_method == 'RK2'


class TestActive:
    time_step = 15 * 60  # seconds
    model_time = datetime(2012, 8, 20, 13)
    sc = sample_sc_release(1, (0, 0, 0))  # no used for anything
    mv = PyMover()

    def test_active_default(self):
        mv = PyMover()  # active range defaults to (-Inf, Inf)
        mv.prepare_for_model_step(self.sc, self.time_step, self.model_time)

        # model_time should be within range at any datetime.
        assert mv.active is True

    def test_active_start_modeltime(self):
        mv = PyMover(active_range=(self.model_time, InfDateTime('inf')))
        mv.prepare_for_model_step(self.sc, self.time_step, self.model_time)

        # model_time should be within active range
        assert mv.active is True

    def test_active_start_after_one_timestep(self):
        start_time = self.model_time + timedelta(seconds=self.time_step)

        mv = PyMover(active_range=(start_time, InfDateTime('inf')))

        mv.prepare_for_model_step(self.sc, self.time_step, self.model_time)

        # model_time + time_step should be within active range
        assert mv.active is False

    def test_active_start_after_half_timestep(self):
        self.mv.active_range = ((self.model_time +
                                 timedelta(seconds=self.time_step / 2)),
                                InfDateTime('inf'))

        self.mv.prepare_for_model_step(self.sc, self.time_step,
                                       self.model_time)

        # model_time + time_step / 2 should be within active range
        assert self.mv.active is True

    # Next test just some more borderline cases that active is set correctly
    def test_active_stop_greater_than_timestep(self):
        self.mv.active_range = (self.model_time,
                                (self.model_time +
                                 timedelta(seconds=1.5 * self.time_step)))

        self.mv.prepare_for_model_step(self.sc, self.time_step,
                                       self.model_time)

        # model_time + 1.5 * time_step should be within active range
        assert self.mv.active is True

    def test_active_stop_after_half_timestep(self):
        self.mv.active_range = (self.model_time,
                                (self.model_time +
                                 timedelta(seconds=0.5 * self.time_step)))

        self.mv.prepare_for_model_step(self.sc, self.time_step,
                                       self.model_time)

        # model_time + 1.5 * time_step should be within active range
        assert self.mv.active is True

    def test_active_stop_less_than_half_timestep(self):
        self.mv.active_range = (self.model_time,
                                (self.model_time +
                                 timedelta(seconds=0.25 * self.time_step)))

        self.mv.prepare_for_model_step(self.sc, self.time_step,
                                       self.model_time)

        # current_model_time = active_stop
        assert self.mv.active is False


def test_get_move():
    '''
    assert base class get_move returns an array of nan[s]
    '''
    time_step = 15 * 60  # seconds
    model_time = datetime(2012, 8, 20, 13)
    sc = sample_sc_release(10, (0, 0, 0))  # no used for anything

    mv = PyMover()
    delta = mv.get_move(sc, time_step, model_time)

    assert np.all(np.isnan(delta))
