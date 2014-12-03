'''
Test some of the base class functionality independent of derived classes.
Just simpler to do the testing here
Tests the Process methods and the Mover's get_move
'''

from datetime import datetime, timedelta

import numpy as np
import pytest

from gnome import movers
from ..conftest import sample_sc_release


def test_exceptions():
    with pytest.raises(ValueError):
        now = datetime.now()
        m = movers.Mover(active_start=now, active_stop=now)
        print m


def test_properties():
    """
    Test default props
    """

    m = movers.Mover()
    assert m.on

    m.on = False
    assert not m.on


class TestActive:
    time_step = 15 * 60  # seconds
    model_time = datetime(2012, 8, 20, 13)
    sc = sample_sc_release(1, (0, 0, 0))  # no used for anything
    mv = movers.Mover()

    def test_active_default(self):
        mv = movers.Mover()
        mv.prepare_for_model_step(self.sc, self.time_step, self.model_time)
        assert mv.active  # model_time = active_start

    def test_active_start_modeltime(self):
        mv = movers.Mover(active_start=self.model_time)
        mv.prepare_for_model_step(self.sc, self.time_step, self.model_time)
        assert mv.active  # model_time = active_start

    def test_active_start_after_one_timestep(self):
        mv = movers.Mover(active_start=self.model_time
                          + timedelta(seconds=self.time_step))
        mv.prepare_for_model_step(self.sc, self.time_step, self.model_time)
        assert not mv.active  # model_time + time_step = active_start

    def test_active_start_after_half_timestep(self):
        self.mv.active_start = \
            self.model_time + timedelta(seconds=self.time_step / 2)
        self.mv.prepare_for_model_step(self.sc, self.time_step,
                                       self.model_time)
        assert self.mv.active  # model_time + time_step/2 = active_start

    # Next test just some more borderline cases that active is set correctly

    def test_active_stop_greater_than_timestep(self):
        self.mv.active_stop = \
            self.model_time + timedelta(seconds=1.5 * self.time_step)
        self.mv.prepare_for_model_step(self.sc, self.time_step,
                                       self.model_time)
        assert self.mv.active    # model_time + 1.5 * time_step = active_stop

    def test_active_stop_after_half_timestep(self):
        self.mv.active_stop = \
            self.model_time + timedelta(seconds=0.5 * self.time_step)
        self.mv.prepare_for_model_step(self.sc, self.time_step,
                                       self.model_time)
        assert self.mv.active    # model_time + 1.5 * time_step = active_stop

    def test_active_stop_less_than_half_timestep(self):
        self.mv.active_stop = \
            self.model_time + timedelta(seconds=0.25 * self.time_step)
        self.mv.prepare_for_model_step(self.sc, self.time_step,
                                       self.model_time)
        assert not self.mv.active    # current_model_time = active_stop


def test_get_move():
    '''
    assert base class get_move returns an array of nan[s]
    '''
    time_step = 15 * 60  # seconds
    model_time = datetime(2012, 8, 20, 13)
    sc = sample_sc_release(10, (0, 0, 0))  # no used for anything

    mv = movers.Mover()
    delta = mv.get_move(sc, time_step, model_time)
    assert np.all(np.isnan(delta))
