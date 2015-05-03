'''
test manual_beaching
'''
from datetime import datetime, timedelta

import numpy as np

from gnome.basic_types import datetime_value_1d
from gnome.weatherers import Beaching
from .test_cleanup import ObjForTests

from pytest import mark

active_start = datetime(2015, 1, 1, 12, 0)

timeseries = [(active_start + timedelta(hours=1), 5),
              (active_start + timedelta(hours=3), 4),
              (active_start + timedelta(hours=4), 2),
              (active_start + timedelta(hours=7), 8)]


@mark.parametrize("timeseries", [timeseries,
                                 np.asarray(timeseries, dtype=datetime_value_1d)])
def test_init(timeseries):
    b = Beaching(active_start, 'l', timeseries, name='test_beaching')

    assert b.name == 'test_beaching'
    assert b.units == 'l'
    assert b.active_start == active_start
    assert b.timeseries[-1][0] == b.active_stop

    ts = np.asarray(timeseries, dtype=datetime_value_1d)

    assert all(b.timeseries['time'] == ts['time'])
    assert all(b.timeseries['value'] == ts['value'])


class TestBeaching(ObjForTests):
    (sc, intrinsic) = ObjForTests.mk_test_objs()
    b = Beaching(active_start, 'l', timeseries, name='test_beaching')

    @mark.parametrize(("model_time", "active"),
                      [(active_start, True),
                       (timeseries[-1][0], False)])
    def test_prepare_for_model_step(self, model_time, active):
        self.reset_and_release()
        self.b.prepare_for_model_step(self.sc, 900, model_time)
        assert self.b.active is active

    def test_remove_mass(self):
        pass
