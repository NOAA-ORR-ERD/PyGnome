'''
test manual_beaching
'''

from datetime import datetime, timedelta
import pytest
import numpy as np

from gnome.basic_types import datetime_value_1d
from gnome.utilities.inf_datetime import InfDateTime

from gnome.weatherers import Beaching

from .test_cleanup import ObjForTests

from pytest import mark

active_range = (datetime(2015, 1, 1, 12, 0),
                InfDateTime('inf'))

timeseries = [(active_range[0] + timedelta(hours=1), 5),
              (active_range[0] + timedelta(hours=1.25), 4),
              (active_range[0] + timedelta(hours=4), 2),
              (active_range[0] + timedelta(hours=7), 8)]


@mark.parametrize(("timeseries", "units"),
                  [(timeseries, 'kg'),
                   (np.asarray(timeseries, dtype=datetime_value_1d), 'l')])
def test_init(timeseries, units):
    b = Beaching(active_range, units, timeseries, name='test_beaching')

    assert b.name == 'test_beaching'
    assert b.units == units
    assert b.active_range[0] == active_range[0]
    assert b.timeseries[-1][0] == b.active_range[1]

    ts = np.asarray(timeseries, dtype=datetime_value_1d)

    assert all(b.timeseries['time'] == ts['time'])
    assert all(b.timeseries['value'] == ts['value'])


class TestBeaching(ObjForTests):
    (sc, weatherers, environment) = ObjForTests.mk_test_objs()
    sc.spills[0].release_time = active_range[0]

    b = Beaching(active_range, 'l', timeseries, name='test_beaching',
                 water=weatherers[0].water)

    substance = sc.spills[0].substance

    @mark.parametrize(("model_time", "active"),
                      [(active_range[0], True),
                       (timeseries[-1][0], False)])
    def test_prepare_for_model_step(self, model_time, active):
        self.reset_and_release()
        self.b.prepare_for_model_step(self.sc, 1800, model_time)

        assert self.b.active is active

    @mark.parametrize(("model_time", "dt", "rate_idx", "rate_dt"),
                      [(active_range[0], 1800, [0], [1800]),
                       (active_range[0] + timedelta(hours=.75), 2700,
                        [0, 1, 2], [900, 900, 900]),
                       (active_range[0] + timedelta(hours=3.5), 2700,
                        [2, 3], [1800, 900])])
    def test_remove_mass(self, model_time, dt, rate_idx, rate_dt):
        '''
        check that _remove_mass() gives correct results for time intervals
        that cross over various timeseries indices.
        '''
        self.reset_and_release()
        self.b.prepare_for_model_step(self.sc, dt, model_time)
        if self.b.active:
            mass_to_rm = self.b._remove_mass(self.b._timestep,
                                             model_time,
                                             self.substance)
            to_rm = 0.0
            for ix, dt in zip(rate_idx, rate_dt):
                to_rm += self.b._rate[ix] * dt

            assert to_rm == mass_to_rm

    def test_weather_elements(self):
        '''
        test weather_elements removes amount of mass specified by timeseries
        '''
        time_step = 900
        total = self.sc.spills[0].get_mass()
        model_time = self.sc.spills[0].release_time

        self.prepare_test_objs()
        self.b.prepare_for_model_run(self.sc)

        print(f"{self.b.water}")
        from gnome.environment import  Water
        self.b.water = Water()

        assert self.sc.mass_balance['observed_beached'] == 0.0

        while (model_time < self.b.active_range[1] + timedelta(seconds=time_step)):
            amt = self.sc.mass_balance['observed_beached']

            self.release_elements(time_step, model_time, self.environment)
            self.step(self.b, time_step, model_time)

            if not self.b.active:
                assert self.sc.mass_balance['observed_beached'] == amt
            else:
                # check total amount removed at each timestep
                assert self.sc.mass_balance['observed_beached'] > amt
            model_time += timedelta(seconds=time_step)

            # check - useful for debugging issues with recursion
            assert np.isclose(total,
                              self.sc.mass_balance['observed_beached'] + self.sc['mass'].sum())

        # following should finally hold true for entire run
        assert np.allclose(total,
                           self.sc.mass_balance['observed_beached'] +
                           self.sc['mass'].sum(), atol=1e-6)

        # volume units
        total_mass = self.b._get_mass(self.substance,
                                      self.b.timeseries['value'].sum(),
                                      self.b.units)

        assert np.isclose(self.sc.mass_balance['observed_beached'],
                          total_mass)

    @pytest.mark.skipif(reason="serialization for weatherers overall needs review")
    def test_serialize_deserialize_update_from_dict(self):
        '''
        test serialize/deserialize works correctly for datetime_value_1d dtype
        numpy arrays
        '''
        json_ = self.b.serialize()
        json_['timeseries'][0] = (json_['timeseries'][0][0],
                                  json_['timeseries'][0][1] + 4)
        d_ = Beaching.deserialize(json_)
        self.b.update_from_dict(d_)
        assert json_ == self.b.serialize()
