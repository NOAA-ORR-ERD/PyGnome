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
              (active_start + timedelta(hours=1.25), 4),
              (active_start + timedelta(hours=4), 2),
              (active_start + timedelta(hours=7), 8)]


@mark.parametrize(("timeseries", "units"),
                  [(timeseries, 'kg'),
                   (np.asarray(timeseries, dtype=datetime_value_1d), 'l')])
def test_init(timeseries, units):
    b = Beaching(active_start, units, timeseries, name='test_beaching')

    assert b.name == 'test_beaching'
    assert b.units == units
    assert b.active_start == active_start
    assert b.timeseries[-1][0] == b.active_stop

    ts = np.asarray(timeseries, dtype=datetime_value_1d)

    assert all(b.timeseries['time'] == ts['time'])
    assert all(b.timeseries['value'] == ts['value'])


class TestBeaching(ObjForTests):
    (sc, intrinsic) = ObjForTests.mk_test_objs()
    sc.spills[0].set('release_time', active_start)
    b = Beaching(active_start, 'l', timeseries, name='test_beaching')
    substance = sc.spills[0].get('substance')

    @mark.parametrize(("model_time", "active"),
                      [(active_start, True),
                       (timeseries[-1][0], False)])
    def test_prepare_for_model_step(self, model_time, active):
        self.reset_and_release()
        self.b.prepare_for_model_step(self.sc, 1800, model_time)
        assert self.b.active is active

    @mark.parametrize(("model_time", "dt", "rate_idx", "rate_dt"),
                      [(active_start, 1800, [0], [1800]),
                       (active_start + timedelta(hours=.75), 2700,
                        [0, 1, 2], [900, 900, 900]),
                       (active_start + timedelta(hours=3.5), 2700,
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
        model_time = self.sc.spills[0].get('release_time')

        self.reset_test_objs()
        self.b.prepare_for_model_run(self.sc)

        assert self.sc.weathering_data['manual_beached'] == 0.0

        while (model_time <
               self.b.active_stop + timedelta(seconds=time_step)):

            amt = self.sc.weathering_data['manual_beached']

            num_rel = self.sc.release_elements(time_step, model_time)
            self.intrinsic.update(num_rel, self.sc, time_step)
            self.b.prepare_for_model_step(self.sc, time_step, model_time)

            self.b.weather_elements(self.sc, time_step, model_time)

            if not self.b.active:
                assert self.sc.weathering_data['manual_beached'] == amt
            else:
                # check total amount removed at each timestep
                assert self.sc.weathering_data['manual_beached'] > amt

            self.b.model_step_is_done(self.sc)
            self.sc.model_step_is_done()
            # model would do the following
            self.sc['age'][:] = self.sc['age'][:] + time_step
            model_time += timedelta(seconds=time_step)

            # check - useful for debugging issues with recursion
            assert np.isclose(total, self.sc.weathering_data['manual_beached']
                              + self.sc['mass'].sum())

        # following should finally hold true for entire run
        assert np.allclose(total, self.sc.weathering_data['manual_beached'] +
                           self.sc['mass'].sum(), atol=1e-6)

        # volume units
        total_mass = self.b._get_mass(self.substance,
                                      self.b.timeseries['value'].sum(),
                                      self.b.units)

        assert np.isclose(self.sc.weathering_data['manual_beached'],
                          total_mass)

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
