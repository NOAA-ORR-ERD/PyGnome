'''
tests for cleanup options
'''
from datetime import datetime, timedelta

from pytest import mark
import numpy as np

from gnome.environment import Water
from gnome.weatherers.intrinsic import IntrinsicProps

from gnome.weatherers import Skimmer
from gnome.spill_container import SpillContainer
from gnome.spill import point_line_release_spill
from ..conftest import test_oil


delay = 1.
time_step = 900


class TestSkimmer:
    rel_time = datetime(2014, 1, 1, 0, 0)
    active_start = rel_time + timedelta(seconds=time_step)
    active_stop = active_start + timedelta(hours=1.)
    amount = 36000.
    units = 'kg'    # leave as SI units

    skimmer = Skimmer(amount,
                      'kg',
                      efficiency=0.3,
                      active_start=active_start,
                      active_stop=active_stop)

    def make_sc_and_release(self):
        water = Water()
        sc = SpillContainer()
        sc.spills += point_line_release_spill(10,
                                              (0, 0, 0),
                                              self.rel_time,
                                              substance=test_oil,
                                              amount=self.amount,
                                              units='kg')
        intrinsic = IntrinsicProps(water)
        intrinsic.initialize(sc)
        sc.prepare_for_model_run(intrinsic.array_types)
        num_rel = sc.release_elements(time_step, self.rel_time)
        return (sc, intrinsic, num_rel)

    def test_prepare_for_model_run(self):
        sc = self.make_sc_and_release()[0]
        self.skimmer.prepare_for_model_run(sc)
        assert sc.weathering_data['skimmed'] == 0.
        assert (self.skimmer._rate ==
                self.skimmer.amount/(self.skimmer.active_stop -
                                     self.skimmer.active_start).total_seconds())

    @mark.parametrize(("model_time", "active", "ts"),
                      [(active_start, True, time_step),
                       (active_start - timedelta(seconds=time_step/3.), True,
                        time_step*2./3.),  # between active_start - active_stop
                       (active_start + timedelta(seconds=time_step/2.), True,
                        time_step),     # between active_start - active_stop
                       (active_stop, False, None),
                       (active_stop - timedelta(seconds=time_step), True,
                        time_step),     # before active stop
                       (active_stop - timedelta(seconds=time_step/3.), True,
                        time_step/3.)   # before active stop
                       ])
    def test_prepare_for_model_step(self, model_time, active, ts):
        ''
        # NOTE: Force SpillContainer to release all particles
        (sc, intrinsic, num_rel) = self.make_sc_and_release()
        intrinsic.update(num_rel, sc)
        self.skimmer.prepare_for_model_step(sc, time_step, model_time)

        assert self.skimmer.active is active
        if active:
            assert self.skimmer._timestep == ts

    def test_weather_elements(self):
        spill_amount = self.amount
        (sc, intrinsic, num_rel) = self.make_sc_and_release()
        sc.rewind()
        sc.prepare_for_model_run(intrinsic.array_types)
        self.skimmer.prepare_for_model_run(sc)

        assert sc.weathering_data['skimmed'] == 0.0

        model_time = self.rel_time
        while (model_time <
               self.skimmer.active_stop + timedelta(seconds=2*time_step)):

            amt_skimmed = sc.weathering_data['skimmed']

            num_rel = sc.release_elements(time_step, model_time)
            intrinsic.update(num_rel, sc)
            self.skimmer.prepare_for_model_step(sc, time_step, model_time)

            self.skimmer.weather_elements(sc, time_step, model_time)

            if not self.skimmer.active:
                assert sc.weathering_data['skimmed'] == amt_skimmed
            else:
                # check total amount removed at each timestep
                assert sc.weathering_data['skimmed'] > amt_skimmed

            self.skimmer.model_step_is_done(sc)
            sc.model_step_is_done()
            # model would do the following
            sc['age'][:] = sc['age'][:] + time_step
            model_time += timedelta(seconds=time_step)

            # check - useful for debugging issues with recursion
            assert (spill_amount ==
                    sc.weathering_data['skimmed'] + sc['mass'].sum())

        # following should finally hold true for entire run
        assert np.allclose(spill_amount,
                           sc.weathering_data['skimmed'] + sc['mass'].sum(),
                           atol=1e-6)
        assert np.allclose(sc.weathering_data['skimmed']/self.skimmer.amount,
                           self.skimmer.efficiency, atol=1e-6)
