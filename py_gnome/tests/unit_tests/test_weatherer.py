#!/usr/bin/env python

'''
Unit tests for the Weatherer classes
'''

from datetime import datetime, timedelta

import pytest
from pytest import raises
from conftest import sample_sc_release

import numpy
np = numpy

from gnome.utilities.inf_datetime import InfDateTime
from gnome.utilities.weathering import weather_curve

from gnome.array_types import rise_vel
from gnome.elements import ElementType, InitRiseVelFromDist

from gnome.weatherers.core import Weatherer

rel_time = datetime(2012, 8, 20, 13)  # yyyy/month/day/hr/min/sec
sc = sample_sc_release(5, (3., 6., 0.),
                       rel_time,
                       uncertain=False,
                       arr_types={'rise_vel': rise_vel},
                       element_type=ElementType({'rise_vel':
                                                 InitRiseVelFromDist()}))
u_sc = sample_sc_release(5, (3., 6., 0.),
                         rel_time,
                         uncertain=True,
                         arr_types={'rise_vel': rise_vel},
                         element_type=ElementType({'rise_vel':
                                                 InitRiseVelFromDist()}))
secs_in_minute = 60


class TestWeatherer:
    wc = weather_curve(((0.333333, 15 * secs_in_minute),
                        (0.333333, 15 * secs_in_minute),
                        (0.333334, 15 * secs_in_minute)),
                       )

    def test_init_exception(self):
        with raises(TypeError):
            Weatherer()

    def test_init(self):
        weatherer = Weatherer(weathering=self.wc)

        print weatherer
        assert weatherer.on == True
        assert weatherer.active == True
        assert weatherer.active_start == InfDateTime('-inf')
        assert weatherer.active_stop == InfDateTime('inf')
        assert weatherer.array_types == {}

    @pytest.mark.parametrize("test_sc", [sc, u_sc])
    def test_one_move(self, test_sc):
        '''
           calls one movement step and checks that we decayed at the expected
           rate.
        '''
        weatherer = Weatherer(weathering=self.wc)

        # TODO: I can't really find a spill that releases LEs with
        #       a non-zero mass.
        #       (Note: The VerticalPlumeSource could be modified to do this
        #              pretty easily)
        #       For now, we just set the mass of our LEs to a known.
        test_sc['mass'][:] = 100.
        print '\nsc["mass"]:\n', test_sc['mass']

        model_time = rel_time
        time_step = 15 * secs_in_minute

        weatherer.prepare_for_model_run()
        weatherer.prepare_for_model_step(test_sc, time_step, model_time)

        decayed_mass = weatherer.get_move(test_sc, time_step, model_time)
        weatherer.model_step_is_done()

        print '\ndecayed_mass:\n', decayed_mass
        assert np.allclose(decayed_mass, 50.)

    @pytest.mark.parametrize("test_sc", [sc, u_sc])
    def test_out_of_bounds_model_time(self, test_sc):
        '''
           Here we test the conditions where the model_time
           is outside the range of the weatherer's active
           start and stop times.
           1: (model_time >= active_stop)
              So basically the time duration for our calculation is zero
              since the time_step will always be greater than model_time.
              And there should be no decay.
           2: (model_time < active_start) and (time_step <= active_start)
              So basically the time duration for our calculation is zero
              and there should be no decay.
           3: (model_time < active_start) and (time_step > active_start)
              So basically the time duration for our calculation will be
              (active_start --> time_step)
              The decay will be calculated for this partial time duration.
        '''
        # rel_time = datetime(2012, 8, 20, 13)
        stop_time = rel_time + timedelta(hours=1)

        test_sc['mass'][:] = 100.
        print '\nsc["mass"]:\n', test_sc['mass']

        # setup test case 1
        model_time = stop_time
        time_step = 15 * secs_in_minute

        weatherer = Weatherer(weathering=self.wc,
                              active_start=rel_time, active_stop=stop_time)

        weatherer.prepare_for_model_run()

        weatherer.prepare_for_model_step(test_sc, time_step, model_time)
        decayed_mass = weatherer.get_move(test_sc, time_step, model_time)
        weatherer.model_step_is_done()

        print '\ndecayed_mass:\n', decayed_mass
        assert np.allclose(decayed_mass, 100.)

        # setup test case 2
        model_time = rel_time - timedelta(minutes=15)
        time_step = 15 * secs_in_minute

        weatherer.prepare_for_model_step(test_sc, time_step, model_time)
        decayed_mass = weatherer.get_move(test_sc, time_step, model_time)
        weatherer.model_step_is_done()

        print '\ndecayed_mass:\n', decayed_mass
        assert np.allclose(decayed_mass, 100.)

        # setup test case 3
        model_time = rel_time - timedelta(minutes=15)
        time_step = 30 * secs_in_minute

        weatherer.prepare_for_model_step(test_sc, time_step, model_time)
        decayed_mass = weatherer.get_move(test_sc, time_step, model_time)
        weatherer.model_step_is_done()

        print '\ndecayed_mass:\n', decayed_mass
        assert np.allclose(decayed_mass, 50.)

    @pytest.mark.parametrize("test_sc", [sc, u_sc])
    def test_out_of_bounds_time_step(self, test_sc):
        '''
           Here we test the conditions where the time_step
           is outside the range of the weatherer's active
           start and stop times.
           4: (model_time < active_stop) and (time_step > active_stop)
              So basically the time duration for our calculation will be
              (model_time --> active_stop)
              The decay will be calculated for this partial time duration.
        '''
        # rel_time = datetime(2012, 8, 20, 13)
        stop_time = rel_time + timedelta(hours=1)

        test_sc['mass'][:] = 100.
        print '\nsc["mass"]:\n', test_sc['mass']

        # setup test case 4
        model_time = stop_time - timedelta(minutes=15)
        time_step = 30 * secs_in_minute

        weatherer = Weatherer(weathering=self.wc,
                              active_start=rel_time, active_stop=stop_time)

        weatherer.prepare_for_model_run()

        weatherer.prepare_for_model_step(test_sc, time_step, model_time)
        decayed_mass = weatherer.get_move(test_sc, time_step, model_time)
        weatherer.model_step_is_done()

        print '\ndecayed_mass:\n', decayed_mass
        assert np.allclose(decayed_mass, 50.)

    @pytest.mark.parametrize("test_sc", [sc, u_sc])
    def test_model_time_range_surrounds_active_range(self, test_sc):
        '''
           Here we test the condition where the model_time and time_step
           specify a time range that completely surrounds the range of the
           weatherer's active start and stop times.
           5: (model_time < active_start) and (time_step > active_stop)
              So basically the time duration for our calculation will be
              (active_start --> active_stop)
              The decay will be calculated for this partial time duration.
        '''
        # rel_time = datetime(2012, 8, 20, 13)
        stop_time = rel_time + timedelta(minutes=15)

        test_sc['mass'][:] = 100.
        print '\nsc["mass"]:\n', test_sc['mass']

        # setup test case 5
        model_time = rel_time - timedelta(minutes=15)
        time_step = 45 * secs_in_minute

        weatherer = Weatherer(weathering=self.wc,
                              active_start=rel_time, active_stop=stop_time)

        weatherer.prepare_for_model_run()

        weatherer.prepare_for_model_step(test_sc, time_step, model_time)
        decayed_mass = weatherer.get_move(test_sc, time_step, model_time)
        weatherer.model_step_is_done()

        print '\ndecayed_mass:\n', decayed_mass
        assert np.allclose(decayed_mass, 50.)
