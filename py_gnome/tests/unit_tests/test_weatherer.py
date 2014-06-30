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

from gnome.array_types import mass, rise_vel, mass_components, half_lives
from gnome.spill.elements import (ElementType,
                            InitMassFromTotalMass,
                            InitRiseVelFromDist,
                            InitMassComponentsFromOilProps,
                            InitHalfLivesFromOilProps
                            )

from gnome.weatherers import Weatherer

rel_time = datetime(2012, 8, 20, 13)  # yyyy/month/day/hr/min/sec
arr_types = {'mass': mass,
             'rise_vel': rise_vel,
             'mass_components': mass_components,
             'half_lives': half_lives
             }
initializers = {'mass': InitMassFromTotalMass(),
                'rise_vel': InitRiseVelFromDist(),
                'mass_components': InitMassComponentsFromOilProps(),
                'half_lives': InitHalfLivesFromOilProps()
                }
sc = sample_sc_release(5, (3., 6., 0.),
                       rel_time,
                       uncertain=False,
                       arr_types=arr_types,
                       element_type=ElementType(initializers))
u_sc = sample_sc_release(5, (3., 6., 0.),
                         rel_time,
                         uncertain=True,
                         arr_types=arr_types,
                         element_type=ElementType(initializers))
secs_in_minute = 60


class TestWeatherer:
    def test_init(self):
        weatherer = Weatherer()

        print weatherer
        assert weatherer.on == True
        assert weatherer.active == True
        assert weatherer.active_start == InfDateTime('-inf')
        assert weatherer.active_stop == InfDateTime('inf')
        assert weatherer.array_types == {'mass_components': mass_components,
                                         'half_lives': half_lives}

    @pytest.mark.parametrize("test_sc", [sc, u_sc])
    def test_one_move(self, test_sc):
        '''
           calls one get_move step and checks that we decayed at the expected
           rate.
        '''
        weatherer = Weatherer()

        print '\nsc["mass"]:\n', test_sc['mass']

        model_time = rel_time
        time_step = 15 * secs_in_minute

        weatherer.prepare_for_model_run()
        weatherer.prepare_for_model_step(test_sc, time_step, model_time)

        decayed_mass = weatherer.get_move(test_sc, time_step, model_time)
        weatherer.model_step_is_done()

        print '\ndecayed_mass:\n', decayed_mass
        assert np.allclose(decayed_mass.sum(1), test_sc['mass'] * .5)

    @pytest.mark.parametrize("test_sc", [sc, u_sc])
    def test_one_weather(self, test_sc):
        '''
           calls one weathering step and checks that we decayed at the expected
           rate.
        '''
        saved_mass = np.copy(test_sc['mass'])
        saved_components = np.copy(test_sc['mass_components'])

        weatherer = Weatherer()

        print '\nsc["mass"]:\n', test_sc['mass']

        model_time = rel_time
        time_step = 15 * secs_in_minute

        weatherer.prepare_for_model_run()
        weatherer.prepare_for_model_step(test_sc, time_step, model_time)

        weatherer.weather_elements(test_sc, time_step, model_time)
        weatherer.model_step_is_done()

        print '\nsc["mass"]:\n', test_sc['mass']
        assert np.allclose(test_sc['mass'], 0.5 * saved_mass)
        assert np.allclose(test_sc['mass_components'].sum(1),
            0.5 * saved_components.sum(1))

        test_sc['mass'] = saved_mass
        test_sc['mass_components'] = saved_components

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

        print '\nsc["mass"]:\n', test_sc['mass']

        # setup test case 1
        model_time = stop_time
        time_step = 15 * secs_in_minute

        weatherer = Weatherer(active_start=rel_time, active_stop=stop_time)

        weatherer.prepare_for_model_run()

        weatherer.prepare_for_model_step(test_sc, time_step, model_time)
        decayed_mass = weatherer.get_move(test_sc, time_step, model_time)
        weatherer.model_step_is_done()

        print '\ndecayed_mass:\n', decayed_mass
        assert np.allclose(decayed_mass.sum(1), 1. * test_sc['mass'])

        # setup test case 2
        model_time = rel_time - timedelta(minutes=15)
        time_step = 15 * secs_in_minute

        weatherer.prepare_for_model_step(test_sc, time_step, model_time)
        decayed_mass = weatherer.get_move(test_sc, time_step, model_time)
        weatherer.model_step_is_done()

        print '\ndecayed_mass:\n', decayed_mass
        assert np.allclose(decayed_mass.sum(1), 1. * test_sc['mass'])

        # setup test case 3
        model_time = rel_time - timedelta(minutes=15)
        time_step = 30 * secs_in_minute

        weatherer.prepare_for_model_step(test_sc, time_step, model_time)
        decayed_mass = weatherer.get_move(test_sc, time_step, model_time)
        weatherer.model_step_is_done()

        print '\ndecayed_mass:\n', decayed_mass
        assert np.allclose(decayed_mass.sum(1), 0.5 * test_sc['mass'])

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

        print '\nsc["mass"]:\n', test_sc['mass']

        # setup test case 4
        model_time = stop_time - timedelta(minutes=15)
        time_step = 30 * secs_in_minute

        weatherer = Weatherer(active_start=rel_time, active_stop=stop_time)

        weatherer.prepare_for_model_run()

        weatherer.prepare_for_model_step(test_sc, time_step, model_time)
        decayed_mass = weatherer.get_move(test_sc, time_step, model_time)
        weatherer.model_step_is_done()

        print '\ndecayed_mass:\n', decayed_mass
        assert np.allclose(decayed_mass.sum(1), 0.5 * test_sc['mass'])

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
        stop_time = rel_time + timedelta(minutes=15)

        print '\nsc["mass"]:\n', test_sc['mass']

        # setup test case 5
        model_time = rel_time - timedelta(minutes=15)
        time_step = 45 * secs_in_minute

        weatherer = Weatherer(active_start=rel_time, active_stop=stop_time)

        weatherer.prepare_for_model_run()

        weatherer.prepare_for_model_step(test_sc, time_step, model_time)
        decayed_mass = weatherer.get_move(test_sc, time_step, model_time)
        weatherer.model_step_is_done()

        print '\ndecayed_mass:\n', decayed_mass
        assert np.allclose(decayed_mass.sum(1), 0.5 * test_sc['mass'])
