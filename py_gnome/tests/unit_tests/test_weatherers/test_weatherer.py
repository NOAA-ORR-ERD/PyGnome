#!/usr/bin/env python

'''
Unit tests for the Weatherer classes
'''

from datetime import datetime, timedelta

import pytest

from ..conftest import sample_sc_release

import numpy
np = numpy

from gnome.utilities.inf_datetime import InfDateTime

from gnome.array_types import mass, rise_vel, mass_components
from gnome.spill.elements import (ElementType,
                                  InitMassFromSpillAmount,
                                  InitRiseVelFromDist,
                                  InitArraysFromOilProps)

from gnome.weatherers import Weatherer, HalfLifeWeatherer

rel_time = datetime(2012, 8, 20, 13)  # yyyy/month/day/hr/min/sec
arr_types = {'mass': mass,
             'rise_vel': rise_vel,
             'mass_components': mass_components,
             }
initializers = [InitMassFromSpillAmount(),
                InitRiseVelFromDist(),
                InitArraysFromOilProps(),
                ]
oil = 'ALAMO'
sc = sample_sc_release(5, (3., 6., 0.),
                       rel_time,
                       uncertain=False,
                       arr_types=arr_types,
                       element_type=ElementType(initializers, substance=oil))
u_sc = sample_sc_release(5, (3., 6., 0.),
                         rel_time,
                         uncertain=True,
                         arr_types=arr_types,
                         element_type=ElementType(initializers, substance=oil))
secs_in_minute = 60


class TestWeatherer:
    def test_init(self):
        weatherer = Weatherer()

        print weatherer
        assert weatherer.on
        assert weatherer.active
        assert weatherer.active_start == InfDateTime('-inf')
        assert weatherer.active_stop == InfDateTime('inf')
        assert weatherer.array_types == {'mass_components': mass_components}

    @pytest.mark.parametrize("test_sc", [sc, u_sc])
    def test_one_weather(self, test_sc):
        '''
        calls one weathering step and checks that we decayed at the expected
        rate. Needs more tests with varying half_lives
        '''
        print '\nsc["mass"]:\n', test_sc['mass']

        orig_mc = np.copy(test_sc['mass_components'])

        model_time = rel_time
        time_step = 15 * secs_in_minute

        hl = tuple([time_step] * test_sc['mass_components'].shape[1])
        weatherer = HalfLifeWeatherer(half_lives=hl)
        weatherer.prepare_for_model_run(test_sc)
        weatherer.prepare_for_model_step(test_sc, time_step, model_time)

        weatherer.weather_elements(test_sc, time_step, model_time)
        weatherer.model_step_is_done()

        print '\nsc["mass"]:\n', test_sc['mass']
        assert np.allclose(0.5 * orig_mc.sum(1), test_sc['mass'])
        assert np.allclose(0.5 * orig_mc, test_sc['mass_components'])
