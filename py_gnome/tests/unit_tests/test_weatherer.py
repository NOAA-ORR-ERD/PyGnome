#!/usr/bin/env python

'''
Unit tests for the Weatherer classes
'''

from datetime import datetime

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

secs_in_minute = 60
time_step = 15 * secs_in_minute
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
model_time = rel_time


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

        weatherer.prepare_for_model_run()
        weatherer.prepare_for_model_step(test_sc, time_step, model_time)

        decayed_mass = weatherer.get_move(test_sc, time_step, model_time)
        weatherer.model_step_is_done()

        print '\ndecayed_mass:\n', decayed_mass
        assert np.allclose(decayed_mass, 50.)
