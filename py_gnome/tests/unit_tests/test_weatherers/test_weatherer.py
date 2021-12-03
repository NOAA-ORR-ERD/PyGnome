#!/usr/bin/env python

'''
Unit tests for the Weatherer classes
'''

from datetime import datetime

import numpy as np

from gnome.utilities.inf_datetime import InfDateTime

from gnome.environment import Water
from gnome.spills.gnome_oil import GnomeOil

from .conftest import weathering_data_arrays, test_oil

from gnome.weatherers import (Weatherer,
                              HalfLifeWeatherer,
                              NaturalDispersion,
                              Dissolution,
                              weatherer_sort)

subs = GnomeOil(test_oil)
rel_time = datetime(2012, 8, 20, 13)  # yyyy/month/day/hr/min/sec


class TestWeatherer(object):
    def test_init(self):
        weatherer = Weatherer()

        print(weatherer)
        assert weatherer.on
        assert weatherer.active
        assert weatherer.active_range == (InfDateTime('-inf'),
                                          InfDateTime('inf'))

    def test_one_weather(self):
        '''
        calls one weathering step and checks that we decayed at the expected
        rate. Needs more tests with varying half_lives
        '''
        time_step = 15.*60
        hl = tuple([time_step] * subs.num_components)
        weatherer = HalfLifeWeatherer(half_lives=hl)
        sc = weathering_data_arrays(weatherer.array_types,
                                    Water(),
                                    time_step)[0]

        print('\nsc["mass"]:\n', sc['mass'])

        orig_mc = np.copy(sc['mass_components'])

        model_time = rel_time

        weatherer.prepare_for_model_run(sc)
        weatherer.prepare_for_model_step(sc, time_step, model_time)
        weatherer.weather_elements(sc, time_step, model_time)
        weatherer.model_step_is_done()

        print('\nsc["mass"]:\n', sc['mass'])
        assert np.allclose(0.5 * orig_mc.sum(1), sc['mass'])
        assert np.allclose(0.5 * orig_mc, sc['mass_components'])


def test_sort_order():
    assert weatherer_sort(Dissolution()) > weatherer_sort(NaturalDispersion())
