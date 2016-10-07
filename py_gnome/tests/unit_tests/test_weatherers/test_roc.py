'''
tests for ROC
'''
from datetime import datetime, timedelta

import numpy as np
from pytest import raises, mark

import unit_conversion as us

from gnome.basic_types import oil_status, fate

from gnome.weatherers.roc import (Burn)
from gnome.weatherers import (WeatheringData,
                              FayGravityViscous,
                              weatherer_sort)
from gnome.spill_container import SpillContainer
from gnome.spill import point_line_release_spill
from gnome.utilities.inf_datetime import InfDateTime
from gnome.environment import Waves, constant_wind, Water

from conftest import test_oil

delay = 1.
time_step = 900
rel_time = datetime(2015, 1, 1, 0, 0)
active_start = rel_time + timedelta(seconds=time_step)
active_stop = active_start + timedelta(hours=1.)
amount = 36000.
units = 'kg'


class ObjForTests:
    @classmethod
    def mk_test_objs(cls, water=None):
        '''
        create SpillContainer and test WeatheringData object est objects so
        we can run ROC tests like a model w/o using a full on model

        NOTE: Use this function to define class level objects. Other methods
            in this calss expect sc and weatherers to be class level objects
        '''

        if water is None:
            water = Water()

        weatherers = [WeatheringData(water), FayGravityViscous(water)]
        weatherers.sort(key=weatherer_sort)
        sc = SpillContainer()
        sc.spills += point_line_release_spill(10,
                                              (0, 0, 0),
                                              rel_time,
                                              substance=test_oil,
                                              amount=amount,
                                              units=units)
        return (sc, weatherers)

    def prepare_test_objs(self, obj_arrays=None):
        '''
        reset test objects
        '''
        self.sc.rewind()
        self.sc.rewind()
        at = set()

        for wd in self.weatherers:
            wd.prepare_for_model_run(self.sc)
            at.update(wd.array_types)

        if obj_arrays is not None:
            at.update(obj_arrays)

        self.sc.prepare_for_model_run(at)

    def reset_and_release(self, rel_time=None, time_step=900.0):
        '''
        reset test objects and release elements
        '''
        self.prepare_test_objs()
        if rel_time is None:
            rel_time = self.sc.spills[0].get('release_time')

        num_rel = self.sc.release_elements(time_step, rel_time)
        if num_rel > 0:
            for wd in self.weatherers:
                wd.initialize_data(self.sc, num_rel)

    def release_elements(self, time_step, model_time):
        '''
        release_elements - return num_released so test article can manipulate
        data arrays if required for testing
        '''
        num_rel = self.sc.release_elements(time_step, model_time)
        if num_rel > 0:
            for wd in self.weatherers:
                wd.initialize_data(self.sc, num_rel)

        return num_rel

    def step(self, test_weatherers, time_step, model_time):
        '''
        do a model step - since WeatheringData and FayGravity Viscouse are last
        in the list do the model step for test_weatherer first, then the rest

        tests don't necessarily add test_weatherer to the list - so provide
        as input
        '''
        # define order
        w_copy = [w for w in self.weatherers]
        w_copy.append(test_weatherer)
        w_copy.sort(key=weatherer_sort)
        
        # after release + initialize weather elements
        for wd in w_copy:
            wd.prepare_for_model_step(self.sc, time_step, model_time)

        # weather_elements
        for wd in w_copy:
            wd.weather_elements(self.sc, time_step, model_time)

        # step is done
        self.sc.model_step_is_done()
        self.sc['age'][:] = self.sc['age'][:] + time_step

        for wd in w_copy:
            wd.model_step_is_done(self.sc)

class TestROCBurn(ObjForTests):
    burn = Burn(offset=50,
                boom_length=250,
                boom_draft=10,
                speed=2,
                throughput=0.75,
                burn_effeciency_type=1)

    (sc, weatherers) = ObjForTests.mk_test_objs()

