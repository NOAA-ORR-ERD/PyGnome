'''
tests for ROC
'''
from datetime import datetime, timedelta

import numpy as np
from pytest import raises, mark

import unit_conversion as us

from gnome.basic_types import oil_status, fate

from gnome.weatherers.roc import (Burn, Disperse, Platform)
from gnome.weatherers import (WeatheringData,
                              FayGravityViscous,
                              weatherer_sort,
                              Emulsification,
                              Evaporation)
from gnome.spill_container import SpillContainer
from gnome.spill import point_line_release_spill
from gnome.utilities.inf_datetime import InfDateTime
from gnome.environment import Waves, constant_wind, Water

from ..conftest import (test_oil, sample_model_weathering2)

delay = 1.
time_step = 900
rel_time = datetime(2012, 9, 15, 12, 0)
active_start = rel_time + timedelta(seconds=time_step)
active_stop = active_start + timedelta(hours=24.)
amount = 36000.
units = 'kg'
wind = constant_wind(15., 0)
water = Water()
waves = Waves(wind, water)

class ROCTests:
    @classmethod
    def mk_objs(cls, sample_model_fcn2):
        model = sample_model_weathering2(sample_model_fcn2, test_oil, 333.0)
        model.set_make_default_refs(True)
        model.environment += [waves, wind, water]
        model.weatherers += Evaporation(wind=wind, water=water)
        model.weatherers += Emulsification(waves=waves)
        return (model.spills.items()[0], model)

    def prepare_test_objs(self, obj_arrays=None):
        self.model.rewind()
        self.model.rewind()
        at = set()

        for wd in self.model.weatherers:
            wd.prepare_for_model_run(self.sc)
            at.update(wd.array_types)

        if obj_arrays is not None:
            at.update(obj_arrays)

        self.sc.prepare_for_model_run(at)

    def reset_and_release(self, rel_time=None, time_step=900.0):
        self.prepare_test_objs()
        if rel_time is None:
            rel_time = self.sc.spills[0].release_time

        num_rel = self.sc.release_elements(time_step, rel_time)
        if num_rel > 0:
            for wd in self.model.weatherers:
                wd.initialize_data(self.sc, num_rel)

    def step(self, test_weatherer, time_step, model_time):
        test_weatherer.prepare_for_model_step(self.sc, time_step, model_time)
        self.model.step()
        test_weatherer.weather_elements(self.sc, time_step, model_time)

class TestRocGeneral(ROCTests):
    burn = Burn(offset=50.0,
                boom_length=250.0,
                boom_draft=10.0,
                speed=2.0,
                throughput=0.75,
                burn_efficiency_type=1,
                timeseries=np.array([(rel_time, rel_time + timedelta(hours=12.))]))

    def test_get_thickness(self, sample_model_fcn2):
        (self.sc, self.model) = ROCTests.mk_objs(sample_model_fcn2)
        self.reset_and_release()
        assert self.burn._get_thickness(self.sc) == 0.0
        self.model.step()
#         assert self.burn._get_thickness(self.sc) == 0.16786582186002749
        self.model.step()
#         assert self.burn._get_thickness(self.sc) == 0.049809899105767913

class TestROCBurn(ROCTests):
    burn = Burn(offset=50.0,
                 boom_length=250.0,
                 boom_draft=10.0,
                 speed=2.0,
                 throughput=0.75,
                 burn_efficiency_type=1,
                 timeseries=np.array([(rel_time, rel_time + timedelta(hours=12.))]))

    def test_prepare_for_model_run(self, sample_model_fcn2):
        (self.sc, self.model) = ROCTests.mk_objs(sample_model_fcn2)
        self.reset_and_release()
        self.burn.prepare_for_model_run(self.sc)
        assert self.sc.mass_balance['burned'] == 0.0
        assert self.sc.mass_balance[self.burn.id] == 0.0
        assert self.sc.mass_balance['boomed'] == 0.0
        assert self.burn._swath_width == 75
        assert self.burn._area == 1718.75
        assert self.burn.boom_draft == 10
        assert self.burn._offset_time == 14.805
        assert round(self.burn._boom_capacity) == 477
        assert len(self.sc.report[self.burn.id]) == 1
        assert self.burn._area_coverage_rate == 0.3488372093023256
        assert len(self.burn.timeseries) == 1

    def test_reports(self, sample_model_fcn2):
        (self.sc, self.model) = ROCTests.mk_objs(sample_model_fcn2)
        self.reset_and_release()
        self.burn.boom_length = 3500.0
        self.burn.prepare_for_model_run(self.sc)
        assert self.burn._swath_width == 1050
        assert len(self.burn.report) == 2

    def test_serialize(self, sample_model_fcn2):
        (self.sc, self.model) = ROCTests.mk_objs(sample_model_fcn2)
        self.reset_and_release()
        self.burn.serialize()

    def test_prepare_for_model_step(self, sample_model_fcn2):
        (self.sc, self.model) = ROCTests.mk_objs(sample_model_fcn2)
        self.reset_and_release()

        self.burn.prepare_for_model_run(self.sc)
        self.burn.prepare_for_model_step(self.sc, time_step, active_start)

        assert self.burn._active == True
        assert self.burn._ts_collected == 93576.38888888889

    def test_weather_elements(self, sample_model_fcn2):
        (self.sc, self.model) = ROCTests.mk_objs(sample_model_fcn2)
        self.reset_and_release()
        burn = Burn(offset=3000.0,
                 boom_length=100.0,
                 boom_draft=10.0,
                 speed=2.0,
                 throughput=0.75,
                 burn_efficiency_type=1,
                 timeseries=np.array([(rel_time, rel_time + timedelta(hours=12.))]))

        self.model.weatherers.append(burn)
        self.model.rewind()
        self.model.step()
        self.model.step()
        assert burn._ts_collected == self.sc.mass_balance['boomed']
        assert burn._ts_collected <= burn._boom_capacity
        assert burn._burn_rate == 0.14
        assert burn._burn_time == 1414.2857142857142
        assert burn._is_boom_full == True
        assert burn._is_collecting == False
        assert burn._is_transiting == True
        collected = self.sc.mass_balance['boomed']
        self.model.step()
        assert burn._is_transiting == False
        assert burn._is_burning == True
        self.model.step()
        assert self.sc.mass_balance['burned'] != 0
        assert burn._burn_time_remaining <= burn._burn_time
        self.model.step()
        assert self.sc.mass_balance['boomed'] == 0
        assert self.sc.mass_balance['burned'] == collected
        assert burn._is_burning == False
        assert burn._is_cleaning == True
        self.model.step()
        self.model.step()
        self.model.step()
        self.model.step()
        self.model.step()
        self.model.step()
        assert burn._is_cleaning == False
        assert burn._is_transiting == True
        assert burn._is_boom_filled == False

        self.model.step()
        self.model.step()
        self.model.step()
        assert burn._active == True
        assert burn._is_collecting == False
        assert burn._is_transiting == False
        assert burn._is_cleaning == False
        assert burn._is_burning == True

    def test_serialization(self):
        b = TestROCBurn.burn
        import pprint as pp
        ser = b.serialize()
        pp.pprint(ser)
        deser = Burn.deserialize(ser)

        pp.pprint(deser)

        b2 = Burn.new_from_dict(deser)
        ser2 = b2.serialize()
        pp.pprint(ser2)

        print 'INCORRECT BELOW'

        ser.pop('id')
        ser2.pop('id')
        assert ser == ser2


    def test_step(self, sample_model_fcn2):
        (self.sc, self.model) = ROCTests.mk_objs(sample_model_fcn2)
        self.reset_and_release()
        self.model.step()


class TestPlatform(ROCTests):

    def test_construction(self):
        p = Platform()
        assert p.units == dict([(k,v[0]) for k, v in Platform._attr.items()])
        p = Platform(_name = "Test Platform")
        assert p.transit_speed == 150
        assert p.max_op_time == 4
        p = Platform(_name = "Test Platform", units = {'transit_speed':'m/s'})
        assert p.units['transit_speed'] == 'm/s'

    def test_serialization(self):
        p = Platform(_name='Test Platform')
        import pprint as pp
        ser = p.serialize()
        pp.pprint(ser)
        deser = Platform.deserialize(ser)

        pp.pprint(deser)

        p2 = Platform.new_from_dict(deser)
        ser2 = p2.serialize()
        pp.pprint(ser2)

        print 'INCORRECT BELOW'

        ser.pop('id')
        ser2.pop('id')
        assert ser == ser2



class TestRocChemDispersion(ROCTests):

    def test_construction(self):
        d = Disperse(name='testname',
                     transit=100,
                     platform='Test Platform')
        #payload in gallons, computation in gallons, so no conversion
