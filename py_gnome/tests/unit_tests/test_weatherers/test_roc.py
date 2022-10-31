'''
tests for ROC
'''

from datetime import datetime, timedelta
def tds(seconds):
    return timedelta(seconds=seconds)

from pathlib import Path

import numpy as np

import pytest

from gnome.environment import Waves, constant_wind, Water

from gnome.weatherers.roc import (Burn, Disperse, Skim, Platform)
from gnome.weatherers import (Emulsification,
                              Evaporation)

from gnome.persist import load

from ..conftest import (test_oil, sample_model_weathering2)

import pprint as pp

output_dir = Path(__file__).parent / "output"

delay = 1.
time_step = 900
rel_time = datetime(2012, 9, 15, 12, 0)
active_start = rel_time + timedelta(seconds=time_step)
active_stop = active_start + timedelta(hours=24.)
amount = 36000.
units = 'kg'
wind = constant_wind(15., 0)
water = Water(temperature = 300.)
waves = Waves(wind, water)


class ROCTests(object):
    @classmethod
    def mk_objs(cls, sample_model_fcn2):
        model = sample_model_weathering2(sample_model_fcn2, test_oil, 333.0)
        model.set_make_default_refs(True)

        model.environment += [waves, wind, water]
 
        model.weatherers += Evaporation(wind=wind, water=water)
        model.weatherers += Emulsification(waves=waves)
        
        environment = {'water': model.environment[2]}
        return (list(model.spills.items())[0], model, environment)

    def prepare_test_objs(self, obj_arrays=None):
        self.model.rewind()
        self.model.rewind()

        at = dict()

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

        num_rel = self.sc.release_elements(rel_time, rel_time + tds(time_step), self.environment)
        
        if num_rel > 0:
            for wd in self.model.weatherers:
                wd.initialize_data(self.sc, num_rel)    

    def step(self, test_weatherer, time_step, model_time):
        test_weatherer.prepare_for_model_step(self.sc, time_step, model_time)
        self.model.step()
        test_weatherer.weather_elements(self.sc, time_step, model_time)


class TestRocGeneral(ROCTests):
    timeseries = np.array([(rel_time, rel_time + timedelta(hours=12.))])
    burn = Burn(offset=50.0,
                boom_length=250.0,
                boom_draft=10.0,
                speed=2.0,
                throughput=0.75,
                burn_efficiency_type=1,
                timeseries=timeseries)
                
    @pytest.mark.skip('new spreading incompatible with roc')
    def test_get_thickness(self, sample_model_fcn2):
        (self.sc, self.model, self.environment) = ROCTests.mk_objs(sample_model_fcn2)

        self.reset_and_release()
       
        assert self.burn._get_thickness(self.sc) == 0.0

        self.model.step()
        # assert self.burn._get_thickness(self.sc) == 0.16786582186002749

        self.model.step()
        # assert self.burn._get_thickness(self.sc) == 0.049809899105767913


class TestROCBurn(ROCTests):
    timeseries = np.array([(rel_time, rel_time + timedelta(hours=12.))])
    burn = Burn(offset=50.0,
                boom_length=250.0,
                boom_draft=10.0,
                speed=2.0,
                throughput=0.75,
                burn_efficiency_type=1,
                timeseries=timeseries)

    def test_prepare_for_model_run(self, sample_model_fcn2):
        self.sc, self.model, self.environment = ROCTests.mk_objs(sample_model_fcn2)

        self.reset_and_release()
        self.burn.prepare_for_model_run(self.sc)

        assert self.sc.mass_balance['burned'] == 0.0

        assert 'systems' in self.sc.mass_balance
        assert self.burn.id in self.sc.mass_balance['systems']

        assert self.sc.mass_balance['systems'][self.burn.id]['boomed'] == 0.0
        assert self.sc.mass_balance['boomed'] == 0.0

        assert self.burn._swath_width == 75
        assert self.burn._area == 1718.75
        assert self.burn.boom_draft == 10
        assert self.burn._offset_time == 14.805
        assert self.burn._area_coverage_rate == 0.3488372093023256

        assert round(self.burn._boom_capacity) == 477

        assert len(self.sc.report[self.burn.id]) == 1
        assert len(self.burn.timeseries) == 1

    def test_reports(self, sample_model_fcn2):
        self.sc, self.model, self.environment = ROCTests.mk_objs(sample_model_fcn2)

        self.reset_and_release()

        self.burn.boom_length = 3500.0

        self.burn.prepare_for_model_run(self.sc)

        assert self.burn._swath_width == 1050
        assert len(self.burn.report) == 2

    def test_serialize(self, sample_model_fcn2):
        self.sc, self.model, self.environment = ROCTests.mk_objs(sample_model_fcn2)

        self.reset_and_release()
        self.burn.serialize()

    def test_prepare_for_model_step(self, sample_model_fcn2):
        self.sc, self.model, self.environment = ROCTests.mk_objs(sample_model_fcn2)

        self.reset_and_release()

        self.burn.prepare_for_model_run(self.sc)
        self.burn.prepare_for_model_step(self.sc, time_step, active_start)

        assert self.burn._active is True

    @pytest.mark.skip("Needs fix after test subject was refactored")
    def test_weather_elements(self, sample_model_fcn2):
        self.sc, self.model, self.environment = ROCTests.mk_objs(sample_model_fcn2)

        self.model.time_step = 900
        self.reset_and_release()

        timeseries = np.array([(rel_time, rel_time + timedelta(hours=12.))])
        burn = Burn(offset=6000.0,
                    boom_length=100.0,
                    boom_draft=10.0,
                    speed=2.0,
                    throughput=0.75,
                    burn_efficiency_type=1,
                    timeseries=timeseries)

        self.model.weatherers.append(burn)
        self.model.rewind()

        self.model.step()
        assert burn._is_burning is False
        assert burn._is_collecting is True
        assert self.sc.mass_balance['burned'] == 0

        self.model.step()
        assert burn._is_burning is False
        assert burn._is_transiting is True
        assert burn._is_boom_full is True
        assert burn._burn_rate == 0.14

        assert np.isclose(burn._boom_capacity, 0, atol=0.01)
        assert self.sc.mass_balance['burned'] == 0
        collected = self.sc.mass_balance['boomed']

        self.model.step()
        assert burn._burn_time == 1414.2857142857142
        assert burn._burn_time_remaining <= burn._burn_time
        assert burn._is_collecting is False
        assert burn._is_cleaning is False
        assert burn._is_burning is True
        assert np.isclose(collected, 1877.2886248344857, 0.001)

        self.model.step()
        assert burn._is_transiting is False
        assert burn._is_burning is True
        assert burn._is_boom_full is False

        self.model.step()
        self.model.step()
        assert burn._is_burning is False
        assert burn._is_cleaning is True
        assert self.sc.mass_balance['boomed'] >= 0
        assert np.isclose(self.sc.mass_balance['boomed'], 0)
        # assert np.isclose(self.sc.mass_balance['burned'], collected)

        self.model.step()
        assert burn._is_burning is False
        assert burn._is_cleaning is True

        self.model.rewind()
        self.model.rewind()

        for step in self.model:
            print('amount in boom', self.sc.mass_balance['boomed'])
            assert self.sc.mass_balance['boomed'] >= 0
            print('===========', step['step_num'], '==============')
            print('collecting', burn._is_collecting)
            print('transiting', burn._is_transiting)
            print('burning', burn._is_burning)
            print('cleaning', burn._is_cleaning)

    def test_serialization(self):
        b = TestROCBurn.burn
        ser = b.serialize()
        b2 = Burn.deserialize(ser)

        print(b._diff(b2))
        assert b == b2

    def test_step(self, sample_model_fcn2):
        self.sc, self.model, self.environment = ROCTests.mk_objs(sample_model_fcn2)

        self.reset_and_release()
        self.model.step()


class TestPlatform(ROCTests):
    def test_construction(self):
        p = Platform()
        assert p.units == dict([(k, v[0]) for k, v in list(Platform._attr.items())])

        p = Platform(_name="Test Platform")
        assert p.transit_speed == 150
        assert p.max_op_time == 4

        p = Platform(_name="Test Platform", units={'transit_speed': 'm/s'})
        assert p.units['transit_speed'] == 'm/s'

    def test_serialization(self):
        p = Platform(_name='Test Platform')

        ser = p.serialize()

        p2 = Platform.deserialize(ser)
        assert p == p2


class TestRocChemDispersion(ROCTests):
    timeseries = np.array([(rel_time, rel_time + timedelta(hours=12.))])
    disp = Disperse(name='test_disperse',
                    transit=100,
                    pass_length=4,
                    # dosage=1,
                    cascade_on=False,
                    cascade_distance=None,
                    timeseries=timeseries,
                    loading_type='simultaneous',
                    pass_type='bidirectional',
                    disp_oil_ratio=None,
                    disp_eff=None,
                    platform='Test Platform',
                    units=None,)

    def test_construction(self):
        _d = Disperse(name='testname',
                      transit=100,
                      platform='Test Platform')
        # payload in gallons, computation in gallons, so no conversion

    def test_serialization(self):
        timeseries = np.array([(rel_time, rel_time + timedelta(hours=12.))])
        p = Disperse(name='test_disperse',
                     transit=100,
                     pass_length=4,
                     # dosage=1,
                     cascade_on=False,
                     cascade_distance=None,
                     timeseries=timeseries,
                     loading_type='simultaneous',
                     pass_type='bidirectional',
                     disp_oil_ratio=None,
                     disp_eff=None,
                     platform='Test Platform',
                     units=None,)

        ser = p.serialize()
        pp.pprint(ser)
        p2 = Disperse.deserialize(ser)
        assert p == p2

    def test_prepare_for_model_run(self, sample_model_fcn2):
        self.sc, self.model, self.environment = ROCTests.mk_objs(sample_model_fcn2)

        self.reset_and_release()
        self.disp.prepare_for_model_run(self.sc)

        # assert self.sc.mass_balance[self.disp.id] == 0.0
        assert self.disp.cur_state == 'retired'
        assert len(self.sc.report[self.disp.id]) == 0
        assert len(self.disp.timeseries) == 1

    def test_prepare_for_model_step(self, sample_model_fcn2):
        timeseries = np.array([(rel_time, rel_time + timedelta(hours=12.))])
        disp = Disperse(name='test_disperse',
                        transit=100,
                        pass_length=4,
                        dosage=1,
                        cascade_on=False,
                        cascade_distance=None,
                        timeseries=timeseries,
                        loading_type='simultaneous',
                        pass_type='bidirectional',
                        disp_oil_ratio=None,
                        disp_eff=None,
                        platform='Test Platform',
                        units=None,)

        self.sc, self.model, self.environment = ROCTests.mk_objs(sample_model_fcn2)

        self.model.weatherers += disp
        self.model.spills[0].amount = 1000
        self.model.spills[0].units = 'gal'

        self.reset_and_release()
        disp.prepare_for_model_run(self.sc)

        print(self.model.start_time)
        print(self.disp.timeseries)
        assert disp.cur_state == 'retired'

        self.model.step()
        print(self.model.current_time_step)

        self.model.step()
        print(list(self.model.spills.items())[0]['viscosity'])
        assert disp.cur_state == 'en_route'
        print(disp._next_state_time)

        self.model.step()
        assert disp.cur_state == 'en_route'
        print(disp.transit)
        print(disp.platform.transit_speed)
        print(disp.platform.one_way_transit_time(disp.transit)/60)

        while disp.cur_state == 'en_route':
            self.model.step()
            off = self.model.current_time_step * self.model.time_step
            print(self.model.start_time + timedelta(seconds=off))

        print('pump_rate ', disp.platform.eff_pump_rate(disp.dosage))

        try:
            for _step in self.model:
                off = self.model.current_time_step * self.model.time_step
                print(self.model.start_time + timedelta(seconds=off))
        except StopIteration:
            pass

    def test_prepare_for_model_step_cont(self, sample_model_fcn2):
        timeseries = np.array([(rel_time, rel_time + timedelta(hours=12.))])
        disp = Disperse(name='test_disperse',
                        transit=100,
                        pass_length=4,
                        # dosage=1,
                        cascade_on=False,
                        cascade_distance=None,
                        timeseries=timeseries,
                        loading_type='simultaneous',
                        pass_type='bidirectional',
                        disp_oil_ratio=None,
                        disp_eff=None,
                        platform='Test Platform',
                        units=None,)

        self.sc, self.model, self.environment = ROCTests.mk_objs(sample_model_fcn2)

        self.model.weatherers += disp

        self.model.spills[0].amount = 20000
        self.model.spills[0].units = 'gal'
        self.model.spills[0].end_release_time = (self.model.start_time +
                                                 timedelta(hours=3))

        self.reset_and_release()
        disp.prepare_for_model_run(self.sc)

        try:
            for _step in self.model:
                off = self.model.current_time_step * self.model.time_step
                print(self.model.start_time + timedelta(seconds=off))
        except StopIteration:
            pass

        # print self.model.start_time
        # print self.disp.timeseries
        # assert self.disp.cur_state == 'retired'

        # self.model.step()
        # print self.model.current_time_step

        # self.model.step()
        # print self.model.spills.items()[0]['viscosity']
        # assert self.disp.cur_state == 'en_route'
        # print self.disp._next_state_time

        # self.model.step()
        # assert self.disp.cur_state == 'en_route'
        # print self.disp.transit
        # print self.disp.platform.transit_speed
        # print self.disp.platform.one_way_transit_time(self.disp.transit)/60

        # while self.disp.cur_state == 'en_route':
        #     self.model.step()
        #     off = self.model.current_time_step * self.model.time_step
        #     print self.model.start_time + timedelta(seconds=off)
        # print ('pump_rate {}'
        #        .format(self.disp.platform.eff_pump_rate(self.disp.dosage)))
        # assert 'disperse' in self.disp.cur_state

        try:
            for _step in self.model:
                off = self.model.current_time_step * self.model.time_step

                assert all(self.sc['mass'] >= 0)
                assert np.all(self.sc['mass_components'] >= 0)
                assert ((self.sc.mass_balance['chem_dispersed'] +
                         self.sc.mass_balance['evaporated']) <=
                        sum(self.sc['init_mass']))
        except StopIteration:
            pass

    def test_prepare_for_model_step_boat(self, sample_model_fcn2):
        timeseries = np.array([(rel_time, rel_time + timedelta(hours=12.))])
        disp = Disperse(name='boat_disperse',
                        transit=20,
                        pass_length=4,
                        # dosage=1,
                        cascade_on=False,
                        cascade_distance=None,
                        timeseries=timeseries,
                        loading_type='simultaneous',
                        pass_type='bidirectional',
                        disp_oil_ratio=None,
                        disp_eff=None,
                        platform='Typical Large Vessel',
                        units=None,
                        onsite_reload_refuel=True)

        self.sc, self.model, self.environment = ROCTests.mk_objs(sample_model_fcn2)

        self.model.weatherers += disp

        self.model.spills[0].amount = 20000
        self.model.spills[0].units = 'gal'
        self.model.spills[0].end_release_time = (self.model.start_time +
                                                 timedelta(hours=3))

        self.reset_and_release()
        disp.prepare_for_model_run(self.sc)
        print(self.model.start_time)
        print(self.disp.timeseries)
        assert disp.cur_state == 'retired'

        self.model.step()
        print(self.model.current_time_step)

        self.model.step()
        print(list(self.model.spills.items())[0]['viscosity'])
        assert disp.cur_state == 'en_route'
        print(disp._next_state_time)

        self.model.step()
        assert disp.cur_state == 'en_route'
        print(disp.transit)
        print(disp.platform.transit_speed)
        print(disp.platform.one_way_transit_time(disp.transit) / 60)

        while disp.cur_state == 'en_route':
            self.model.step()
            off = self.model.current_time_step * self.model.time_step
            print(self.model.start_time + timedelta(seconds=off))

        print('pump_rate ', disp.platform.eff_pump_rate(disp.dosage))

        try:
            for _step in self.model:
                off = self.model.current_time_step * self.model.time_step
                print(self.model.start_time + timedelta(seconds=off))
        except StopIteration:
            pass


class TestRocSkim(ROCTests):
    skim = Skim(speed=2.0,
                storage=2000.0,
                swath_width=150,
                group='A',
                throughput=0.75,
                nameplate_pump=100.0,
                skim_efficiency_type='meh',
                recovery=0.75,
                recovery_ef=0.75,
                decant=0.75,
                decant_pump=150.0,
                discharge_pump=1000.0,
                rig_time=timedelta(minutes=30).total_seconds(),
                timeseries=[(datetime(2012, 9, 15, 12, 0),
                             datetime(2012, 9, 16, 1, 0))],
                transit_time=timedelta(hours=2).total_seconds())

    def test_prepare_for_model_run(self, sample_model_fcn2):
        self.sc, self.model, self.environment = ROCTests.mk_objs(sample_model_fcn2)

        self.reset_and_release()
        self.skim.prepare_for_model_run(self.sc)

    def test_prepare_for_model_step(self, sample_model_fcn2):
        self.sc, self.model, self.environment = ROCTests.mk_objs(sample_model_fcn2)

        self.reset_and_release()

        self.skim.prepare_for_model_run(self.sc)
        self.skim.prepare_for_model_step(self.sc, time_step, active_start)

        assert self.skim._active is True

    def test_weather_elements(self, sample_model_fcn2):
        self.sc, self.model, self.environment = ROCTests.mk_objs(sample_model_fcn2)

        self.reset_and_release()

        skim = Skim(speed=2.0,
                    storage=2000.0,
                    swath_width=150,
                    group='A',
                    throughput=0.75,
                    nameplate_pump=100.0,
                    skim_efficiency_type='meh',
                    recovery=0.75,
                    recovery_ef=0.75,
                    decant=0.75,
                    decant_pump=150.0,
                    discharge_pump=1000.0,
                    rig_time=timedelta(minutes=30),
                    timeseries=[(datetime(2012, 9, 15, 12, 0),
                                 datetime(2012, 9, 16, 1, 0))],
                    transit_time=timedelta(hours=2))

        self.model.weatherers.append(skim)

        self.model.rewind()

        self.model.step()
        self.model.step()
        self.model.step()
        self.model.step()

    def test_serialization(self):
        s = TestRocSkim.skim

        ser = s.serialize()

        s2 = Skim.deserialize(ser)


        assert s == s2

    def test_model_save(self, sample_model_fcn2):
        s = TestRocSkim.skim

        self.sc, self.model, self.environment = ROCTests.mk_objs(sample_model_fcn2)

        self.model.weatherers.append(s)
        self.model.save(output_dir)

    def test_model_load(self):
        _m = load('Model.zip')
