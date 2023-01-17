'''
tests for cleanup options
'''

from datetime import datetime, timedelta

import numpy as np
from pytest import raises, mark

import nucos as uc

from gnome.basic_types import oil_status, fate

from gnome.weatherers.cleanup import CleanUpBase
from gnome.weatherers import (
                              FayGravityViscous,
                              Skimmer,
                              Burn,
                              ChemicalDispersion,
                              weatherer_sort)
from gnome.spill_container import SpillContainer
from gnome.spills import surface_point_line_spill
from gnome.utilities.inf_datetime import InfDateTime
from gnome.environment import Waves, constant_wind, Water

from gnome.ops import weathering_array_types

from .conftest import test_oil

delay = 1.
time_step = 900

rel_time = datetime(2014, 1, 1, 0, 0)

active_start = rel_time + timedelta(seconds=time_step)
active_range = (rel_time + timedelta(seconds=time_step),
                active_start + timedelta(hours=1.))

amount = 36000.
units = 'kg'    # leave as SI units


class ObjForTests(object):
    @classmethod
    def mk_test_objs(cls, water=None):
        '''
        create SpillContainer and test WeatheringData object test objects so
        we can run Skimmer, Burn like a model without using a full on Model

        NOTE: Use this function to define class level objects. Other methods
            in this class expect sc, and weatherers to be class level objects
        '''
        # spreading does not need to be initialized correctly for these tests,
        # but since we are mocking the model, let's do it correctly
        if water is None:
            water = Water(temperature = 300.) 
        environment = {'water': water} 
        # keep this order
        weatherers = [FayGravityViscous(water),]
        weatherers.sort(key=weatherer_sort)
        sc = SpillContainer()
        print("******************")
        print("Adding a spill to spill container")
        sc.spills += surface_point_line_spill(10,
                                              (0, 0, 0),
                                              rel_time,
                                              substance=test_oil,
                                              amount=amount,
                                              units='kg',
                                              water=water)
        return (sc, weatherers, environment)

    def prepare_test_objs(self, obj_arrays=None):
        '''
        reset test objects
        '''
        # print '\n------------\n', 'reset_and_release', '\n------------'
        self.sc.rewind()
        at = dict()
        for wd in self.weatherers:
            wd.prepare_for_model_run(self.sc)
            at.update(wd.array_types)

        if obj_arrays is not None:
            at.update(obj_arrays)
        
        at.update(weathering_array_types)
        
        self.sc.prepare_for_model_run(at)

    def reset_and_release(self, rel_time=None, time_step=900.0):
        '''
        reset test objects and release elements
        '''
        self.prepare_test_objs()
        if rel_time is None:
            # there is only one spill, use its release time
            rel_time = self.sc.spills[0].release_time
# 01/25/2022 add environment object below sc.release
        num_rel = self.sc.release_elements(rel_time, rel_time + timedelta(seconds=time_step), self.environment)
        if num_rel > 0:
            for wd in self.weatherers:
                wd.initialize_data(self.sc, num_rel)

    def release_elements(self, time_step, model_time, environment):
        '''
        release_elements - return num_released so test article can manipulate
        data arrays if required for testing
        '''
        num_rel = self.sc.release_elements(model_time, model_time + timedelta(seconds=time_step), self.environment)

        if num_rel > 0:
            for wd in self.weatherers:
                wd.initialize_data(self.sc, num_rel)

        return num_rel

    def step(self, test_weatherer, time_step, model_time):
        '''
        do a model step - since WeatheringData and FayGravityViscous are last
        in the list, do the model step for test_weatherer first, then the rest

        tests don't necessarily add test_weatherer to the list - so provide
        as input
        '''
        # define order
        w_copy = [w for w in self.weatherers]
        w_copy.append(test_weatherer)
        w_copy.sort(key=weatherer_sort)

        # after release + initialize, weather elements
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


class TestCleanUpBase(object):
    base = CleanUpBase()

    def test_init(self):
        base = CleanUpBase()
        assert base.efficiency == 1.0

    def test_set_efficiency(self):
        '''
        efficiency updates only if value is valid
        '''
        curr_val = .4
        self.base.efficiency = curr_val

        self.base.efficiency = -0.1
        assert self.base.efficiency == curr_val

        self.base.efficiency = 1.1
        assert self.base.efficiency == curr_val

        self.base.efficiency = 0.0
        assert self.base.efficiency == 0.0

        self.base.efficiency = 1.0
        assert self.base.efficiency == 1.0


class TestSkimmer(ObjForTests):
    (sc, weatherers, environment) = ObjForTests.mk_test_objs()

    skimmer = Skimmer(amount,
                      units='kg',
                      efficiency=0.3,
                      active_range=active_range,
                      water=weatherers[0].water)

    def test_prepare_for_model_run(self):
        self.reset_and_release()
        self.skimmer.prepare_for_model_run(self.sc)

        assert self.sc.mass_balance['skimmed'] == 0.
        assert (self.skimmer._rate ==
                (self.skimmer.amount /
                 (self.skimmer.active_range[1] -
                  self.skimmer.active_range[0]).total_seconds()))

    @mark.parametrize(("model_time", "active", "ts"),
                      [(active_range[0], True, time_step),
                       (active_range[0] - timedelta(seconds=time_step / 3.), True,
                        time_step * 2. / 3.),  # between active start - active stop
                       (active_range[0] + timedelta(seconds=time_step / 2.), True,
                        time_step),     # between active start - active stop
                       (active_range[1], False, None),
                       (active_range[1] - timedelta(seconds=time_step), True,
                        time_step),     # before active stop
                       (active_range[1] - timedelta(seconds=time_step / 3.), True,
                        time_step / 3.)   # before active stop
                       ])
    def test_prepare_for_model_step(self, model_time, active, ts):
        '''
        assert that the _timestep property is being set correctly. This is
        less than or equal to time_step specified by model. Since we define
        a rate object, this is just to ensure the amount Skimmed between the
        duration specified matches given a constant skim rate.
        '''
        self.reset_and_release()
        self.skimmer.prepare_for_model_step(self.sc, time_step, model_time)

        assert self.skimmer.active is active
        if active:
            assert self.skimmer._timestep == ts
            mask = self.sc['fate_status'] & fate.skim == fate.skim
            assert mask.sum() > 0

    @mark.parametrize("avg_frac_water", [0.0, 0.4])
    def test_weather_elements(self, avg_frac_water):
        '''
        Test only the mass removed due to Skimmer operation:
        1) sc['mass'] + sc.mass_balance['skimmed'] =  spill_amount
        2) sc.mass_balance['skimmed']/skimmer.amount = skimmer.efficiency
        '''
        self.prepare_test_objs()
        self.skimmer.prepare_for_model_run(self.sc)

        assert self.sc.mass_balance['skimmed'] == 0.0

        model_time = rel_time
        while (model_time <
               self.skimmer.active_range[1] + timedelta(seconds=2*time_step)):

            amt_skimmed = self.sc.mass_balance['skimmed']
            num_rel = self.release_elements(time_step, model_time, self.environment)
            if num_rel > 0:
                self.sc['frac_water'][:] = avg_frac_water

            self.step(self.skimmer, time_step, model_time)

            if not self.skimmer.active:
                assert self.sc.mass_balance['skimmed'] == amt_skimmed
            else:
                # check total amount removed at each timestep
                assert self.sc.mass_balance['skimmed'] > amt_skimmed

            model_time += timedelta(seconds=time_step)

            # check - useful for debugging issues with recursion
            assert np.isclose(amount,
                              (self.sc.mass_balance['skimmed'] +
                               self.sc['mass'].sum()))

        # following should finally hold true for entire run
        assert np.allclose(amount, self.sc.mass_balance['skimmed'] +
                           self.sc['mass'].sum(), atol=1e-6)
        # efficiency decreased since only (1 - avg_frac_water) is the fraction
        # of oil collected by skimmer
        assert np.allclose(self.sc.mass_balance['skimmed']/self.skimmer.amount,
                           self.skimmer.efficiency * (1 - avg_frac_water),
                           atol=1e-6)


class TestBurn(ObjForTests):
    '''
    Define a default object
    default units are SI
    '''
    (sc, weatherers, environment) = ObjForTests.mk_test_objs()
    spill = sc.spills[0]
    op = spill.substance
    volume = spill.get_mass() / op.standard_density

    thick = 1
    area = (0.5 * volume) / thick

    # test with non SI units
    burn = Burn(area, thick, active_range,
                area_units='km^2', thickness_units='km',
                efficiency=1.0)

    def test_init(self):
        '''
        active stop is ignored if set by user
        '''
        burn = Burn(self.area,
                    self.thick,
                    active_range=(active_start, InfDateTime('inf')),
                    name='test_burn',
                    on=False)   # this is ignored!

        # use burn constant for test - it isn't stored anywhere
        duration = (uc.convert('Length', burn.thickness_units, 'm',
                               burn.thickness) -
                    burn._min_thickness) / burn._burn_constant

        assert (burn.active_range[1] ==
                burn.active_range[0] + timedelta(seconds=duration))
        assert burn.name == 'test_burn'
        assert not burn.on

    def test_update_active_start(self):
        '''
        active stop should be updated if we update active start or thickness
        '''
        burn = Burn(self.area,
                    self.thick,
                    active_range=(active_start, InfDateTime('inf')),
                    name='test_burn',
                    on=False)   # this is ignored!

        # use burn constant for test - it isn't stored anywhere
        duration = ((uc.convert('Length', burn.thickness_units, 'm',
                                burn.thickness) -
                     burn._min_thickness) /
                    burn._burn_constant)

        assert (burn.active_range[1] ==
                burn.active_range[0] + timedelta(seconds=duration))

        # after changing active start, active stop should still match the
        # duration.
        burn.active_range = (burn.active_range[0] + timedelta(days=1),
                             burn.active_range[1])
        assert (burn.active_range[1] ==
                burn.active_range[0] + timedelta(seconds=duration))

    @mark.parametrize(("area_units", "thickness_units"), [("m", "in"),
                                                          ("m^2", "l")])
    def test_units_exception(self, area_units, thickness_units):
        ''' tests incorrect units raises exception '''
        with raises(uc.InvalidUnitError):
            self.burn.area_units = "in"

        with raises(uc.InvalidUnitError):
            self.burn.thickness_units = "m^2"

        with raises(uc.InvalidUnitError):
            Burn(10, 1, (datetime.now(), InfDateTime('inf')),
                 area_units=area_units,
                 thickness_units=thickness_units)

    def test_prepare_for_model_run(self):
        ''' check _oilwater_thickness, _burn_duration is reset'''
        self.burn._oilwater_thickness = 0.002   # reached terminal thickness
        self.burn.prepare_for_model_run(self.sc)
        assert (self.burn._oilwater_thickness ==
                uc.convert('Length', self.burn.thickness_units, 'm',
                           self.burn.thickness))

    def test_prepare_for_model_step(self):
        '''
        once thickness reaches _min_thickness, test the mover becomes inactive
        if mover is off, it is also inactive
        '''
        self.burn.prepare_for_model_run(self.sc)
        self._oil_thickness = 0.002
        self.burn.prepare_for_model_step(self.sc, time_step, rel_time)
        assert not self.burn._active

        # turn it off
        self.burn.on = False
        self.burn.prepare_for_model_step(self.sc, time_step, rel_time)
        assert not self.burn._active

    def _weather_elements_helper(self, burn, avg_frac_water=0.0):
        '''
        refactored model run from test_weather_elements to this helper function
        It is also used by next test:
        test_elements_weather_slower_with_frac_water
        '''
        self.prepare_test_objs()

        model_time = rel_time
        burn.prepare_for_model_run(self.sc)

        # once burn becomes active, sc.mass_balance['burned'] > 0.0 and
        # burn becomes inactive once
        # burn._oil_thickness == burn._min_thickness
        step_num = 0
        while ((model_time > burn.active_range[0] and
                burn.active) or self.sc.mass_balance['burned'] == 0.0):

            num_rel = self.release_elements(time_step, model_time, self.environment)
            if num_rel > 0:
                self.sc['frac_water'][:] = avg_frac_water

            self.step(burn, time_step, model_time)
            dt = timedelta(seconds=time_step)
            if (model_time + dt <= burn.active_range[0] or
                    model_time >= burn.active_range[1]):
                # if model_time + dt == burn.active start, then start the burn
                # in next step
                assert not burn.active
            else:
                assert burn.active

            model_time += dt
            step_num += 1

            if step_num > 100:
                # none of the tests take that long, so break it
                msg = "Test took more than 100 iterations for Burn, check test"
                raise Exception(msg)
                break

        # check that active stop is being correctly set
        assert InfDateTime('inf') > burn.active_range[1]
        print('\nCompleted steps: {0:2}'.format(step_num))

    @mark.parametrize(("thick", "avg_frac_water", "units"), [(0.003, 0, 'm'),
                                                             (1, 0, 'm'),
                                                             (1, 0.3, 'm'),
                                                             (100, 0.3, 'cm')])
    def test_weather_elements(self, thick, avg_frac_water, units):
        '''
        weather elements and test. frac_water is 0. Test thickness in units
        other than 'm'.

        1) tests the expected burned mass equals 'burned' amount stored in
           mass_balance
        2) also tests the mass_remaining is consistent with what we expect
        3) tests the mass of LEs set for burn equals the mass of oil given
           avg_frac_water and the thickness, and area. Since we cannot have
           a fraction of an LE, the difference should be within the mass of
           one LE.

        Also sets the 'frac_water' to 0.5 for one of the tests just to ensure
        it works.
        '''
        self.spill.num_elements = 500
        print('starting num_elements = ', self.spill.num_elements)
        print('time_step = ', time_step)

        thick_si = uc.convert('Length', units, 'm', thick)
        area = (0.5 * self.volume) / thick_si
        water = self.weatherers[0].water

        burn = Burn(area, thick, active_range, thickness_units=units,
                    efficiency=1.0, water=water)

        # return the initial value of burn._oil_thickness - this is starting
        # thickness of the oil
        self._weather_elements_helper(burn, avg_frac_water)

        # following should finally hold true for entire run
        v = self.sc.mass_balance['burned'] + self.sc['mass'].sum()
        assert np.allclose(amount, v, atol=1e-6)

        # want mass of oil thickness * area gives volume of oil-water so we
        # need to scale this by (1 - avg_frac_water)
        exp_burned = ((thick_si - burn._min_thickness) * burn.area *
                      (1 - avg_frac_water) *
                      self.op.standard_density)
        assert np.isclose(self.sc.mass_balance['burned'], exp_burned)

        mask = self.sc['fate_status'] & fate.burn == fate.burn

        # given LEs are discrete elements, we cannot add a fraction of an LE
        mass_per_le = self.sc['init_mass'][mask][0]
        exp_init_oil_mass = (burn.area * thick_si * (1 - avg_frac_water) *
                             self.op.standard_density)
        assert (self.sc['init_mass'][mask].sum() - exp_init_oil_mass <
                mass_per_le and
                self.sc['init_mass'][mask].sum() - exp_init_oil_mass >= 0.0)

        mass_remain_for_burn_LEs = self.sc['mass'][mask].sum()

        duration = ((burn.active_range[1] - burn.active_range[0])
                    .total_seconds() / 3600)
        print(('Current Thickness: {0:.3f}, '
               'Duration (hrs): {1:.3f}').format(burn._oilwater_thickness,
                                                 duration))

        exp_mass_remain = (burn._oilwater_thickness * burn.area *
                           (1 - avg_frac_water) *
                           self.op.standard_density)
        # since we don't adjust the thickness anymore need to use min_thick
        min_thick = .002
        exp_mass_remain = (min_thick * burn.area *
                           (1.0 - avg_frac_water) *
                           self.op.standard_density)

        assert np.allclose(exp_mass_remain, mass_remain_for_burn_LEs,
                           rtol=0.001)

    def test_elements_weather_faster_with_frac_water(self):
        '''
        Tests that avg_water_frac > 0 weathers faster
        '''
        self.spill.num_elements = 500
        area = (0.5 * self.volume) / 1.
        water = self.weatherers[0].water

        burn1 = Burn(area, 1., active_range, efficiency=1.0, water=water)
        burn2 = Burn(area, 1., active_range, efficiency=1.0, water=water)
        burn3 = Burn(area, 1., active_range, efficiency=1.0, water=water)

        self._weather_elements_helper(burn1)
        self._weather_elements_helper(burn2, avg_frac_water=0.3)
        self._weather_elements_helper(burn3, avg_frac_water=0.5)

        print("frac_water", 1.0, "burn_duration", \
            round((burn1.active_range[1] - burn1.active_range[0])
                  .total_seconds()))
        print("frac_water", 0.3, "burn_duration", \
            round((burn2.active_range[1] - burn2.active_range[0])
                  .total_seconds()))
        print("frac_water", 0.9, "burn_duration", \
            round((burn3.active_range[1] - burn3.active_range[0])
                  .total_seconds()))
        assert (burn1.active_range[1] - burn1.active_range[0] <
                burn2.active_range[1] - burn2.active_range[0] <
                burn3.active_range[1] - burn3.active_range[0])

    def test_efficiency(self):
        '''
            tests efficiency.
            - If burn2 efficiency is 0.7 and burn1 efficiency is 1.0,
              then burn2_amount/burn1_amount = 0.7
            - Also checks the burn duration is not effected by efficiency

            The burn duration for both will be the same since efficiency only
            effects the amount of oil burned. The rate at which the oil/water
            mixture goes down to 2mm only depends on fractional water content.
        '''
        self.spill.num_elements = 500
        area = (0.5 * self.volume) / 1.
        eff = 0.7
        water = self.weatherers[0].water

        burn1 = Burn(area, 1., active_range, efficiency=1.0, water=water)
        burn2 = Burn(area, 1., active_range, efficiency=eff, water=water)

        self._weather_elements_helper(burn1)
        amount_burn1 = self.sc.mass_balance['burned']
        self._weather_elements_helper(burn2)
        amount_burn2 = self.sc.mass_balance['burned']

        assert np.isclose(amount_burn2 / amount_burn1, eff)
        assert burn1.active_range == burn2.active_range

    def test_zero_efficiency(self):
        '''
            tests efficiency.
            - If burn2 efficiency is 0.0 and burn1 efficiency is 1.0,
              then burn2_amount/burn1_amount = 0.0
            - Also checks the burn duration is not effected by efficiency

            The burn duration for both will be the same since efficiency only
            effects the amount of oil burned. The rate at which the oil/water
            mixture goes down to 2mm only depends on fractional water content.
        '''
        self.spill.num_elements = 500
        area = (0.5 * self.volume) / 1.
        eff = 0.0
        water = self.weatherers[0].water

        burn1 = Burn(area, 1., active_range, efficiency=1.0, water=water)
        burn2 = Burn(area, 1., active_range, efficiency=eff, water=water)

        self._weather_elements_helper(burn1)
        amount_burn1 = self.sc.mass_balance['burned']

        with raises(Exception):
            self._weather_elements_helper(burn2)

        amount_burn2 = self.sc.mass_balance['burned']

        assert np.isclose(amount_burn2/amount_burn1, eff)
        assert burn1.active_range == burn2.active_range

#     def test_update_from_dict(self):
#         '''
#         test that the update_from_dict correctly sets efficiency to None
#         if it is dropped from json payload if user chose compute from wind
#         '''
#         self.burn.wind = constant_wind(5, 0)
#         json_ = self.burn.serialize()
#         assert self.burn.efficiency is not None
#         del json_['efficiency']
#
#         dict_ = Burn.deserialize(json_)
#         dict_['wind'] = self.burn.wind
#         assert 'wind' in dict_
#         self.burn.update_from_dict(dict_)
#         assert self.burn.efficiency is None
#
#         json_['efficiency'] = .4
#
#         # make sure None for wind doesn't break it
#         dict_['wind'] = None
#         dict_ = Burn.deserialize(json_)
#         self.burn.update_from_dict(dict_)
#         assert self.burn.efficiency == json_['efficiency']
#
#         # update area/thickness
#         st = self.burn.active_range[1]
#         self.burn.thickness *= 2
#         assert self.burn.active_range[1] > st
#
#         # burn duration just depents on thickness - not area
#         st = self.burn.active_range[1]
#         self.burn.area *= 2
#         assert self.burn.active_range[1] == st


class TestChemicalDispersion(ObjForTests):
    (sc, weatherers, environment) = ObjForTests.mk_test_objs()
    spill = sc.spills[0]
    op = spill.substance
    spill_pct = 0.2  # 20%
    c_disp = ChemicalDispersion(spill_pct,
                                active_range,
                                efficiency=0.3)

    def test_prepare_for_model_run(self):
        '''
        test efficiency gets set correctly
        '''
        self.prepare_test_objs()

        assert 'chem_dispersed' not in self.sc.mass_balance

        self.c_disp.prepare_for_model_run(self.sc)
        assert self.sc.mass_balance['chem_dispersed'] == 0.0

    def test_set_efficiency(self):
        '''
        for wave height > 6.4 m, efficiency goes to 0
        '''
        # make wind large so efficiency goes to 0
        waves = Waves(constant_wind(0, 0), water=Water())
        c_disp = ChemicalDispersion(self.spill_pct,
                                    active_range,
                                    waves=waves)
        pts = np.array([[0, 0], [0, 0]])
        c_disp._set_efficiency(pts, self.spill.release_time)
        assert c_disp.efficiency == 1.0

        c_disp.efficiency = 0
        waves.wind.timeseries = (waves.wind.timeseries[0]['time'], (100, 0))
        c_disp._set_efficiency(pts, self.spill.release_time)
        assert np.all(c_disp.efficiency == 0)

    @mark.parametrize("efficiency", (0.5, 1.0))
    def test_prepare_for_model_step(self, efficiency):
        '''
        updated: efficiency now does impact the mass of LEs marked as
                 having been sprayed. precent_sprayed also impacts the
                 mass of LEs marked as disperse.
        '''
        self.reset_and_release()
        self.c_disp.efficiency = efficiency

        assert np.all(self.sc['fate_status'] == fate.surface_weather)

        self.c_disp.prepare_for_model_run(self.sc)
        self.c_disp.prepare_for_model_step(self.sc, time_step, active_start)
        d_mass = self.sc['mass'][self.sc['fate_status'] == fate.disperse].sum()

        assert d_mass == (self.c_disp.fraction_sprayed *
                          self.spill.get_mass() *
                          efficiency)
        exp_mass = (self.spill.get_mass() *
                    self.c_disp.fraction_sprayed *
                    efficiency)
        assert d_mass - exp_mass < self.sc['mass'][0]

    @mark.parametrize("frac_water", (0.5, 0.0))
    def test__update_LE_status_codes(self, frac_water):
        '''
        efficiency does not impact the mass of LEs marked as having been
        sprayed. precent_sprayed determines percent of LEs marked as disperse.
        '''
        self.reset_and_release()
        self.sc['frac_water'][:] = frac_water
        assert np.all(self.sc['fate_status'] == fate.surface_weather)

        self.c_disp.prepare_for_model_run(self.sc)
        self.c_disp.prepare_for_model_step(self.sc, time_step, active_start)
        d_mass = self.sc['mass'][self.sc['fate_status'] == fate.disperse].sum()

        assert d_mass == self.c_disp.fraction_sprayed * self.spill.get_mass()
        exp_mass = self.spill.get_mass() * self.c_disp.fraction_sprayed
        assert d_mass - exp_mass < self.sc['mass'][0]

    @mark.parametrize("efficiency", (1.0, 0.7))
    def test_weather_elements(self, efficiency):
        self.prepare_test_objs()
        self.c_disp.efficiency = efficiency
        self.c_disp.prepare_for_model_run(self.sc)

        assert self.sc.mass_balance['chem_dispersed'] == 0.0

        model_time = self.spill.release_time
        while (model_time < self.c_disp.active_range[1] +
               timedelta(seconds=time_step)):
            amt_disp = self.sc.mass_balance['chem_dispersed']
            self.release_elements(time_step, model_time, self.environment)
            self.step(self.c_disp, time_step, model_time)

            if not self.c_disp.active:
                assert self.sc.mass_balance['chem_dispersed'] == amt_disp
            else:
                assert self.sc.mass_balance['chem_dispersed'] > amt_disp

            model_time += timedelta(seconds=time_step)

        assert np.allclose(amount, self.sc.mass_balance['chem_dispersed'] +
                           self.sc['mass'].sum(), atol=1e-6)
        assert np.allclose(self.sc.mass_balance['chem_dispersed'] /
                           self.spill.get_mass(),
                           self.c_disp.fraction_sprayed * efficiency)
