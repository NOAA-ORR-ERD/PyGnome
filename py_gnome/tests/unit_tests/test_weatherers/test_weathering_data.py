'''
test objects defined in weathering_data module
'''

from datetime import datetime, timedelta

import numpy as np
import pytest
from testfixtures import log_capture

from gnome.environment import Water
from gnome.weatherers import WeatheringData, FayGravityViscous
from gnome.spills import surface_point_line_spill
from gnome.spills.gnome_oil import GnomeOil
from gnome.spill_container import SpillContainer

from ..conftest import test_oil


default_ts = 900  # default timestep for tests
water = Water()
water = Water(283.16)  # 10C, not the default!


class TestWeatheringData(object):
    def test_init(self):
        WeatheringData(water)

    def sample_sc_intrinsic(self, num_elements, rel_time, add_at=None):
        '''
        initialize Sample SC and WeatheringData object
        objects are constructed and prepare_for_model_run() is invoked on all
        '''
        wd = WeatheringData(water)
        end_time = rel_time + timedelta(hours=1)
        spills = [surface_point_line_spill(num_elements,
                                           (0, 0, 0),
                                           rel_time,
                                           end_release_time=end_time,
                                           amount=100,
                                           units='kg',
                                           substance=test_oil)]
        sc = SpillContainer()
        sc.spills += spills
        at = wd.array_types
        if add_at is not None:
            at.update(add_at)
        sc.prepare_for_model_run(at)

        # test initialization as well
        wd.prepare_for_model_run(sc)
        for val in list(sc.mass_balance.values()):
            assert val == 0.0

        # test initialization as well
        return (sc, wd)

    def sample_sc_wd_spreading(self, num_elements, rel_time):
        '''
        return sample SC, WeatheringData object and FayGravityViscous object
        objects are constructed and prepare_for_model_run() is invoked on all
        '''
        spread = FayGravityViscous()
        (sc, wd) = self.sample_sc_intrinsic(num_elements,
                                            rel_time,
                                            spread.array_types)
        spread.water = wd.water
        spread.prepare_for_model_run(sc)

        return (sc, wd, spread)

    def mock_weather_data(self, sc, wd, zero_elems=5):
        '''
        helper function that mocks a weatherer - like evaporation. It simply
        changes the mass_fraction and updates frac_evap accordingly
        '''
        for _, data in sc.itersubstancedata(wd.array_types):
            # following simulates weathered/evaporated oil
            data['mass_components'][:, :zero_elems] = 0
            data['mass'][:] = sc['mass_components'].sum(1)
            data['frac_evap'][:] = 1 - data['mass']/data['init_mass']

        sc.update_from_fatedataview()

    def step(self, wd, sc, rel_time, time_step=default_ts):
        '''
        WeatheringData is updating intrinsic properties in each step - invoke
        weathering_elements()
        '''
        # say we are now in 2nd step - no new particles are released
        # just updating the previously released particles
        wd.prepare_for_model_step(sc, time_step, rel_time)
        wd.weather_elements(sc, time_step, rel_time)
        wd.model_step_is_done(sc)

    @pytest.mark.parametrize("vary_mf", [True, False])
    def test_density_visc_update(self, vary_mf):
        '''
        If no weathering, then density should remain unchanged since mass
        fraction is not changing. Viscosity is also unchanged if no weathering
        '''
        rel_time = datetime.now().replace(microsecond=0)
        (sc, wd) = self.sample_sc_intrinsic(100, rel_time)
        spill = sc.spills[0]
        init_dens = spill.substance.density_at_temp(wd.water.temperature)
        init_visc = spill.substance.kvis_at_temp(wd.water.temperature)

        num = sc.release_elements(default_ts, rel_time)
        wd.initialize_data(sc, num)

        assert np.allclose(sc['density'], init_dens)
        assert np.allclose(sc['viscosity'], init_visc)

        # need this so 'area' computation doesn't break
        # todo: this shouldn't be required, revisit this!
        sc['age'] += default_ts
        if vary_mf:
            self.mock_weather_data(sc, wd)
            self.step(wd, sc, rel_time)

            # viscosity/density
            # should weathered density/viscosity always increase?
            assert np.all(sc['density'] > init_dens)
            assert np.all(sc['viscosity'] > init_visc)
        else:
            # nothing weathered and no emulsion so equations should have
            # produced no change
            self.step(wd, sc, rel_time)
            assert np.allclose(sc['density'], init_dens)
            assert np.allclose(sc['viscosity'], init_visc)

    @pytest.mark.parametrize("vary_frac_water", (False, True))
    def test_density_update_frac_water(self, vary_frac_water):
        rel_time = datetime.now().replace(microsecond=0)
        (sc, wd) = self.sample_sc_intrinsic(100, rel_time)
        spill = sc.spills[0]
        init_dens = spill.substance.density_at_temp(wd.water.temperature)
        init_visc = spill.substance.kvis_at_temp(wd.water.temperature)

        num = sc.release_elements(default_ts, rel_time)
        wd.initialize_data(sc, num)
        assert np.allclose(sc['density'], init_dens)
        assert np.allclose(sc['viscosity'], init_visc)

        # need this so 'area' computation doesn't break
        # todo: this shouldn't be required, revisit this!
        sc['age'] += default_ts
        if vary_frac_water:
            sc['frac_water'][:] = 0.3
            self.step(wd, sc, rel_time)

            exp_res = (wd.water.get('density') * sc['frac_water'] +
                       (1 - sc['frac_water']) * init_dens)
            #assert np.all(sc['density'] == exp_res)
            assert np.allclose(sc['density'], exp_res)
            assert np.all(sc['density'] > init_dens)
            assert np.all(sc['viscosity'] > init_visc)
        else:
            self.step(wd, sc, rel_time)
            assert np.allclose(sc['density'], init_dens)
            assert np.allclose(sc['viscosity'], init_visc)

    def test_density_threshold(self):
        '''
        check that density does not fall below water density
        '''
        rel_time = datetime.now().replace(microsecond=0)
        (sc, wd) = self.sample_sc_intrinsic(100, rel_time)
        num = sc.release_elements(default_ts, rel_time)
        wd.initialize_data(sc, num)

        self.mock_weather_data(sc, wd, 3)
        sc['age'] += default_ts

        # create a mock_water type on which we can set the density - only for
        # this test
        # fixme: can we really not simply override the density of a Water object?
        mock_water = type(str('mock_water'),  # str for py2-3 compatibility
                          (Water,),
                          dict(density=sc['density'][0] - 10))

        # say we are now in 2nd step - no new particles are released
        # so just updating the previously released particles
        wd.water = mock_water()
        self.step(wd, sc, rel_time)
        assert np.all(sc['density'] >= wd.water.density)

    def test_intrinsic_props_vary_num_LEs(self):
        '''
        Release rate in kg/sec is the same; however, vary the number of LEs
        used to model the spill. The following properties should be independent
        of the number of LEs used to model the same spill so the values should
        match.
        '''
        rel_time = datetime.now().replace(microsecond=0)

        # need at least 4 LEs so one released in each timestep
        # to compare the 'mass' in each timestep is equal irrespective of LE
        # using less than 4 LEs will fail some asserts
        (sc1, wd1, spread1) = self.sample_sc_wd_spreading(40, rel_time)
        (sc2, wd2, spread2) = self.sample_sc_wd_spreading(100, rel_time)

        ts = 900
        for i in range(-1, 5):
            curr_time = rel_time + timedelta(seconds=i * ts)
            num1 = sc1.release_elements(ts, curr_time)
            if num1 > 0:
                for w in (wd1, spread1):
                    w.initialize_data(sc1, num1)

            num2 = sc2.release_elements(ts, curr_time)
            if num2 > 0:
                for w in (wd2, spread2):
                    w.initialize_data(sc2, num2)

            for w, sc in zip((wd1, spread1, wd2, spread2),
                             (sc1, sc1, sc2, sc2)):
                # invoke weather_elements in step
                self.step(w, sc, curr_time)

            # for all LEs with same age values should be same
            if num1 == 0:
                assert num2 == 0
            else:
                assert num1 < num2
                assert np.allclose(sc1['mass'].sum(), sc2['mass'].sum(),
                                   atol=1e-6)

            ages = np.unique(sc1['age'])
            for age in ages:
                mask1 = sc1['age'] == age
                mask2 = sc2['age'] == age
                assert np.allclose(sc1['mass'][mask1].sum(),
                                   sc2['mass'][mask2].sum(), atol=1e-6)

                # bulk_init_volume/area
                assert (np.unique(sc1['bulk_init_volume'][mask1]) ==
                        sc1['bulk_init_volume'][mask1][0])
                assert np.allclose(sc1['bulk_init_volume'][mask1][0],
                                   sc2['bulk_init_volume'][mask2][0], atol=1e-6)
                assert np.isclose(sc1['fay_area'][mask1].sum(),
                                  sc2['fay_area'][mask2].sum())

            # model would update the age
            sc1['age'] += ts
            sc2['age'] += ts
            print('Completed step: ', i)

    def test_update_intrinsic_props(self):
        '''
        test multiple spills with same substance
        '''
        weatherers = [WeatheringData(water), FayGravityViscous(water)]

        rel_time = datetime.now().replace(microsecond=0)
        end_time = rel_time + timedelta(hours=1)
        spills = [surface_point_line_spill(100,
                                           (0, 0, 0),
                                           rel_time,
                                           end_release_time=end_time,
                                           amount=100,
                                           units='kg',
                                           substance=test_oil),
                  surface_point_line_spill(50,
                                           (0, 0, 0),
                                           rel_time + timedelta(hours=.25),
                                           substance=test_oil,
                                           amount=100,
                                           units='kg')
                  ]
        sc = SpillContainer()
        sc.spills += spills
        at = dict()
        for w in weatherers:
            at.update(w.array_types)

        sc.prepare_for_model_run(at)

        # test initialization as well
        for w in weatherers:
            w.prepare_for_model_run(sc)

        for val in list(sc.mass_balance.values()):
            assert val == 0.0

        # test initialization as well

        ts = 900
        for i in range(-1, 5):
            curr_time = rel_time + timedelta(seconds=i * ts)
            num_released = sc.release_elements(ts, curr_time)

            if num_released > 0:
                for w in weatherers:
                    w.initialize_data(sc, num_released)

            for w in weatherers:
                self.step(w, sc, curr_time)

            for key, val in sc.mass_balance.items():
                if len(sc) > 0 and key not in ('beached',
                                               'non_weathering',
                                               'off_maps'):
                    assert val > 0
                else:
                    # everything, including avg_density is 0 if nothing is
                    # released
                    assert val == 0.0

            if len(sc) > 0:
                # area arrays initialized correctly
                mask = sc['age'] == 0
                if np.any(~mask):
                    # sc['fay_area'][mask] is initial area of blob
                    # sc['fay_area'][~mask] is area of aged blob
                    assert (sc['fay_area'][mask].sum() !=
                            sc['fay_area'][~mask].sum())

                assert all(sc['fay_area'] > 0)
                assert all(sc['init_mass'] > 0)

                # wd props arrays initialized correctly
                assert all(sc['density'] > 0)
                assert all(sc['viscosity'] > 0)

            sc['age'] += ts     # model would do this operation
            print('Completed step: ', i)

    def test_bulk_init_volume_fay_area_two_spills(self):
        '''
        for two different spills, ensure bulk_init_volume and fay_area is set
        correctly based on the blob of volume released from each spill.
        The volume of the blob should be associated only with its own spill and
        it should be based on water temperature at release time.
        '''
        rel_time = datetime.now().replace(microsecond=0)
        (sc, wd, spread) = self.sample_sc_wd_spreading(100, rel_time)
        sc.spills[0].end_release_time = None
        # add another spill to compare with
        sc.spills += surface_point_line_spill(100, (0, 0, 0),
                                              rel_time,
                                              amount=10,
                                              units='kg',
                                              substance=test_oil)
        op = sc.spills[0].substance
        rho = op.density_at_temp(wd.water.temperature)
        b_init_vol = [spill.get_mass() / rho for spill in sc.spills]

        sc.prepare_for_model_run(wd.array_types)
        wd.prepare_for_model_run(sc)
        spread.prepare_for_model_run(sc)

        print("step 1:", sc['density'])
        # release elements
        num = sc.release_elements(default_ts, rel_time)
        print("step 2:", sc['density'])
        if num > 0:
            for w in (wd, spread):
                w.initialize_data(sc, num)
        print("step 3:", sc['density'])
        print("expected density", rho)

        # bulk_init_volume is set in same order as b_init_vol
        mask = sc['spill_num'] == 0
        assert np.allclose(sc['bulk_init_volume'][mask], b_init_vol[0])
        assert np.allclose(sc['bulk_init_volume'][~mask], b_init_vol[1])
        assert np.all(sc['fay_area'][mask] != sc['fay_area'][~mask])
        i_area = sc['fay_area'].copy()

        # update age and test fay_area update remains unequal
        sc['age'][:] = default_ts

        for w in (wd, spread):
            self.step(w, sc, rel_time)

        assert np.all(sc['fay_area'][mask] != sc['fay_area'][~mask])
        assert np.all(sc['fay_area'] > i_area)

    @log_capture()
    def test_density_error(self, l):
        '''
        log error if init density is less than water
        todo: should this raise a runtime error. May want to change how this
            works
        '''
        l.uninstall()
        rel_time = datetime.now().replace(microsecond=0)
        (sc, wd) = self.sample_sc_intrinsic(100, rel_time)
        wd.water.set('temperature', 288, 'K')
        wd.water.set('salinity', 0, 'psu')
        new_subs = GnomeOil('oil_crude')
        # reset the density
        new_subs.densities = [1004.0]
        new_subs.density_ref_temps = [288.15]

        new_subs.water = wd.water
        sc.spills[0].substance = new_subs
        ats = {}
        ats.update(sc.spills[0].all_array_types)
        ats.update(wd.all_array_types)

        # substance changed - do a rewind
        sc.rewind()
        sc.prepare_for_model_run(ats)
        l.install()

        num = sc.release_elements(default_ts, rel_time)

        # only capture and test density error

        if num > 0:
            wd.initialize_data(sc, num)

        msg = ("{0} will sink at given water temperature: {1} {2}. "
               "Set density to water density".format(new_subs.name,
                                                     288.0,
                                                     'K'))
        l.check_present(('gnome.spills.gnome_oil.GnomeOil',
                 'ERROR',
                 msg))


    def test_no_substance(self):
        '''
        test trajectory only case - in this case there is no data for a
        substance so weatherers should be off but having it on shouldn't break
        anything.
        '''
        rel_time = datetime.now().replace(microsecond=0)
        (sc, wd) = self.sample_sc_intrinsic(100, rel_time)
        sc.spills[0].substance = None
        # substance changed - do a rewind
        sc.rewind()
        sc.prepare_for_model_run(wd.array_types)
        wd.prepare_for_model_run(sc)

        num = sc.release_elements(default_ts, rel_time)
        wd.initialize_data(sc, num)
