'''
test objects defined in intrinsic module
'''
from datetime import datetime, timedelta

import numpy as np
import pytest
from testfixtures import log_capture

from gnome.environment import Water
from gnome.weatherers.intrinsic import WeatheringData
from gnome.spill import point_line_release_spill
from gnome.spill_container import SpillContainer
from gnome.basic_types import oil_status, fate as bt_fate

from ..conftest import test_oil


default_ts = 900  # default timestep for tests
water = Water()


class TestWeatheringData:
    def test_init(self):
        WeatheringData(water)

    def sample_sc_intrinsic(self, num_elements, rel_time):
        '''
        initialize Sample SC and WeatheringData object
        '''
        intrinsic = WeatheringData(Water())
        end_time = rel_time + timedelta(hours=1)
        spills = [point_line_release_spill(num_elements,
                                           (0, 0, 0),
                                           rel_time,
                                           end_release_time=end_time,
                                           amount=100,
                                           units='kg',
                                           substance=test_oil)]
        sc = SpillContainer()
        sc.spills += spills
        sc.prepare_for_model_run(intrinsic.array_types)

        # test initialization as well
        intrinsic.prepare_for_model_run(sc)
        for val in sc.weathering_data.values():
            assert val == 0.0

        # test initialization as well
        return (sc, intrinsic)

    def mock_weather_data(self, sc, intrinsic, zero_elems=5):
        '''
        helper function that mocks a weatherer - like evaporation. It simply
        changes the mass_fraction and updates frac_lost accordingly
        '''
        for substance, data in sc.itersubstancedata(intrinsic.array_types):
            # following simulates weathered/evaporated oil
            data['mass_components'][:, :zero_elems] = 0
            data['mass'][:] = sc['mass_components'].sum(1)
            data['frac_lost'][:] = 1 - data['mass']/data['init_mass']

        sc.update_from_fatedataview()

    @pytest.mark.parametrize("vary_mf", [True, False])
    def test_density_visc_update(self, vary_mf):
        '''
        If no weathering, then density should remain unchanged since mass
        fraction is not changing. Viscosity is also unchanged if no weathering
        '''
        rel_time = datetime.now().replace(microsecond=0)
        (sc, intrinsic) = self.sample_sc_intrinsic(100, rel_time)
        spill = sc.spills[0]
        init_dens = \
            spill.get('substance').get_density(intrinsic.water.temperature)
        init_visc = \
            spill.get('substance').get_viscosity(intrinsic.water.temperature)

        num = sc.release_elements(default_ts, rel_time)
        intrinsic.update(num, sc)
        assert np.allclose(sc['density'], init_dens)
        assert np.allclose(sc['viscosity'], init_visc)

        # need this so 'area' computation doesn't break
        # todo: this shouldn't be required, revisit this!
        sc['age'] += default_ts
        if vary_mf:
            self.mock_weather_data(sc, intrinsic)

            # say we are now in 2nd step - no new particles are released
            # just updating the previously released particles
            intrinsic.update(0, sc)

            # viscosity/density
            # should weathered density/viscosity always increase?
            assert np.all(sc['density'] > init_dens)
            assert np.all(sc['viscosity'] > init_visc)
        else:
            # nothing weathered and no emulsion so equations should have
            # produced no change
            intrinsic.update(0, sc)
            assert np.allclose(sc['density'], init_dens)
            assert np.allclose(sc['viscosity'], init_visc)

    @pytest.mark.parametrize("vary_frac_water", (False, True))
    def test_density_update_frac_water(self, vary_frac_water):
        rel_time = datetime.now().replace(microsecond=0)
        (sc, intrinsic) = self.sample_sc_intrinsic(100, rel_time)
        spill = sc.spills[0]
        init_dens = \
            spill.get('substance').get_density(intrinsic.water.temperature)
        init_visc = \
            spill.get('substance').get_viscosity(intrinsic.water.temperature)

        num = sc.release_elements(default_ts, rel_time)
        intrinsic.update(num, sc)
        assert np.allclose(sc['density'], init_dens)
        assert np.allclose(sc['viscosity'], init_visc)

        # need this so 'area' computation doesn't break
        # todo: this shouldn't be required, revisit this!
        sc['age'] += default_ts
        if vary_frac_water:
            sc['frac_water'][:] = 0.3
            intrinsic.update(0, sc)

            exp_res = (intrinsic.water.get('density') * sc['frac_water'] +
                       (1 - sc['frac_water']) * init_dens)
            assert np.all(sc['density'] == exp_res)
            assert np.all(sc['density'] > init_dens)
            assert np.all(sc['viscosity'] > init_visc)
        else:
            intrinsic.update(0, sc)
            assert np.allclose(sc['density'], init_dens)
            assert np.allclose(sc['viscosity'], init_visc)

    def test_density_threshold(self):
        '''
        check that density does not fall below water density
        '''
        rel_time = datetime.now().replace(microsecond=0)
        (sc, intrinsic) = self.sample_sc_intrinsic(100, rel_time)
        num = sc.release_elements(default_ts, rel_time)
        intrinsic.update(num, sc)
        self.mock_weather_data(sc, intrinsic, 3)
        sc['age'] += default_ts

        # create a mock_water type on which we can set the density - only for
        # this test
        mock_water = type('mock_water',
                          (Water,),
                          dict(density=sc['density'][0] - 10))

        # say we are now in 2nd step - no new particles are released
        # so just updating the previously released particles
        intrinsic.water = mock_water()
        intrinsic.update(0, sc)
        assert np.all(sc['density'] >= intrinsic.water.density)

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
        (sc1, intrinsic1) = self.sample_sc_intrinsic(4, rel_time)
        (sc2, intrinsic2) = self.sample_sc_intrinsic(100, rel_time)

        ts = 900
        for i in range(-1, 5):
            curr_time = rel_time + timedelta(seconds=i * ts)
            num1 = sc1.release_elements(ts, curr_time)
            intrinsic1.update(num1, sc1)

            num2 = sc2.release_elements(ts, curr_time)
            intrinsic2.update(num2, sc2)

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
            print 'Completed step: ', i

    def test_update_intrinsic_props(self):
        intrinsic = WeatheringData(water)

        rel_time = datetime.now().replace(microsecond=0)
        end_time = rel_time + timedelta(hours=1)
        spills = [point_line_release_spill(10,
                                           (0, 0, 0),
                                           rel_time,
                                           end_release_time=end_time,
                                           amount=100,
                                           units='kg',
                                           substance=test_oil),
                  point_line_release_spill(5,
                                           (0, 0, 0),
                                           rel_time + timedelta(hours=.25),
                                           substance=test_oil,
                                           amount=100,
                                           units='kg')
                  ]
        sc = SpillContainer()
        sc.spills += spills
        sc.prepare_for_model_run(intrinsic.array_types)

        # test initialization as well
        intrinsic.prepare_for_model_run(sc)
        for val in sc.weathering_data.values():
            assert val == 0.0

        # test initialization as well

        ts = 900
        for i in range(-1, 5):
            curr_time = rel_time + timedelta(seconds=i * ts)
            num_released = sc.release_elements(ts, curr_time)
            intrinsic.update(num_released, sc)
            for key, val in sc.weathering_data.iteritems():
                if len(sc) > 0 and key not in ('beached', 'non_weathering'):
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

                # intrinsic props arrays initialized correctly
                assert all(sc['density'] > 0)
                assert all(sc['viscosity'] > 0)

            sc['age'] += ts     # model would do this operation
            print 'Completed step: ', i

    def test_update_fate_status(self):
        '''
        test update_fate_status() as it is invoked by Model after elements
        beach
        '''
        rel_time = datetime.now().replace(microsecond=0)
        (sc, intrinsic) = self.sample_sc_intrinsic(100, rel_time)
        num = sc.release_elements(default_ts, rel_time)
        intrinsic.update(num, sc)

        # in next step and set some particles as beached
        beach_mask = np.arange(2, 20, 2)
        sc['status_codes'][beach_mask] = oil_status.on_land

        # during weathering, intrinsic updates fate_status
        intrinsic.update_fate_status(sc)
        assert np.all(sc['fate_status'][beach_mask] == bt_fate.non_weather)
        sc['age'] += default_ts    # model updates age

        # next step, assume no particles released
        intrinsic.update(0, sc)     # no new particles released

        # in the step a subset of particles are reflaoted
        refloat = beach_mask[:-5]
        still_beached = list(set(beach_mask).difference(refloat))
        sc['status_codes'][refloat] = oil_status.in_water
        sc['positions'][refloat, 2] = 4     # just check for surface

        # during weathering, intrinsic updates fate_status
        intrinsic.update_fate_status(sc)
        assert np.all(sc['status_codes'][still_beached] == oil_status.on_land)
        assert np.all(sc['fate_status'][still_beached] == bt_fate.non_weather)
        assert np.all(sc['fate_status'][refloat] == bt_fate.subsurf_weather)
        assert np.all(sc['status_codes'][refloat] == oil_status.in_water)

    def test_bulk_init_volume_fay_area_two_spills(self):
        '''
        for two different spills, ensure bulk_init_volume and fay_aray is set
        correctly based on the blob of volume released from each spill.
        The volume of the blob should be associated only with its own spill and
        it should be based on water temperature at release time.
        '''
        rel_time = datetime.now().replace(microsecond=0)
        (sc, intrinsic) = self.sample_sc_intrinsic(1, rel_time)
        sc.spills[0].set('end_release_time', None)
        sc.spills += point_line_release_spill(1, (0, 0, 0),
                                              rel_time,
                                              amount=10,
                                              units='kg',
                                              substance=test_oil)
        op = sc.spills[0].get('substance')
        rho = op.get_density(intrinsic.water.temperature)
        b_init_vol = [spill.get_mass()/rho for spill in sc.spills]
        print b_init_vol

        sc.prepare_for_model_run(intrinsic.array_types)
        intrinsic.prepare_for_model_run(sc)

        # release elements
        num = sc.release_elements(default_ts, rel_time)
        intrinsic.update(num, sc)

        # bulk_init_volume is set in same order as b_init_vol
        print sc['bulk_init_volume']
        print b_init_vol
        assert np.all(sc['bulk_init_volume'] == b_init_vol)
        assert sc['fay_area'][0] != sc['fay_area'][1]
        i_area = sc['fay_area'].copy()

        # update age and test fay_area update remains unequal
        sc['age'][:] = default_ts
        intrinsic.update(0, sc)
        assert sc['fay_area'][0] != sc['fay_area'][1]
        assert np.all(sc['fay_area'] > i_area)

    @log_capture()
    def test_density_error(self, l):
        '''
        log error if init density is less than water
        '''
        l.uninstall()
        rel_time = datetime.now().replace(microsecond=0)
        (sc, intrinsic) = self.sample_sc_intrinsic(1, rel_time)
        intrinsic.water.set('temperature', 288, 'K')
        intrinsic.water.set('salinity', 0, 'psu')
        new_subs = 'TEXTRACT, STAR ENTERPRISE'
        sc.spills[0].set('substance', new_subs)

        # substance changed - do a rewind
        sc.rewind()
        sc.prepare_for_model_run(intrinsic.array_types)

        num = sc.release_elements(default_ts, rel_time)

        # only capture and test density error
        l.install()
        intrinsic.update(num, sc)
        assert all(sc['fay_area'] == 0.)

        msg = ("{0} will sink at given water temperature: {1} {2}. "
               "Set density to water density".format(new_subs,
                                                     288.0,
                                                     'K'))
        l.check(('gnome.weatherers.intrinsic.WeatheringData',
                 'ERROR',
                 msg))

    def test_no_substance(self):
        rel_time = datetime.now().replace(microsecond=0)
        (sc, intrinsic) = self.sample_sc_intrinsic(1, rel_time)
        sc.spills[0].set('substance', None)
        # substance changed - do a rewind
        sc.rewind()
        sc.prepare_for_model_run(intrinsic.array_types)
        intrinsic.prepare_for_model_run(sc)

        num = sc.release_elements(default_ts, rel_time)
        intrinsic.update(num, sc)
        intrinsic.update(0, sc)
