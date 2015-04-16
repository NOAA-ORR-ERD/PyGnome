'''
test objects defined in intrinsic module
'''
from datetime import datetime, timedelta

import numpy as np
import pytest

from gnome import constants
from gnome.environment import Water
from gnome.weatherers.intrinsic import FayGravityViscous, WeatheringData
from gnome.spill import point_line_release_spill
from gnome.spill_container import SpillContainer
from gnome.basic_types import oil_status, fate as bt_fate

from ..conftest import test_oil


# scalar inputs
num_elems = 10
water_viscosity = 0.000001
bulk_init_vol = 100.0     # m^3
elem_rel_bouy = 0.2
default_ts = 900  # default timestep for tests

water = Water()


def data_arrays(num_elems=10):
    '''
    return a dict of numpy arrays similar to SpillContainer's data_arrays
    All elements are released together so they have same bulk_init_volume
    '''
    bulk_init_volume = np.asarray([bulk_init_vol] * num_elems)
    relative_bouyancy = np.asarray([elem_rel_bouy] * num_elems)
    age = np.zeros_like(bulk_init_volume, dtype=int)
    area = np.zeros_like(bulk_init_volume)

    return (bulk_init_volume, relative_bouyancy, age, area)


# todo: update tests for new spreading model
@pytest.mark.skipif
class TestFayGravityViscous:
    spread = FayGravityViscous()

    def expected(self, init_vol, p_age, elem_rel_bouy=elem_rel_bouy):
        '''
        Use this to ensure equations entered correctly in FayGravityViscous
        Equations are easier to examine here
        '''
        k1 = self.spread.spreading_const[0]
        k2 = self.spread.spreading_const[1]
        g = constants.gravity
        nu_h2o = water_viscosity
        dbuoy = elem_rel_bouy
        A0 = np.pi*(k2**4/k1**2)*(((init_vol)**5*g*dbuoy)/(nu_h2o**2))**(1./6.)

        dFay = k2**2./16.*(g*dbuoy*init_vol**2/np.sqrt(nu_h2o*p_age))
        dEddy = 0.033*p_age**(4./25)
        p_area = A0 + (dFay + dEddy)*p_age

        return (A0, p_area)

    def test_exceptions(self):
        '''
        if relative_bouyancy is < 0, it just raises an exception
        '''
        with pytest.raises(ValueError):
            'relative_bouyancy >= 0'
            self.spread.init_area(water_viscosity,
                                  -elem_rel_bouy,
                                  bulk_init_vol)

        with pytest.raises(ValueError):
            'relative_bouyancy >= 0'
            (bulk_init_volume, relative_bouyancy, age, area) = data_arrays()
            relative_bouyancy[0] = -relative_bouyancy[0]
            age[:] = 900
            self.spread.update_area(water_viscosity,
                                    relative_bouyancy,
                                    bulk_init_volume,
                                    area,
                                    age)
        with pytest.raises(ValueError):
            'age must be > 0'
            (bulk_init_volume, relative_bouyancy, age, area) = data_arrays()
            self.spread.update_area(water_viscosity,
                                    relative_bouyancy,
                                    bulk_init_volume,
                                    age,
                                    area)

    @pytest.mark.parametrize("num", (1, 10))
    def test_values_same_age(self, num):
        '''
        Compare output of _init_area and _update_thickness to expected output
        returned by self.expected() function.
        '''
        (bulk_init_volume, relative_bouyancy, age, area) = data_arrays(num)
        area[:] = self.spread.init_area(water_viscosity,
                                        elem_rel_bouy,
                                        bulk_init_volume[0])/len(area)

        # bulk_init_volume[0] and age[0] represents the volume and age of all
        # particles released at once
        # computes the init_area and updated area for particles at 900 sec
        (A0, p_area) = self.expected(bulk_init_volume[0], default_ts)
        assert A0 == area.sum()

        age[:] = 900
        self.spread.update_area(water_viscosity,
                                relative_bouyancy,
                                bulk_init_volume,
                                area,
                                age)

        assert np.isclose(area.sum(), p_area)

    def test_values_vary_age(self):
        '''
        test update_area works correctly for a continuous spill with varying
        age array
        '''
        (bulk_init_volume, relative_bouyancy, age, area) = \
            data_arrays(10)
        (a0, area_900) = self.expected(bulk_init_volume[0], 900)
        age[0::2] = 900
        area[0::2] = a0/len(area[0::2])  # initialize else divide by 0 error

        (a0, area_1800) = self.expected(bulk_init_volume[1], 1800)
        age[1::2] = 1800
        area[1::2] = a0/len(area[1::2])  # initialize else divide by 0 error

        # now invoke update_area
        area[:] = self.spread.update_area(water_viscosity,
                                          relative_bouyancy,
                                          bulk_init_volume,
                                          area,
                                          age)
        assert np.isclose(area[0::2].sum(), area_900)
        assert np.isclose(area[1::2].sum(), area_1800)

    def test_values_vary_age_bulk_init_vol_rel_bouy(self):
        '''
        vary bulk_init_vol and age
        '''
        (bulk_init_volume, relative_bouyancy, age, area) = \
            data_arrays(10)
        relative_bouyancy[0] = 0.1
        age[0::2] = 900
        bulk_init_volume[0::2] = 60
        (a0, area_900) = self.expected(bulk_init_volume[0], age[0],
                                       np.mean(relative_bouyancy[0::2]))
        area[0::2] = a0/len(area[0::2])  # initialize else divide by 0 error

        age[1::2] = 1800
        (a0, area_1800) = self.expected(bulk_init_volume[1], age[1])
        area[1::2] = a0/len(area[1::2])  # initialize else divide by 0 error

        # now invoke update_area
        area[:] = self.spread.update_area(water_viscosity,
                                          relative_bouyancy,
                                          bulk_init_volume,
                                          area,
                                          age)
        assert np.isclose(area[0::2].sum(), area_900)
        assert np.isclose(area[1::2].sum(), area_1800)

    def test_minthickness_values(self):
        '''
        tests that when blob reaches minimum thickness, area no longer changes
        '''
        (bulk_init_volume, relative_bouyancy, age, area) = data_arrays()
        area[:] = self.spread.init_area(water_viscosity,
                                        relative_bouyancy[0],
                                        bulk_init_volume[0])

        # assume first 4 elements are released together in one blob
        area[:4] = (bulk_init_volume[0]/self.spread.thickness_limit)/4
        i_area = area[0]

        # elements with same age have the same area since area is computed for
        # blob released at given time. So age must be different to
        # differentiate two blobs
        age[:4] = 1800
        age[4:] = 900

        self.spread.update_area(water_viscosity,
                                relative_bouyancy,
                                bulk_init_volume,
                                area,
                                age)
        assert np.all(area[:4] == i_area)
        assert np.all(area[4:] > i_area)


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
        intrinsic.initialize(sc)
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
        intrinsic.update(num, sc, default_ts)
        assert np.allclose(sc['density'], init_dens)
        assert np.allclose(sc['viscosity'], init_visc)

        # need this so 'area' computation doesn't break
        # todo: this shouldn't be required, revisit this!
        sc['age'] += default_ts
        if vary_mf:
            self.mock_weather_data(sc, intrinsic)

            # say we are now in 2nd step - no new particles are released
            # just updating the previously released particles
            intrinsic.update(0, sc, default_ts)

            # viscosity/density
            # should weathered density/viscosity always increase?
            assert np.all(sc['density'] > init_dens)
            assert np.all(sc['viscosity'] > init_visc)
        else:
            # nothing weathered so equations should have produced no change
            intrinsic.update(0, sc, default_ts)
            assert np.allclose(sc['density'], init_dens)
            assert np.allclose(sc['viscosity'], init_visc)

    def test_density_threshold(self):
        '''
        check that density does not fall below water density
        '''
        rel_time = datetime.now().replace(microsecond=0)
        (sc, intrinsic) = self.sample_sc_intrinsic(100, rel_time)
        num = sc.release_elements(default_ts, rel_time)
        intrinsic.update(num, sc, default_ts)
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
        intrinsic.update(0, sc, default_ts)
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
            intrinsic1.update(num1, sc1, ts)

            num2 = sc2.release_elements(ts, curr_time)
            intrinsic2.update(num2, sc2, ts)

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
        intrinsic.initialize(sc)
        for val in sc.weathering_data.values():
            assert val == 0.0

        # test initialization as well

        ts = 900
        for i in range(-1, 5):
            curr_time = rel_time + timedelta(seconds=i * ts)
            num_released = sc.release_elements(ts, curr_time)
            intrinsic.update(num_released, sc, ts)
            for key, val in sc.weathering_data.iteritems():
                if len(sc) > 0 and key != 'beached':
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
        intrinsic.update(num, sc, default_ts)

        # in next step and set some particles as beached
        beach_mask = np.arange(2, 20, 2)
        sc['status_codes'][beach_mask] = oil_status.on_land

        # during weathering, intrinsic updates fate_status
        intrinsic.update_fate_status(sc)
        assert np.all(sc['fate_status'][beach_mask] == bt_fate.non_weather)
        sc['age'] += default_ts    # model updates age

        # next step, assume no particles released
        intrinsic.update(0, sc, default_ts)     # no new particles released

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
        intrinsic.initialize(sc)

        # release elements
        num = sc.release_elements(default_ts, rel_time)
        intrinsic.update(num, sc, default_ts)

        # bulk_init_volume is set in same order as b_init_vol
        print sc['bulk_init_volume']
        print b_init_vol
        assert np.all(sc['bulk_init_volume'] == b_init_vol)
        assert sc['fay_area'][0] != sc['fay_area'][1]
        i_area = sc['fay_area'].copy()

        # update age and test fay_area update remains unequal
        sc['age'][:] = default_ts
        intrinsic.update(0, sc, default_ts)
        assert sc['fay_area'][0] != sc['fay_area'][1]
        assert np.all(sc['fay_area'] > i_area)
