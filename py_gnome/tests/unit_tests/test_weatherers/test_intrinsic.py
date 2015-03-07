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
elem_volume = 0.00001
elem_thick = 965.0    # m - dummy value for testing
elem_rel_bouy = 0.2

water = Water()


def data_arrays(num_elems=10):
    '''
    return a dict of numpy arrays similar to SpillContainer's data_arrays
    All elements are released together so they have same init_volume
    '''
    init_volume = np.asarray([elem_volume*num_elems] * num_elems)
    relative_bouyancy = np.asarray([elem_rel_bouy] * num_elems)
    age = np.zeros_like(init_volume, dtype=int)
    area = np.zeros_like(init_volume)
    init_area = np.zeros_like(init_volume)
    thickness = np.asarray([elem_thick] * num_elems)

    return (init_volume, relative_bouyancy, age, area, init_area, thickness)


class TestFayGravityViscous:
    spread = FayGravityViscous()

    def expected(self, init_vol, p_age):
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
                                  elem_volume * 10,
                                  -elem_rel_bouy)

        with pytest.raises(ValueError):
            'relative_bouyancy >= 0'
            (init_volume,
             relative_bouyancy,
             age, area, init_area, thickness) = data_arrays()
            relative_bouyancy[0] = -relative_bouyancy[0]
            age[:] = 900
            self.spread.update_area(water_viscosity,
                                    init_area,
                                    init_volume,
                                    relative_bouyancy,
                                    age,
                                    thickness,
                                    area,
                                    out=area)
        with pytest.raises(ValueError):
            'age must be > 0'
            (init_volume,
             relative_bouyancy,
             age, area, init_area, thickness) = data_arrays()
            self.spread.update_area(water_viscosity,
                                    init_area,
                                    init_volume,
                                    relative_bouyancy,
                                    age,
                                    thickness,
                                    area,
                                    out=area)

    def test_values(self):
        '''
        Compare output of _init_area and _update_area to expected output
        returned by self.expected() function.
        For _update_area, 'use_list' = True means the inputs are lists instead
        of numpy arrays
        '''
        (init_volume,
         relative_bouyancy,
         age, area, init_area, thickness) = data_arrays()
        init_area[:] = self.spread.init_area(water_viscosity,
                                             init_volume[0],
                                             relative_bouyancy)

        age[:] = 900
        # init_volume[0] and age[0] represents the volume and age of all
        # particles released at once
        # computes the init_area and updated area for particles at 900 sec
        (A0, p_area) = self.expected(init_volume[0], age[0])
        assert all(A0 == init_area)

        self.spread.update_area(water_viscosity,
                                init_area,
                                init_volume,
                                relative_bouyancy,
                                age,
                                thickness,
                                area,
                                out=area)

        assert all(area == p_area)

    def test_minthickness_values(self):
        (init_volume,
         relative_bouyancy,
         age, area, init_area, thickness) = data_arrays()
        init_area[:] = self.spread.init_area(water_viscosity,
                                             sum(init_volume),
                                             relative_bouyancy)
        area[:] = init_area     # initial value

        age[:] = 900
        thickness[[0, 2, 8]] = self.spread.thickness_limit

        self.spread.update_area(water_viscosity,
                                init_area,
                                init_volume,
                                relative_bouyancy,
                                age,
                                thickness,
                                area,
                                out=area)
        mask = thickness > self.spread.thickness_limit
        assert np.all(area[mask] > init_area[mask])
        assert np.all(area[~mask] == init_area[~mask])


class TestWeatheringData:
    def test_init(self):
        WeatheringData(water)

    def sample_sc_intrinsic(self, num_elements, rel_time):
        '''
        initialize Sample SC and WeatheringData object
        '''
        intrinsic = WeatheringData(water)
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

        num = sc.release_elements(900, rel_time)
        intrinsic.update(num, sc)
        assert np.allclose(sc['density'], init_dens)
        assert np.allclose(sc['viscosity'], init_visc)

        # need this so 'area' computation doesn't break
        # todo: this shouldn't be required, revisit this!
        sc['age'] += 900
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
            # nothing weathered so equations should have produced no change
            intrinsic.update(0, sc)
            assert np.allclose(sc['density'], init_dens)
            assert np.allclose(sc['viscosity'], init_visc)

    def test_density_threshold(self):
        '''
        check that density does not fall below water density
        '''
        rel_time = datetime.now().replace(microsecond=0)
        (sc, intrinsic) = self.sample_sc_intrinsic(100, rel_time)
        num = sc.release_elements(900, rel_time)
        intrinsic.update(num, sc)
        self.mock_weather_data(sc, intrinsic, -1)
        sc['age'] += 900

        # say we are now in 2nd step - no new particles are released
        # just updating the previously released particles
        intrinsic.water.density = 960   # force this for test
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
        (sc10, intrinsic10) = self.sample_sc_intrinsic(10, rel_time)
        (sc100, intrinsic100) = self.sample_sc_intrinsic(100, rel_time)

        ts = 900
        for i in range(-1, 5):
            curr_time = rel_time + timedelta(seconds=i * ts)
            num10 = sc10.release_elements(ts, curr_time)
            intrinsic10.update(num10, sc10)

            num100 = sc100.release_elements(ts, curr_time)
            intrinsic100.update(num100, sc100)

            # for all LEs with same age values should be same
            if num10 == 0:
                assert num100 == 0
            else:
                assert num10 < num100
                assert np.allclose(sc10['mass'].sum(), sc100['mass'].sum(),
                                   atol=1e-6)

            ages = np.unique(sc10['age'])
            for age in ages:
                mask10 = sc10['age'] == age
                mask100 = sc100['age'] == age
                assert np.allclose(sc10['mass'][mask10].sum(),
                                   sc100['mass'][mask100].sum(), atol=1e-6)

                # init_volume/init_area/area
                assert (np.unique(sc10['init_volume'][mask10]) ==
                        sc10['init_volume'][mask10][0])
                assert np.allclose(sc10['init_volume'][mask10][0],
                                   sc100['init_volume'][mask100][0], atol=1e-6)
                assert np.allclose(np.unique(sc10['init_area'][mask10]),
                                   sc10['init_area'][mask10][0], atol=1e-6)
                assert np.allclose(sc10['area'][mask10][0],
                                   sc100['area'][mask100][0], atol=1e-6)

                # thickness
                assert (np.unique(sc10['thickness'][mask10]) ==
                        sc10['thickness'][mask10][0])
                assert np.allclose(sc10['thickness'][mask10][0],
                                   sc100['thickness'][mask100][0], atol=1e-6)

            # model would update the age
            sc10['age'] += ts
            sc100['age'] += ts
            print 'Completed step: ', i

    @pytest.mark.parametrize(("s0", "s1"),
                             [("ALASKA NORTH SLOPE",
                               "ALASKA NORTH SLOPE"),
                              ("ALASKA NORTH SLOPE",
                               "ALASKA NORTH SLOPE, OIL & GAS")])
    def test_update_intrinsic_props(self, s0, s1):
        intrinsic = WeatheringData(water)

        rel_time = datetime.now().replace(microsecond=0)
        end_time = rel_time + timedelta(hours=1)
        spills = [point_line_release_spill(10,
                                           (0, 0, 0),
                                           rel_time,
                                           end_release_time=end_time,
                                           amount=100,
                                           units='kg',
                                           substance=s0),
                  point_line_release_spill(5,
                                           (0, 0, 0),
                                           rel_time + timedelta(hours=.25),
                                           substance=s1,
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
            intrinsic.update(num_released, sc)
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
                assert all(sc['init_area'][mask] == sc['area'][mask])
                assert all(sc['init_area'][~mask] < sc['area'][~mask])

                assert all(sc['thickness'] > 0)
                assert all(sc['init_mass'] > 0)
                assert all(sc['relative_bouyancy'] > 0)

                # intrinsic props arrays initialized correctly
                assert all(sc['density'] > 0)
                assert all(sc['viscosity'] > 0)
                if s0 != s1:
                    assert np.any(sc['mass_components'] > 0)
                else:
                    assert np.any(sc['mass_components'] != 0)

            sc['age'] += ts     # model would do this operation
            print 'Completed step: ', i

    def test_update_fate_status(self):
        '''
        test update_fate_status() as it is invoked by Model after elements
        beach
        '''
        rel_time = datetime.now().replace(microsecond=0)
        (sc, intrinsic) = self.sample_sc_intrinsic(100, rel_time)
        num = sc.release_elements(900, rel_time)
        intrinsic.update(num, sc)

        # in next step and set some particles as beached
        beach_mask = np.arange(2, 20, 2)
        sc['status_codes'][beach_mask] = oil_status.on_land

        # during weathering, intrinsic updates fate_status
        intrinsic.update_fate_status(sc)
        assert np.all(sc['fate_status'][beach_mask] == bt_fate.non_weather)
        sc['age'] += 900    # model updates age

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