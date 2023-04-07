'''
Test Langmuir() - very simple object with only one method
'''





import sys

from datetime import datetime

import numpy as np
import pytest

from gnome import constants
from gnome.environment import constant_wind, Water
from gnome.weatherers import FayGravityViscous, Langmuir
from .test_cleanup import ObjForTests
from gnome import scripting as gs
from .conftest import test_oil

# scalar inputs - for testing
rel_buoy = 0.2  # relative_buoyancy of oil
num_elems = 10
water_viscosity = 0.000001
bulk_init_vol = 1000.0     # m^3
default_ts = 900  # default timestep for tests


def data_arrays(num_elems=10):
    '''
    return a dict of numpy arrays similar to SpillContainer's data_arrays
    All elements are released together so they have same bulk_init_volume
    '''
    bulk_init_volume = np.asarray([bulk_init_vol] * num_elems)
    age = np.zeros_like(bulk_init_volume, dtype=int)
    area = np.zeros_like(bulk_init_volume)

    return (bulk_init_volume, age, area)


class TestFayGravityViscous(object):
    spread = FayGravityViscous()
    spread._set_thickness_limit(1e-4)    # thickness_limit based on viscosity

    def expected(self, init_vol, p_age, dbuoy=rel_buoy):
        '''
        Use this to ensure equations entered correctly in FayGravityViscous
        Equations are easier to examine here
        '''
        k1 = self.spread.spreading_const[0]
        k2 = self.spread.spreading_const[1]
        k_nu = self.spread.spreading_const[2]
        g = constants.gravity
        nu_h2o = water_viscosity

        A0 = (np.pi *
              (k2 ** 4 / k1 ** 2) *
              (((init_vol) ** 5 * g * dbuoy) / (nu_h2o ** 2)) ** (1. / 6.))

        p_area = A0
        for i in range(0, int(p_age/default_ts)):
            C = (np.pi * k_nu ** 2 * (init_vol ** 2 * g * dbuoy / np.sqrt(nu_h2o)) ** (1. / 3.))
            K = 4 * np.pi * 2 * .033

            blob_area_fgv = .5 * (C**2 / p_area) * default_ts
            blob_area_diffusion = ((7. / 6.) * K * (p_area / K) ** (1. / 7.)) * default_ts
            p_area = p_area + blob_area_fgv + blob_area_diffusion
        '''
        p_area = (np.pi *
                  # correct k_nu, Spreading Law coefficient -- Eq.(6.14), 11/23/2021
                  #k2 ** 2 *
                  k_nu ** 2 *
                  (init_vol ** 2 * g * dbuoy * p_age ** 1.5) ** (1. / 3.) /
                  (nu_h2o ** (1. / 6.)))
        '''
        return (A0, p_area)

    def test_exception_init_area(self):
        '''
        if relative_bouyancy is < 0, it just raises an exception
        '''
        # kludge for py2/3 different behaviour with pow() or a negative number
        Err = ValueError if sys.version_info.major == 2 else TypeError
        with pytest.raises(Err):
            'relative_bouyancy >= 0'
            self.spread.init_area(water_viscosity, -rel_buoy, data_arrays(num=1))

    def test_exception_update_area(self):
        '''
        if relative_bouyancy is < 0, it just raises an exception
        '''

        with pytest.raises(ValueError):
            'age must be > 0'
            bulk_init_volume, relative_bouyancy, age, area = data_arrays()

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
        bulk_init_volume, age, area = data_arrays(num)

        area[:] = self.spread.init_area(water_viscosity,
                                        rel_buoy,
                                        bulk_init_volume[0]) / len(area)

        # bulk_init_volume[0] and age[0] represents the volume and age of all
        # particles released at once
        # computes the init_area and updated area for particles at 900 sec
        (A0, p_area) = self.expected(bulk_init_volume[0], default_ts)
        assert A0 == area.sum()

        vol_frac_le = np.zeros_like(bulk_init_volume)
        vol_frac_le[:] = 1.0 / num
        max_area_le = (bulk_init_volume / self.spread.thickness_limit) * vol_frac_le
        age[:] = 900
        for i in range(0, int(age[0]/default_ts)):
            area = self.spread.update_area(water_viscosity,
                                    rel_buoy,
                                    bulk_init_volume, #[0],
                                    area,
                                    max_area_le,
                                    default_ts,
                                    vol_frac_le, #[0],
                                    age)

        assert np.isclose(area.sum(), p_area)

    def test_values_vary_age(self):
        '''
        test update_area works correctly for a continuous spill with varying
        age array
        '''
        bulk_init_volume, age, area = data_arrays(10)
        a0, area_900 = self.expected(bulk_init_volume[0], 900)

        age[0::2] = 900
        area[0::2] = a0 / len(area[0::2])  # initialize else divide by 0 error

        a0, area_1800 = self.expected(bulk_init_volume[1], 1800)

        age[1::2] = 1800
        area[1::2] = a0 / len(area[1::2])  # initialize else divide by 0 error

        vol_frac_le = np.zeros_like(bulk_init_volume)
        vol_frac_le[0::2] = 1.0 / len(area[0::2])
        vol_frac_le[1::2] = 1.0 / len(area[1::2])

        max_area_le = (bulk_init_volume / self.spread.thickness_limit) * vol_frac_le
        # now invoke update_area
        for age_le in np.unique(age):
            mask = age == age_le
            for i in range(0, int(age_le/default_ts)):
                    area[mask] = self.spread.update_area(water_viscosity,
                                                rel_buoy,
                                                bulk_init_volume[mask],
                                                area[mask],
                                                max_area_le[mask],
                                                default_ts,
                                                vol_frac_le[mask],
                                                age[mask])


        assert np.isclose(area[0::2].sum(), area_900)
        assert np.isclose(area[1::2].sum(), area_1800)

    def test_values_vary_age_bulk_init_vol(self):
        '''
        vary bulk_init_vol and age
        '''
        bulk_init_volume, age, area = data_arrays(10)

        age[0::2] = 900
        bulk_init_volume[0::2] = 60

        a0, area_900 = self.expected(bulk_init_volume[0], age[0], rel_buoy)
        area[0::2] = a0 / len(area[0::2])  # initialize else divide by 0 error

        age[1::2] = 1800
        a0, area_1800 = self.expected(bulk_init_volume[1], age[1])

        area[1::2] = a0 / len(area[1::2])  # initialize else divide by 0 error

        vol_frac_le = np.zeros_like(bulk_init_volume)
        vol_frac_le[0::2] = 1.0 / len(area[0::2])
        vol_frac_le[1::2] = 1.0 / len(area[1::2])

        max_area_le = (bulk_init_volume / self.spread.thickness_limit) * vol_frac_le
        # now invoke update_area
        for age_le in np.unique(age):
            mask = age == age_le
            for i in range(0, int(age_le/default_ts)):
                    area[mask] = self.spread.update_area(water_viscosity,
                                                rel_buoy,
                                                bulk_init_volume[mask],
                                                area[mask],
                                                max_area_le[mask],
                                                default_ts,
                                                vol_frac_le[mask],
                                                age[mask])

        assert np.isclose(area[0::2].sum(), area_900)
        assert np.isclose(area[1::2].sum(), area_1800)

    def test_minthickness_values(self):
        '''
        tests that when blob reaches minimum thickness, area no longer changes
        '''
        bulk_init_volume, age, area = data_arrays()

        area[:] = self.spread.init_area(water_viscosity,
                                        rel_buoy,
                                        bulk_init_volume[0])

        # elements with same age have the same area since area is computed for
        # blob released at given time. So age must be different to
        # differentiate two blobs
        time = self.spread._time_to_reach_max_area(water_viscosity,
                                                   rel_buoy,
                                                   bulk_init_volume[0])
        age[:4] = np.ceil(time)
        # divide max blob area into 4 LEs
        i_area = bulk_init_volume[0] / self.spread.thickness_limit / 4

        age[4:] = 900

        vol_frac_le = np.zeros_like(bulk_init_volume)
        vol_frac_le[:4] = 1.0 / len(age[:4])
        vol_frac_le[4:] = 1.0 / len(age[4:])

        max_area_le = (bulk_init_volume / self.spread.thickness_limit) * vol_frac_le

        # now invoke update_area
        for age_le in np.unique(age):
            mask = age == age_le
            for i in range(0, int(age_le/default_ts)):
                    area[mask] = self.spread.update_area(water_viscosity,
                                                rel_buoy,
                                                bulk_init_volume[mask],
                                                area[mask],
                                                max_area_le[mask],
                                                default_ts,
                                                vol_frac_le[mask],
                                                age[mask])

        assert np.all(area[:4] == i_area)
        assert np.all(area[4:] < i_area)

    def test_two_spills(self):
        start_time = gs.asdatetime("2015-05-14")

        model_two_spills = gs.Model(start_time=start_time,
                     duration=gs.days(10.0),
                     time_step=15 * 60,
                     uncertain=False)

        model_one_spill1 = gs.Model(start_time=start_time,
                     duration=gs.days(10.0),
                     time_step=15 * 60,
                     uncertain=False)

        model_one_spill2 = gs.Model(start_time=start_time,
                     duration=gs.days(10.0),
                     time_step=15 * 60,
                     uncertain=False)

        spill_0 = gs.surface_point_line_spill(num_elements=16,
                                        start_position=(-164.791878561,
                                                        69.6252597267, 0.0),
                                        release_time=start_time,
                                        end_release_time=start_time + gs.hours(1.0),
                                        amount=100,
                                        substance=test_oil,
                                        units='bbl')

        spill_1 = gs.surface_point_line_spill(num_elements=16,
                                        start_position=(-164.791878561,
                                                        69.6252597267, 0.0),
                                        release_time=start_time,
                                        end_release_time=start_time + gs.hours(1.0),
                                        amount=200,
                                        substance=test_oil,
                                        units='bbl')

        spill_2 = gs.surface_point_line_spill(num_elements=16,
                                        start_position=(-164.791878561,
                                                        69.6252597267, 0.0),
                                        release_time=start_time,
                                        end_release_time=start_time + gs.hours(1.0),
                                        amount=100,
                                        substance=test_oil,
                                        units='bbl')

        spill_3 = gs.surface_point_line_spill(num_elements=16,
                                        start_position=(-164.791878561,
                                                        69.6252597267, 0.0),
                                        release_time=start_time,
                                        end_release_time=start_time + gs.hours(1.0),
                                        amount=200,
                                        substance=test_oil,
                                        units='bbl')

        model_two_spills.spills += spill_0
        model_two_spills.spills += spill_1

        model_two_spills.environment += constant_wind(5., 0, 'm/s')
        model_two_spills.weatherers += FayGravityViscous(Water(273.15 + 20.0))

        model_one_spill1.spills += spill_2

        model_one_spill1.environment += constant_wind(5., 0, 'm/s')
        model_one_spill1.weatherers += FayGravityViscous(Water(273.15 + 20.0))

        model_one_spill2.spills += spill_3

        model_one_spill2.environment += constant_wind(5., 0, 'm/s')
        model_one_spill2.weatherers += FayGravityViscous(Water(273.15 + 20.0))

        for i in range(0,10):
            next(model_two_spills)
            next(model_one_spill1)
            next(model_one_spill2)

        s_mask = model_two_spills.spills._spill_container['spill_num'] == 0
        area_0 = sum(model_two_spills.spills._spill_container['area'][s_mask])

        s_mask = model_two_spills.spills._spill_container['spill_num'] == 1
        area_1 = sum(model_two_spills.spills._spill_container['area'][s_mask])

        s_mask = model_one_spill1.spills._spill_container['spill_num'] == 0
        area_2 = sum(model_one_spill1.spills._spill_container['area'][s_mask])

        s_mask = model_one_spill2.spills._spill_container['spill_num'] == 0
        area_3 = sum(model_one_spill2.spills._spill_container['area'][s_mask])

        assert area_0 == area_2
        assert area_1 == area_3

class TestLangmuir(ObjForTests):
    thick = 1e-4
    wind = constant_wind(5, 0)
    model_time = datetime(2015, 1, 1, 12, 0)
    water = Water()

    lang = Langmuir(water, wind)

    vmin, vmax = lang._wind_speed_bound(rel_buoy, thick)

    sc, weatherers, environment = ObjForTests.mk_test_objs(water)

    def test_init(self):
        langmuir = Langmuir(self.water, self.wind)
        assert langmuir.wind is self.wind

    @pytest.mark.parametrize(("langmuir", "speed", "exp_bound"),
                             [(lang, vmin - 0.01 * vmin, 1.0),
                              (lang, vmax + 0.01 * vmax, 0.1)])
    def test_speed_bounds(self, langmuir, speed, exp_bound):
        '''
        Check that the input speed for Langmuir object returns frac_coverage
        within bounds:
            0.1 <= frac_cov <= 1.0
        '''
        self.lang.wind.timeseries = (self.lang.wind.timeseries['time'][0],
                                     (speed, 0.0))

        # rel_buoy is always expected to be a numpy array
        frac_cov = langmuir._get_frac_coverage(np.array([0, 0]),
                                               self.model_time,
                                               np.asarray([rel_buoy]),
                                               self.thick)
        assert frac_cov == exp_bound

    @pytest.mark.skipif(reason='serialization for weatherers overall '
                               'needs review')
    def test_update_from_dict(self):
        '''
        just a simple test to ensure schema/serialize/deserialize is correctly
        setup
        '''
        j = self.l.serialize()
        j['wind']['timeseries'][0] = (j['wind']['timeseries'][0][0],
                                      (j['wind']['timeseries'][0][1][0] + 1, 0)
                                      )

        updated = self.l.update_from_dict(Langmuir.deserialize(j))

        assert updated
        assert self.l.serialize() == j

    # langmuir temporarily turned off
    #@pytest.mark.xfail
    def test_weather_elements(self):
        '''
        use ObjMakeTests from test_cleanup to setup test
        Langmuir weather_elements must be called after weather elements
        for other objects
        '''
        langmuir = Langmuir(self.water, constant_wind(5., 0.))

        self.prepare_test_objs(langmuir.array_types)
        langmuir.prepare_for_model_run(self.sc)

        # create WeatheringData object, initialize instantaneously released
        # elements
        model_time = self.sc.spills[0].release_time
        time_step = 900.
        self.release_elements(time_step, model_time, self.environment)
        self.step(langmuir, time_step, model_time)

        assert langmuir.active
        assert np.all(self.sc['area'] < self.sc['fay_area'])
        assert np.all(self.sc['frac_coverage'] < 1.0)
