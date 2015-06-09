'''
Test Langmuir() - very simple object with only one method
'''
from datetime import datetime, timedelta

import numpy as np
import pytest

from gnome import constants
from gnome.environment import constant_wind, Water
from gnome.weatherers.spreading import FayGravityViscous
from gnome.weatherers import Langmuir, WeatheringData
from gnome.spill.elements import floating
#from ..conftest import (sample_sc_release,
#                        test_oil)
from .test_cleanup import ObjForTests

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


class TestFayGravityViscous:
    spread = FayGravityViscous()
    spread._set_thickness_limit(1e-4)    # thickness_limit based on viscosity

    def expected(self, init_vol, p_age, dbuoy=rel_buoy):
        '''
        Use this to ensure equations entered correctly in FayGravityViscous
        Equations are easier to examine here
        '''
        k1 = self.spread.spreading_const[0]
        k2 = self.spread.spreading_const[1]
        g = constants.gravity
        nu_h2o = water_viscosity
        A0 = np.pi*(k2**4/k1**2)*(((init_vol)**5*g*dbuoy)/(nu_h2o**2))**(1./6.)

        p_area = (np.pi*k2**2 * (init_vol**2 * g * dbuoy * p_age**1.5)**(1./3)
                  / (nu_h2o**(1./6.)))

        return (A0, p_area)

    def test_exceptions(self):
        '''
        if relative_bouyancy is < 0, it just raises an exception
        '''
        with pytest.raises(ValueError):
            'relative_bouyancy >= 0'
            self.spread.init_area(water_viscosity,
                                  -rel_buoy,
                                  bulk_init_vol)

        with pytest.raises(ValueError):
            'age must be > 0'
            (bulk_init_volume, relative_bouyancy, age, area) = \
                data_arrays()
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
        (bulk_init_volume, age, area) = \
            data_arrays(num)
        area[:] = self.spread.init_area(water_viscosity,
                                        rel_buoy,
                                        bulk_init_volume[0])/len(area)

        # bulk_init_volume[0] and age[0] represents the volume and age of all
        # particles released at once
        # computes the init_area and updated area for particles at 900 sec
        (A0, p_area) = self.expected(bulk_init_volume[0], default_ts)
        assert A0 == area.sum()

        age[:] = 900
        self.spread.update_area(water_viscosity,
                                rel_buoy,
                                bulk_init_volume,
                                area,
                                age)

        assert np.isclose(area.sum(), p_area)

    def test_values_vary_age(self):
        '''
        test update_area works correctly for a continuous spill with varying
        age array
        '''
        (bulk_init_volume, age, area) = \
            data_arrays(10)
        (a0, area_900) = self.expected(bulk_init_volume[0], 900)
        age[0::2] = 900
        area[0::2] = a0/len(area[0::2])  # initialize else divide by 0 error

        (a0, area_1800) = self.expected(bulk_init_volume[1], 1800)
        age[1::2] = 1800
        area[1::2] = a0/len(area[1::2])  # initialize else divide by 0 error

        # now invoke update_area
        area[:] = self.spread.update_area(water_viscosity,
                                          rel_buoy,
                                          bulk_init_volume,
                                          area,
                                          age)
        assert np.isclose(area[0::2].sum(), area_900)
        assert np.isclose(area[1::2].sum(), area_1800)

    def test_values_vary_age_bulk_init_vol(self):
        '''
        vary bulk_init_vol and age
        '''
        (bulk_init_volume, age, area) = \
            data_arrays(10)
        age[0::2] = 900
        bulk_init_volume[0::2] = 60
        (a0, area_900) = self.expected(bulk_init_volume[0], age[0], rel_buoy)
        area[0::2] = a0/len(area[0::2])  # initialize else divide by 0 error

        age[1::2] = 1800
        (a0, area_1800) = self.expected(bulk_init_volume[1], age[1])
        area[1::2] = a0/len(area[1::2])  # initialize else divide by 0 error

        # now invoke update_area
        area[:] = self.spread.update_area(water_viscosity,
                                          rel_buoy,
                                          bulk_init_volume,
                                          area,
                                          age)
        assert np.isclose(area[0::2].sum(), area_900)
        assert np.isclose(area[1::2].sum(), area_1800)

    def test_minthickness_values(self):
        '''
        tests that when blob reaches minimum thickness, area no longer changes
        '''
        (bulk_init_volume, age, area) = \
            data_arrays()
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
        i_area = bulk_init_volume[0]/self.spread.thickness_limit/4

        age[4:] = 900

        self.spread.update_area(water_viscosity,
                                rel_buoy,
                                bulk_init_volume,
                                area,
                                age)
        assert np.all(area[:4] == i_area)
        assert np.all(area[4:] < i_area)


class TestLangmuir:
    thick = 1e-4
    wind = constant_wind(5, 0)
    model_time = datetime(2015, 1, 1, 12, 0)
    water = Water()

    l = Langmuir(water, wind)
    (vmin, vmax) = l._wind_speed_bound(rel_buoy, thick)

    def test_init(self):
        l = Langmuir(self.water, self.wind)
        assert l.wind is self.wind

    @pytest.mark.parametrize(("l", "speed", "exp_bound"),
                             [(l, vmin - 0.01 * vmin, 1.0),
                              (l, vmax + 0.01 * vmax, 0.1)])
    def test_speed_bounds(self, l, speed, exp_bound):
        '''
        Check that the input speed for Langmuir object returns frac_coverage
        within bounds:
            0.1 <= frac_cov <= 1.0
        '''
        self.l.wind.timeseries = (self.l.wind.timeseries['time'][0],
                                  (speed, 0.0))
        frac_cov = l._get_frac_coverage(self.model_time, rel_buoy, self.thick)
        assert frac_cov == exp_bound

    def test_update_from_dict(self):
        '''
        just a simple test to ensure schema/serialize/deserialize is correclty
        setup
        '''
        j = self.l.serialize()
        j['wind']['timeseries'][0] = \
            (j['wind']['timeseries'][0][0],
             (j['wind']['timeseries'][0][1][0] + 1, 0))
        updated = self.l.update_from_dict(Langmuir.deserialize(j))
        assert updated
        assert self.l.serialize() == j

    @pytest.mark.xfail(reason="still working on test")
    def test_weather_elements(self):
        l = Langmuir(self.water, constant_wind(5., 0.))
        #(self.sc, self.weatherers) = ObjForTests.mk_test_objs()

        # create WeatheringData object, initialize instantaneously released
        # elements
        arrays = l.array_types
        wd = WeatheringData(self.water)
        arrays.update(wd.array_types)
        et = floating(substance=test_oil)
        time_step = 15. * 60
        sc = sample_sc_release(num_elements=1,
                               element_type=et,
                               arr_types=arrays,
                               time_step=time_step)

        # prepare_for_model_run will happen before release; however, this is
        # convenient and it doesn't break anything so call
        # prepare_for_model_run after the release
        wd.prepare_for_model_run(sc)
        wd.update(sc.num_released, sc)
        water_kvis = self.water.get('kinematic_viscosity')
        # update the age of one of the LEs by n_age
        n_age = (wd.spreading.
                 _time_to_reach_max_area(water_kvis,
                                         wd._init_relative_buoyancy,
                                         sc['bulk_init_volume'][0]))

        model_time = sc.spills[0].get('release_time')

        def step():
            l.prepare_for_model_step(sc, time_step, model_time)
            assert l.active
            l.weather_elements(sc, time_step, model_time)
            l.model_step_is_done()

        l.prepare_for_model_run(sc)
        # do two steps
        step()
        assert np.all(sc['area'] == sc['fay_area'])
        # age LEs such that max_area is reached but no new LEs released
        sc['age'][:] += np.ceil(n_age)
        model_time += timedelta(seconds=time_step)
        num = sc.release_elements(default_ts, model_time)
        wd.update(num, sc)

        step()
        assert np.all(sc['area'] < sc['fay_area'])
