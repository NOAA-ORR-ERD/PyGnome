'''
test objects defined in intrinsic module
'''
from datetime import datetime, timedelta

import numpy as np
import pytest

from gnome.environment import Water
from gnome.weatherers.intrinsic import FayGravityViscous, IntrinsicProps
from gnome.array_types import area, mol
from gnome.spill.elements import floating_weathering
from gnome.spill import point_line_release_spill
from gnome.spill_container import SpillContainer


# scalar inputs
num_elems = 10
water_viscosity = 0.000001
init_volume = num_elems * 0.01
relative_bouyancy = 0.2

# array inputs
init_vol_array = np.asarray([init_volume/num_elems] * num_elems)
age = np.zeros_like(init_vol_array, dtype=int)
age[:] = 900
d_bouy_array = np.asarray([relative_bouyancy] * num_elems)

water = Water()


class TestFayGravityViscous:
    spread = FayGravityViscous()

    def expected(self, init_vol, p_age):
        '''
        Use this to ensure equations entered correctly in FayGravityViscous
        Equations are a little easier to examine here - but if constants change,
        then this will fail, since this hard codes them. It's just another check
        to help find issues/errors
        '''
        k1 = self.spread.spreading_const[0]
        k2 = self.spread.spreading_const[1]
        g = 9.81
        nu_h2o = water_viscosity
        dbuoy = relative_bouyancy
        A0 = np.pi*(k2**4/k1**2)*(((init_vol)**5*g*dbuoy)/(nu_h2o**2))**(1./6.)

        V0 = init_vol/num_elems
        dFay = k2**2./16.*(g*dbuoy*V0**2/np.sqrt(nu_h2o*p_age))
        dEddy = 0.033*p_age**(4./25)
        p_area = A0 + (dFay + dEddy)*p_age

        return (A0, p_area)

    def test_values(self):
        # compare to expected results
        (A0, p_area) = self.expected(init_volume, age[0])
        init_area = self.spread.init_area(water_viscosity, init_volume,
                                          relative_bouyancy)
        area = self.spread.update_area(water_viscosity,
                                       init_area,
                                       init_vol_array,
                                       d_bouy_array,
                                       age)
        assert A0 == init_area
        assert all(area == p_area)

    def test_scalar_inputs(self):
        '''
        1. init_area is a scalar if init_volume is a scalar
        2. area array returned by update_area has same shape as init_vol_array
        3. for age > 0, area > init_area
        '''
        init_area = self.spread.init_area(water_viscosity, init_volume,
                                          relative_bouyancy)
        area = self.spread.update_area(water_viscosity,
                                       init_area,
                                       init_volume,
                                       relative_bouyancy,
                                       age[0])
        assert np.isscalar(init_area)
        assert np.isscalar(area)
        assert area > init_area

    @pytest.mark.parametrize("i_vol", [init_volume, init_vol_array])
    def test_array_inputs(self, i_vol):
        '''
        1. init_area is an array if init_volume is an array
        2. shape of area array returned by update_area is same as vol_array
        3. for age > 0, area > init_area
        '''
        init_area = self.spread.init_area(water_viscosity, i_vol,
                                          relative_bouyancy)
        area = self.spread.update_area(water_viscosity,
                                       init_area,
                                       init_vol_array,
                                       d_bouy_array,
                                       age)
        if np.isscalar(i_vol):
            assert np.isscalar(init_area)
        else:
            assert init_area.shape == init_vol_array.shape
        assert area.shape == init_vol_array.shape
        assert np.all(area > init_area)

    def test_area_at_age_0(self):
        '''
        For age == 0, update_area == init_area
        For age > 0, update_area > init_area
        '''
        init_vol_array = np.asarray([init_volume] * 10)
        age = np.asarray([0] * 10)
        age[:7] = 1
        d_bouy_array = np.asarray([relative_bouyancy] * 10)

        init_area = self.spread.init_area(water_viscosity, init_vol_array,
                                          relative_bouyancy)
        area = self.spread.update_area(water_viscosity,
                                       init_area,
                                       init_vol_array,
                                       d_bouy_array,
                                       age)
        mask = age == 0
        assert np.all(init_area[mask] == area[mask])
        assert np.all(area[~mask] > init_area[~mask])


class TestIntrinsicProps:
    def test_init(self):
        intrinsic = IntrinsicProps(water)
        assert len(intrinsic.array_types) == 2

        intrinsic = IntrinsicProps(water,
                                   {'area': area})
        for key in ('init_area', 'init_volume', 'relative_bouyancy'):
            assert key in intrinsic.array_types
        assert len(intrinsic.array_types) == 5

        intrinsic.update_array_types({})
        assert 'density' in intrinsic.array_types
        assert len(intrinsic.array_types) == 2

    def test_update_intrinsic_props(self):
        arrays = {'area': area,
                  'mol': mol}
        intrinsic = IntrinsicProps(water, arrays)
        arrays.update(intrinsic.array_types)

        rel_time = datetime.now().replace(microsecond=0)
        end_time = rel_time + timedelta(hours=1)
        et = floating_weathering(substance='ALAMO')
        spills = [point_line_release_spill(10,
                                           (0, 0, 0),
                                           rel_time,
                                           end_release_time=end_time,
                                           element_type=et,
                                           amount=100,
                                           units='kg'),
                  point_line_release_spill(5,
                                           (0, 0, 0),
                                           rel_time + timedelta(hours=.25),
                                           element_type=floating_weathering(substance=et.substance),
                                           amount=100,
                                           units='kg')
                  ]
        sc = SpillContainer()
        sc.spills += spills
        sc.prepare_for_model_run(arrays)
        ts = 900
        for i in range(-1, 5):
            curr_time = rel_time + timedelta(seconds=i * ts)
            num_released = sc.release_elements(ts, curr_time)
            intrinsic.update(num_released, sc)
            mask = sc['age'] == 0
            for val in sc.weathering_data.values():
                if len(sc) > 0:
                    assert val > 0
                else:
                    # everything, including avg_density is 0 if nothing is
                    # released
                    assert val == 0.0

            for at in intrinsic.array_types:
                assert np.all(sc[at] != 0)
                assert all(sc['init_area'][mask] == sc['area'][mask])
                assert all(sc['init_area'][~mask] < sc['area'][~mask])
            sc['age'] += ts     # model would do this operation
