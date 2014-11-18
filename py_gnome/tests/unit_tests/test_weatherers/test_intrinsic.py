'''
test objects defined in intrinsic module
'''
from datetime import datetime, timedelta

import numpy as np

from gnome.environment import Water
from gnome.weatherers.intrinsic import FayGravityViscous, IntrinsicProps
from gnome.array_types import area, mol
from gnome.spill.elements import floating_weathering
from gnome.spill import point_line_release_spill
from gnome.spill_container import SpillContainer


# scalar inputs
water_viscosity = 0.000001
init_volume = 100 * 0.01
relative_bouyancy = 0.2

# array inputs
vol_array = np.asarray(init_volume).reshape(-1)
age = np.zeros_like(vol_array, dtype=int)
age[:] = 1
d_bouy_array = np.asarray(relative_bouyancy).reshape(-1)

water = Water()


class TestFayGravityViscous:
    def test_scalar_inputs(self):
        '''
        '''
        spread = FayGravityViscous()
        init_area = spread.set_init_area(water_viscosity, init_volume,
                                         relative_bouyancy)
        area = spread.update_area(water_viscosity,
                                  init_area,
                                  vol_array,
                                  d_bouy_array,
                                  age)
        assert np.isscalar(init_area)
        assert area.shape == vol_array.shape
        assert area > init_area

    def test_array_inputs(self):
        spread = FayGravityViscous()
        init_area = spread.set_init_area(water_viscosity,
                                         vol_array,
                                         relative_bouyancy)
        area = spread.update_area(water_viscosity,
                                  init_area,
                                  vol_array,
                                  d_bouy_array,
                                  age)
        assert init_area.shape == vol_array.shape
        assert area.shape == vol_array.shape
        assert np.all(area > init_area)

    def test_area_at_age_0(self):
        vol_array = np.asarray([init_volume] * 10)
        age = np.asarray([0] * 10)
        age[:7] = 1
        d_bouy_array = np.asarray([relative_bouyancy] * 10)

        spread = FayGravityViscous()
        init_area = spread.set_init_area(water_viscosity,
                                         vol_array,
                                         relative_bouyancy)
        area = spread.update_area(water_viscosity,
                                  init_area,
                                  vol_array,
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
