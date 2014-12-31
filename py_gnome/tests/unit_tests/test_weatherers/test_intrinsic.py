'''
test objects defined in intrinsic module
'''
from datetime import datetime, timedelta

import numpy as np
import pytest

from gnome.environment import Water, constants
from gnome.weatherers.intrinsic import FayGravityViscous, IntrinsicProps
from gnome.array_types import area, mol
from gnome.spill import point_line_release_spill
from gnome.spill_container import SpillContainer


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
    '''
    init_volume = np.asarray([elem_volume] * num_elems)
    relative_bouyancy = np.asarray([elem_rel_bouy] * num_elems)
    age = np.zeros_like(init_volume, dtype=int)
    area = np.zeros_like(init_volume)
    init_area = np.zeros_like(init_volume)
    thickness = np.asarray([elem_thick] * num_elems)

    return (init_volume, relative_bouyancy, age, area, init_area, thickness)


class TestFayGravityViscous:
    spread = FayGravityViscous()

    def expected(self, init_vol, p_age, num_elems):
        '''
        Use this to ensure equations entered correctly in FayGravityViscous
        Equations are easier to examine here
        '''
        k1 = self.spread.spreading_const[0]
        k2 = self.spread.spreading_const[1]
        g = constants['gravity']
        nu_h2o = water_viscosity
        dbuoy = elem_rel_bouy
        A0 = np.pi*(k2**4/k1**2)*(((init_vol)**5*g*dbuoy)/(nu_h2o**2))**(1./6.)

        V0 = init_vol/num_elems
        dFay = k2**2./16.*(g*dbuoy*V0**2/np.sqrt(nu_h2o*p_age))
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
                                             sum(init_volume),
                                             relative_bouyancy)

        age[:] = 900
        (A0, p_area) = self.expected(sum(init_volume),
                                     age[0],
                                     len(init_volume))
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


class TestIntrinsicProps:
    def test_init(self):
        IntrinsicProps(water)

    @pytest.mark.parametrize(("s0", "s1"), [("ALAMO", "ALAMO"),
                                            ("ALAMO", "AGUA DULCE")])
    def test_update_intrinsic_props(self, s0, s1):
        intrinsic = IntrinsicProps(water)

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
            for val in sc.weathering_data.values():
                if len(sc) > 0:
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
                    assert np.all(sc['mass_components'] > 0)

            sc['age'] += ts     # model would do this operation
            print 'Completed step: ', i
