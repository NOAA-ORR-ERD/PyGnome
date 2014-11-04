'''
For now just define a FayGravityInertial class here
State is not persisted yet - we just have a default object that gets
attached to Evaporation
'''

import numpy as np

from gnome.basic_types import oil_status
from gnome.array_types import (density,
                               init_volume,
                               init_area,
                               relative_bouyancy,
                               area)
from gnome.environment import constants


class FayGravityViscous(object):
    def __init__(self):
        self.spreading_const = (1.53, 1.21)
        self.water = None

    def set_initial_area(self):
        'this should be a constant - compute only once'

    def update_area(self):
        'update area'


class IntrinsicProps(object):
    '''
    Updates intrinsic properties of Oil
    Doesn't have an id like other gnome objects. It isn't exposed to
    application since Model will automatically instantiate if there
    are any Weathering objects defined
    '''
    def __init__(self,
                 water,
                 array_types=None):
        ''
        self.water = water
        self.spreading_const = (1.53, 1.21)
        self.array_types = {'density': density,
                            'init_volume': init_volume,
                            'init_area': init_area,
                            'relative_bouyancy': relative_bouyancy}
        if array_types:
            self.update_array_types(array_types)

    def update_array_types(self, m_array_types):
        '''
        update array_types based on weather model's array_types. For instance,
        if there is no Evaporation weatherer, then 'area' will not be in
        model's array_types, and we can remove init_volume, init_area,
        relative_bouyancy
        '''
        if 'area' not in m_array_types:
            for key in ('init_volume', 'relative_bouyancy', 'init_area'):
                if key in self.array_types:
                    del self.array_types[key]

    def update(self, num_new_released, sc):
        '''
        Uses 'substance' properties together with 'water' properties to update
        'density', 'init_volume', etc
        The 'init_volume' is not updated at each step; however, it depends on
        the 'density' which must be set/updated first and this depends on
        water object. So it was easiest to initialize the 'init_volume' for
        newly released particles here.
        '''
        self._update_intrinsic_props(num_new_released, sc)
        self._update_weathering_data(num_new_released, sc)

    def _update_weathering_data(self, new_LEs, sc):
        '''
        intrinsic LE properties not set by any weatherer so let SpillContainer
        set these - will user be able to use select weatherers? Currently,
        evaporation defines 'density' data array
        '''
        mask = sc['status_codes'] == oil_status.in_water
        sc.weathering_data['floating'] = np.sum(sc['mass'][mask])

        amount_released = np.sum(sc['mass'][-new_LEs:])
        if 'amount_released' in sc.weathering_data:
            sc.weathering_data['amount_released'] += amount_released
        else:
            sc.weathering_data['amount_released'] = amount_released

        # update avg_density only if density array exists
        # wasted cycles at present since all values in density for given
        # timestep should be the same, but that will likely change
        sc.weathering_data['avg_density'] = np.average(sc['density'])

    def _update_intrinsic_props(self, new_LEs, sc):
        water_temp = self.water.get('temperature', 'K')
        for spill in sc.spills:
            mask = sc.get_spill_mask(spill)

            # update properties associated with spill
            # todo: shouldn't this be time dependent?
            rho = spill.get('substance').get_density(water_temp)
            sc['density'][mask] = rho
            if 'area' in sc:
                self._update_area_arrays(sc, mask, new_LEs)

        if 'area' in sc:
            self._update_area(sc)

    def _update_area_arrays(self, sc, mask, new_LEs):
        ''' update areas required for computing 'area' arrays'''
        sc['relative_bouyancy'][mask] = \
            self._set_relative_bouyancy(sc['density'][mask])
        # initialize only new elements released by spill
        if new_LEs > 0:
            in_spill = mask[-new_LEs:]  # new LEs in this spill
            if np.any(in_spill):
                sc['init_volume'][-new_LEs:][in_spill] = \
                    np.sum(sc['mass'][-new_LEs:][in_spill] / sc['density'][mask],
                           0)

                init_volume = sc['init_volume'][-new_LEs:][in_spill][0]
                sc['init_area'][-new_LEs:][in_spill] = \
                    self._set_init_area(init_volume,
                                        sc['density'][-new_LEs:][in_spill],
                                        sc['relative_bouyancy'][-new_LEs:][in_spill])

    def _set_init_area(self, init_volume, rho_array, dbouy_array):
        'set initial area of each LE - used to Fay Gravity Viscous spreading'
        return (np.pi*(self.spreading_const[1]**4/self.spreading_const[0]**2) *
                (((init_volume)**5*constants['g']*dbouy_array) /
                 (self.water.get('kinematic_viscosity', 'St')**2))**(1./6.))

    def _update_area(self, sc):
        'update area area'
        dFay = 0.
        dEddy = 0.
        nu_h2o = self.water.get('kinematic_viscosity', 'St')
        if np.any(sc['age'][:] > 0):
            dFay = (self.spreading_const[1]**2./16. *
                    (constants['g']*sc['relative_bouyancy'] *
                     sc['init_volume']**2 / np.sqrt(nu_h2o*sc['age'])))
            dEddy = 0.033*sc['age']**(4/25)

        sc['area'][:] = sc['init_area'] + (dFay + dEddy) * sc['age']

    def _set_relative_bouyancy(self, rho_oil):
        '''
        relative bouyancy of oil: (rho_water - rho_oil) / rho_water
        only 3 lines but made it a function for easy testing
        '''
        rho_h2o = self.water.get('density', 'kg/m^3')
        return (rho_h2o - rho_oil)/rho_h2o
