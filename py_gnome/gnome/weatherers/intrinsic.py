'''
For now just define a FayGravityInertial class here
State is not persisted yet - we just have a default object that gets
attached to Evaporation
'''

import numpy as np

from gnome.basic_types import oil_status
from gnome import GnomeId
from gnome.array_types import (density,
                               init_volume,
                               init_area,
                               area)
from gnome.environment import constants, Water


class FayGravityViscous(object):
    def __init__(self):
        self.spread_const = (1.53, 1.21)
        self.water = None
        self.array_types = {}

    def _initial_area(self):
        'this should be a constant - compute only once'

    def _update_area(self):
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
                 spreading=FayGravityViscous()):
        ''
        self.water = water
        self.spreading = spreading
        self.array_types = {'density': density,
                            'init_volume': init_volume,
                            'init_area': init_area,
                            'area': area}

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

    def _update_intrinsic_props(self, num_new_released, sc):
        water_temp = self.water.get('temperature', 'K')
        for spill in sc.spills:
            mask = sc.get_spill_mask(spill)
            rho = spill.get('substance').get_density(water_temp)
            sc['density'][mask] = rho

            # initailize
            if num_new_released > 0:
                sc['init_volume'][-num_new_released:] = \
                    np.sum(sc['mass'][-num_new_released:] / rho, 0)

    def _update_weathering_data(self, num_new_released, sc):
        '''
        intrinsic LE properties not set by any weatherer so let SpillContainer
        set these - will user be able to use select weatherers? Currently,
        evaporation defines 'density' data array
        '''
        mask = sc['status_codes'] == oil_status.in_water
        sc.weathering_data['floating'] = np.sum(sc['mass'][mask])

        amount_released = np.sum(sc['mass'][-num_new_released:])
        if 'amount_released' in sc.weathering_data:
            sc.weathering_data['amount_released'] += amount_released
        else:
            sc.weathering_data['amount_released'] = amount_released

        # update avg_density only if density array exists
        # wasted cycles at present since all values in density for given
        # timestep should be the same, but that will likely change
        sc.weathering_data['avg_density'] = np.average(sc['density'])
