'''
For now just define a FayGravityInertial class here
State is not persisted yet - we just have a default object that gets
attached to Evaporation
'''

import numpy as np

from gnome.basic_types import oil_status
from gnome.array_types import (density,
                               viscosity,
                               mass_components,
                               init_volume,
                               init_area,
                               relative_bouyancy,
                               area,
                               mol)
from gnome.environment import constants


class FayGravityViscous(object):
    def __init__(self):
        self.spreading_const = (1.53, 1.21)

    def set_init_area(self,
                      water_viscosity,
                      init_volume,
                      relative_bouyancy,
                      out=None):
        '''
        Initial area is computed for each LE only once. This can take scalars
        or numpy arrays for input since the initial_area for a bunch of LEs
        will be the same - scalar input is supported

        :param water_viscosity: viscosity of water
        :type water_viscosity: float and a scalar
        :param init_volume: initial volume
        :type init_volume: numpy.ndarray with dtype=float
        :param relative_bouyancy: relative bouyance of oil wrt water:
            (rho_water - rho_oil)/rho_water where rho defines density
        :type relative_bouyancy: numpy.ndarray with dtype=float

        Optional:

        :param out:
        :type out: numpy.ndarray with out.shape == init_volume.shape
        '''
        if out is None:
            out = np.zeros_like(init_volume)
            out = (out, out.reshape(-1))[out.shape == ()]

        out[:] = (np.pi*(self.spreading_const[1]**4/self.spreading_const[0]**2)
                  * (((init_volume)**5*constants['g']*relative_bouyancy) /
                     (water_viscosity**2))**(1./6.))

        if np.isscalar(init_volume):
            out = out[0]

        return out

    def update_area(self,
                    water_viscosity,
                    init_area,
                    init_volume,
                    relative_bouyancy,
                    age,
                    out=None):
        '''
        Update area and stuff it in out array. This takes numpy arrays as
        input for init_volume, relative_bouyancy and age. Each element of the
        array is the property for an LE - array should be the same shape.
        '''
        if out is None:
            out = np.zeros_like(init_area, dtype=np.float64)
            out = (out, out.reshape(-1))[out.shape == ()]

        if np.isscalar(age):
            age = np.asarray(age).reshape(-1)

        out[:] = init_area
        mask = age > 0
        if np.any(mask):
            dFay = (self.spreading_const[1]**2./16. *
                    (constants['g']*relative_bouyancy[mask] *
                     init_volume[mask]**2 /
                     np.sqrt(water_viscosity*age[mask])))
            dEddy = 0.033*age[mask]**(4/25)
            out[mask] += (dFay + dEddy) * age[mask]

        if np.isscalar(init_volume):
            out = out[0]

        return out


class IntrinsicProps(object):
    '''
    Updates intrinsic properties of Oil
    Doesn't have an id like other gnome objects. It isn't exposed to
    application since Model will automatically instantiate if there
    are any Weathering objects defined

    Use this to manage data_arrays associated with weathering that are not
    defined in Weatherers. This is inplace of defining initializers for every
    single array, let IntrinsicProps set/initialize/update these arrays.
    '''
    # group array_types by a key that if present will require these optional
    # arrays. For instance, Evaporation requires 'area' which needs:
    #
    #    'area': ('relative_bouyancy', 'init_area')
    #
    # IntrinsicProps sets 'area' but it does not define it as an array_type.
    # The 'area' array_type is set/required by a Weatherer like
    # Evaporation. This object just sets the 'area' array and to do so it
    # requires these additional arrays
    _array_types_group = \
        {'area': {'init_area': init_area,
                  'relative_bouyancy': relative_bouyancy},
         'mol': {'mass_components': mass_components}}

    def __init__(self,
                 water,
                 array_types=None,
                 spreading=FayGravityViscous()):
        self.water = water
        self.spreading = spreading
        self.array_types = {'density': density,
                            'init_volume': init_volume,
                            'viscosity': viscosity}
        if array_types:
            self.update_array_types(array_types)

    def update_array_types(self, m_array_types):
        '''
        update array_types based on weather model's array_types. For instance,
        if there is no Evaporation weatherer, then 'area' will not be in
        model's array_types, and we can remove init_volume, init_area,
        relative_bouyancy
        '''
        for key, val in self._array_types_group.iteritems():
            if key in m_array_types:
                self.array_types.update(val)
            else:
                for atype in val:   # otherwise remove associated array_types
                    if atype in self.array_types:
                        del self.array_types[atype]

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
        if len(sc) == 0:
            # nothing released yet - set everything to 0.0
            for key in ('avg_density', 'floating', 'amount_released',
                        'avg_viscosity'):
                sc.weathering_data[key] = 0.0
        else:
            mask = sc['status_codes'] == oil_status.in_water
            # update avg_density from density array
            # wasted cycles at present since all values in density for given
            # timestep should be the same, but that will likely change
            sc.weathering_data['avg_density'] = np.average(sc['density'])
            sc.weathering_data['avg_viscosity'] = np.average(sc['viscosity'])
            sc.weathering_data['floating'] = np.sum(sc['mass'][mask])

            amount_released = np.sum(sc['mass'][-new_LEs:])
            if 'amount_released' in sc.weathering_data:
                sc.weathering_data['amount_released'] += amount_released
            else:
                sc.weathering_data['amount_released'] = amount_released

    def _update_intrinsic_props(self, new_LEs, sc):
        water_temp = self.water.get('temperature', 'K')
        for spill in sc.spills:
            mask = sc.get_spill_mask(spill)

            # update properties associated with spill
            # todo: shouldn't this be time dependent?
            rho = spill.get('substance').get_density(water_temp)
            sc['density'][mask] = rho
            sc['viscosity'][mask] = \
                spill.get('substance').get_viscosity(water_temp)

            if 'area' in sc:
                self._update_area_arrays(sc, mask, new_LEs)

            if 'mol' in sc:
                mw = spill.get('substance').molecular_weight
                sc['mol'][mask] = np.sum(sc['mass_components'][mask, :]/mw, 1)

        if 'area' in sc:
            self.spreading.update_area(self.water.get('kinematic_viscosity',
                                                      'St'),
                                       sc['init_area'],
                                       sc['init_volume'],
                                       sc['relative_bouyancy'],
                                       sc['age'],
                                       sc['area'])

    def _update_area_arrays(self, sc, mask, new_LEs):
        ''' update areas required for computing 'area' arrays'''
        sc['relative_bouyancy'][mask] = \
            self._set_relative_bouyancy(sc['density'][mask])
        # initialize only new elements released by spill
        if new_LEs > 0:
            in_spill = mask[-new_LEs:]  # new LEs in this spill
            if np.any(in_spill):
                sc['init_volume'][-new_LEs:][in_spill] = \
                    np.sum(sc['mass'][-new_LEs:][in_spill] /
                           sc['density'][-new_LEs:][in_spill], 0)

                # init volume is same for all these new LEs so just provide
                # a scalar value
                init_volume = sc['init_volume'][-new_LEs:][in_spill][0]

                # Cannot change the init_area in place since the following:
                #    sc['init_area'][-new_LEs:][in_spill]
                # is an advanced indexing operation that makes a copy anyway
                sc['init_area'][-new_LEs:][in_spill] = \
                    self.spreading.set_init_area(init_volume,
                        self.water.get('kinematic_viscosity', 'St'),
                        sc['relative_bouyancy'][-new_LEs:][in_spill],
                        sc['init_area'][-new_LEs:][in_spill])

    def _set_relative_bouyancy(self, rho_oil):
        '''
        relative bouyancy of oil: (rho_water - rho_oil) / rho_water
        only 3 lines but made it a function for easy testing
        '''
        rho_h2o = self.water.get('density', 'kg/m^3')
        return (rho_h2o - rho_oil)/rho_h2o
