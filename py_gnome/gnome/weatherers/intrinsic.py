'''
This module was originally intended to hold classes that initialize weathering
data arrays that are not set by any weathering process. It was also meant to
update the intrinsic properties of the LEs, hence the name 'intrinsic.py'
However, it sets and updates weathering data arrays including intrinsic data
like 'viscosity', 'density' and other data. Call the class WeatheringData()
which is defined in a gnome model if there are weatherers defined.

For now just define a FayGravityInertial class here
It is only used by WeatheringData to update the 'area' and related arrays
'''
import numpy as np
from repoze.lru import lru_cache

from gnome.basic_types import oil_status, fate
from gnome import AddLogger, constants


class FayGravityViscous(object):
    def __init__(self):
        self.spreading_const = (1.53, 1.21)
        self.thickness_limit = .0001

    def init_area(self,
                  water_viscosity,
                  init_volume,
                  relative_bouyancy):
        '''
        Initial area is computed for each LE only once. This takes scalars
        inputs since water_viscosity, init_volume and relative_bouyancy for a
        bunch of LEs released together will be the same.

        :param water_viscosity: viscosity of water
        :type water_viscosity: float
        :param init_volume: total initial volume of all LEs released together
        :type init_volume: float
        :param relative_bouyancy: relative bouyance of oil wrt water:
            (rho_water - rho_oil)/rho_water where rho defines density
        :type relative_bouyancy: float

        Equation:
        A0 = np.pi*(k2**4/k1**2)*(((n_LE*V0)**5*g*dbuoy)/(nu_h2o**2))**(1./6.)
        '''
        self._check_relative_bouyancy(relative_bouyancy)
        out = (np.pi*(self.spreading_const[1]**4/self.spreading_const[0]**2)
               * (((init_volume)**5*constants.gravity*relative_bouyancy) /
                  (water_viscosity**2))**(1./6.))

        return out

    def _check_relative_bouyancy(self, rel_bouy):
        '''
        For now just raise an error if any relative_bouyancy is < 0. These
        particles will sink, ask how we want to deal with them. They should
        be removed or we should only look at floating particles when computing
        area?
        '''
        if np.any(rel_bouy < 0):
            raise ValueError("Found particles with relative_bouyancy < 0. "
                             "Area does not handle this case at present.")

    def update_area(self,
                    water_viscosity,
                    init_area,
                    init_volume,
                    relative_bouyancy,
                    age,
                    thickness,
                    area,   # update only if thickness > thickness_lim
                    frac_coverage=None,
                    out=None):
        '''
        Update area and stuff it in out array. This takes numpy arrays
        as input for init_volume, relative_bouyancy and age. Each
        element of the array is the property for an LE - array should be the
        same shape.

        Since this is for updating area, it assumes age > 0 for all elements.
        It is used inside WeatheringData and invoked for particles with age > 0

        It only updates the area for particles with thickness > xxx
        Since the frac_coverage should only be applied to particles which are
        updated, let's apply this in here.

        todo: unsure if thickness check should be here or outside this object.
        Since thickness limit is here, leave it for now, but maybe
        eventually move thickness_limit to OilProps/make it property of
        substance - say 'max_spreading_thickness', then move thickness check
        and frac_coverage back to WeatheringData
        '''
        self._check_relative_bouyancy(relative_bouyancy)
        if np.any(age == 0):
            raise ValueError('for new particles use init_area - age '
                             'must be > 0')

        if out is None:
            out = np.zeros_like(init_volume, dtype=np.float64)

        # ADIOS 2 used 0.1 mm as a minimum average spillet thickness for crude
        # oil and heavy refined products and 0.01 mm for lighter refined
        # products. Use 0.1mm for now
        out[:] = area
        mask = thickness > self.thickness_limit  # units of meters
        if np.any(mask):
            out[mask] = init_area[mask]
            dFay = (self.spreading_const[1]**2./16. *
                    (constants.gravity*relative_bouyancy[mask] *
                     init_volume[mask]**2 /
                     np.sqrt(water_viscosity*age[mask])))
            dEddy = 0.033*age[mask]**(4./25)
            out[mask] += (dFay + dEddy) * age[mask]

            # apply fraction coverage here so particles less than min thickness
            # are not changed
            if frac_coverage is not None:
                out[mask] *= frac_coverage[mask]

        return out


class WeatheringData(AddLogger):
    '''
    Updates intrinsic properties of Oil
    Doesn't have an id like other gnome objects. It isn't exposed to
    application since Model will automatically instantiate if there
    are any Weathering objects defined

    Use this to manage data_arrays associated with weathering that are not
    defined in Weatherers. This is inplace of defining initializers for every
    single array, let WeatheringData set/initialize/update these arrays.
    '''
    def __init__(self,
                 water,
                 spreading=FayGravityViscous()):
        self.water = water
        self.spreading = spreading
        self.array_types = {'fate_status', 'positions', 'status_codes',
                            'density', 'viscosity',
                            'mass_components', 'mass',
                            # init volume of particles released together
                            'init_volume',
                            'init_mass', 'frac_water', 'frac_lost',
                            'area', 'init_area', 'relative_bouyancy',
                            'frac_coverage', 'thickness', 'age'}

        # following used to update viscosity
        self.visc_curvfit_param = 1.5e3     # units are sec^0.5 / m
        self.visc_f_ref = 0.84

    def initialize(self, sc):
        '''
        1. initialize standard keys:
        avg_density, floating, amount_released, avg_viscosity to 0.0
        2. set init_density for all ElementType objects in each Spill
        '''
        # nothing released yet - set everything to 0.0
        for key in ('avg_density', 'floating', 'amount_released',
                    'avg_viscosity', 'beached'):
            sc.weathering_data[key] = 0.0

    def update(self, num_new_released, sc):
        '''
        Uses 'substance' properties together with 'water' properties to update
        'density', 'init_volume', etc
        The 'init_volume' is not updated at each step; however, it depends on
        the 'density' which must be set/updated first and this depends on
        water object. So it was easiest to initialize the 'init_volume' for
        newly released particles here.
        '''
        if len(sc) > 0:
            self._update_intrinsic_props(sc)
            self._update_weathering_data(num_new_released, sc)

    def _update_weathering_data(self, new_LEs, sc):
        '''
        intrinsic LE properties not set by any weatherer so let SpillContainer
        set these - will user be able to use select weatherers? Currently,
        evaporation defines 'density' data array
        '''
        # update avg_density from density array
        # wasted cycles at present since all values in density for given
        # timestep should be the same, but that will likely change
        # Any optimization in doing the following?:
        #   (sc['mass'] * sc['density']).sum()/sc['mass'].sum()
        # todo: test weighted average
        sc.weathering_data['avg_density'] = \
            np.sum(sc['mass']/np.sum(sc['mass']) * sc['density'])
        sc.weathering_data['avg_viscosity'] = \
            np.sum(sc['mass']/sc['mass'].sum() * sc['viscosity'])

        # floating includes LEs marked to be skimmed + burned + dispersed
        # todo: remove fate_status and add 'surface' to status_codes. LEs
        # marked to be skimmed, burned, dispersed will also be marked as
        # 'surface' so following can get cleaned up.
        sc.weathering_data['floating'] = \
            (sc['mass'][sc['fate_status'] == fate.surface_weather].sum() +
             sc['mass'][sc['fate_status'] & fate.skim == fate.skim].sum() +
             sc['mass'][sc['fate_status'] & fate.burn == fate.burn].sum() +
             sc['mass'][sc['fate_status'] & fate.disperse == fate.disperse].sum())

        sc.weathering_data['beached'] = sc['mass'][sc['status_codes'] ==
                                                   oil_status.on_land].sum()

        # add 'non_weathering' key if any mass is released for nonweathering
        # particles.
        nonweather = sc['mass'][sc['fate_status'] == fate.non_weather].sum()
        if nonweather > 0:
            sc.weathering_data['non_weathering'] = nonweather

        if new_LEs > 0:
            amount_released = np.sum(sc['mass'][-new_LEs:])
            if 'amount_released' in sc.weathering_data:
                sc.weathering_data['amount_released'] += amount_released
            else:
                sc.weathering_data['amount_released'] = amount_released

    def _update_intrinsic_props(self, sc):
        '''
        - initialize 'density', 'viscosity', and other optional arrays for
        newly released particles.
        - update intrinsic properties like 'density', 'viscosity' and optional
        arrays for previously released particles
        '''

        for substance, data in sc.itersubstancedata(self.array_types,
                                                    fate='all'):
            'update properties only if elements are released'
            if len(data['density']) == 0:
                continue

            # could also use 'age' but better to use an uninitialized var since
            # we might end up changing 'age' to something with less than a
            # time_step resolution
            new_LEs_mask = data['density'] == 0
            if sum(new_LEs_mask) > 0:
                self._init_new_particles(new_LEs_mask, data, substance)
            if sum(~new_LEs_mask) > 0:
                self._update_old_particles(~new_LEs_mask, data, substance)

        sc.update_from_fatedataview(fate='all')

    def update_fate_status(self, sc):
        '''
        Update fate status after model invokes move_elements()
        - elements will beach or refloat
        - then Model will update fate_status of elements that beached/refloated

        Model calls this and input is spill container, not a view of the data
        '''
        # for old particles, update fate_status
        # particles in_water or off_maps continue to weather
        # only particles on_land stop weathering
        non_w_mask = sc['status_codes'] == oil_status.on_land
        sc['fate_status'][non_w_mask] = fate.non_weather

        # update old particles that may now have refloated
        # only want to do this for particles with a valid substance - if
        # substance is None, they do not weather
        # also get all data for a substance since we are modifying the
        # fate_status - lets not use it to filter data
        for substance, data in sc.itersubstancedata(self.array_types,
                                                    fate='all'):
            mask = np.asarray([True] * len(data['fate_status']))
            self._init_fate_status(mask, data)

        sc.update_from_fatedataview(fate='all')

    def _init_new_particles(self, mask, data, substance):
        '''
        initialize new particles released together in a given timestep

        :param mask: mask gives only the new LEs in data arrays
        :type mask: numpy bool array
        :param data: dict containing numpy arrays
        :param substance: OilProps object defining the substance spilled
        '''
        water_temp = self.water.get('temperature', 'K')
        data['density'][mask] = substance.get_density(water_temp)

        # initialize mass_components -
        # sub-select mass_components array by substance.num_components.
        # Currently, the physics for modeling multiple spills with different
        # substances is not being correctly done in the same model. However,
        # let's put some basic code in place so the data arrays can infact
        # contain two substances and the code does not raise exceptions. The
        # mass_components are zero padded for substance which has fewer
        # psuedocomponents. Subselecting mass_components array by
        # [mask, :substance.num_components] ensures numpy operations work
        data['mass_components'][mask, :substance.num_components] = \
            (np.asarray(substance.mass_fraction, dtype=np.float64) *
             (data['mass'][mask].reshape(len(data['mass'][mask]), -1)))

        data['init_mass'][mask] = data['mass'][mask]

        if substance.get_viscosity(water_temp) is not None:
            'make sure we do not add NaN values'
            data['viscosity'][mask] = \
                substance.get_viscosity(water_temp)

        '''
        Sets relative_bouyancy, init_volume, init_area, thickness all of
        which are required when computing the 'area' of each LE
        '''
        data['relative_bouyancy'][mask] = \
            self._set_relative_bouyancy(data['density'][mask])

        # Cannot change the init_area in place since the following:
        #    sc['init_area'][-new_LEs:][in_spill]
        # is an advanced indexing operation that makes a copy anyway
        # Also, init_volume is same for all these new LEs so just provide
        # a scalar value
        data['init_volume'][mask] = np.sum(data['init_mass'][mask] /
                                           data['density'][mask], 0)
        data['init_area'][mask] = \
            self.spreading.init_area(self.water.get('kinematic_viscosity',
                                                    'square meter per second'),
                                     data['init_volume'][mask][0],
                                     data['relative_bouyancy'][mask][0])
        data['area'][mask] = data['init_area'][mask]
        data['thickness'][mask] = data['init_volume'][mask]/data['area'][mask]

        # initialize the fate_status array based on positions and status_codes
        self._init_fate_status(mask, data)

    def _init_fate_status(self, update_LEs_mask, data):
        '''
        initialize fate_status for newly released LEs or refloated LEs
        For refloated LEs, the mask should apply to non_weather LEs.
        Currently, the 'status_codes' is separate from 'fate_status' and we
        don't want to reset the 'fate_status' of LEs that have been marked
        as 'skim' or 'burn' or 'disperse'. This should only apply for newly
        released LEs (currently marked as non_weather since that's the default)
        and for refloated LEs which should also have been marked as non_weather
        when they beached.
        '''
        surf_mask = \
            np.logical_and(update_LEs_mask,
                           np.logical_and(data['positions'][:, 2] == 0,
                                          data['status_codes'] ==
                                          oil_status.in_water))
        subs_mask = \
            np.logical_and(update_LEs_mask,
                           np.logical_and(data['positions'][:, 2] > 0,
                                          data['status_codes'] ==
                                          oil_status.in_water))

        # set status for new_LEs correctly
        data['fate_status'][surf_mask] = fate.surface_weather
        data['fate_status'][subs_mask] = fate.subsurf_weather

    @lru_cache(2)
    def _get_kv1_weathering_visc_update(self, v0):
        '''
        kv1 is constant.
        It defining the exponential change in viscosity as it weathers due to
        the fraction lost to evaporation/dissolution:
            v(t) = v' * exp(kv1 * f_lost_evap_diss)

        kv1 = sqrt(v0) * 1500
        if kv1 < 1, then return 1
        if kv1 > 10, then return 10

        Since this is fixed for an oil, it only needs to be computed once. Use
        lru_cache on this function to cache the result for a given initial
        viscosity: v0
        '''
        # find kv1
        kv1 = np.sqrt(v0) * self.visc_curvfit_param
        if kv1 < 1:
            kv1 = 1

        if kv1 > 10:
            kv1 = 10

        return kv1

    @lru_cache(2)
    def _get_k_rho_weathering_dens_update(self, substance):
        '''
        use lru_cache on substance. substance is an OilProps object, if this
        object stays the same, then return the cached value for k_rho
        This depends on initial mass fractions, initial density and fixed
        component densities
        '''
        # update density/viscosity/relative_bouyance/area for previously
        # released elements
        rho0 = substance.get_density(self.water.get('temperature', 'K'))

        # dimensionless constant
        k_rho = (rho0 /
                 (substance.component_density * substance.mass_fraction).sum())

        return k_rho

    def _update_old_particles(self, mask, data, substance):
        '''
        update density, area
        '''
        k_rho = self._get_k_rho_weathering_dens_update(substance)
        # sub-select mass_components array by substance.num_components.
        # Currently, the physics for modeling multiple spills with different
        # substances is not being correctly done in the same model. However,
        # let's put some basic code in place so the data arrays can infact
        # contain two substances and the code does not raise exceptions. The
        # mass_components are zero padded for substance which has fewer
        # psuedocomponents. Subselecting mass_components array by
        # [mask, :substance.num_components] ensures numpy operations work
        mass_frac = \
            (data['mass_components'][mask, :substance.num_components] /
             data['mass'][mask].reshape(np.sum(mask), -1))
        # check if density becomes > water, set it equal to water in this case
        new_rho = k_rho*(substance.component_density * mass_frac).sum(1)
        if np.any(new_rho > self.water.density):
            new_rho[new_rho > self.water.density] = self.water.density
            self.logger.info(self._pid + "during update, density is greater "
                             "than water density - set it to water density ")

        data['density'][mask] = new_rho

        # following implementation results in an extra array called
        # fw_d_fref but is easy to read
        v0 = substance.get_viscosity(self.water.get('temperature', 'K'))
        if v0 is not None:
            kv1 = self._get_kv1_weathering_visc_update(v0)
            fw_d_fref = data['frac_water'][mask]/self.visc_f_ref
            data['viscosity'][mask] = \
                (v0 * np.exp(kv1 *
                             data['frac_lost'][mask]) *
                 (1 + (fw_d_fref/(1.187 - fw_d_fref)))**2.49)

        # todo: Need formulas to update density
        # prev_rel = sc.num_released-new_LEs
        # if prev_rel > 0:
        #    update density, viscosity .. etc

        # update self.spreading.thickness_limit based on type of substance
        # create 'frac_coverage' array and pass it in to scale area by it
        # update_area will only update the area for particles with
        # thickness greater than some minimum thickness and the frac_coverage
        # is only applied to LEs whose area is updated. Elements below a min
        # thickness should not be updated
        data['area'][mask] = \
            self.spreading.update_area(self.water.get('kinematic_viscosity',
                                                      'square meter per second'),
                                       data['init_area'][mask],
                                       data['init_volume'][mask],
                                       data['relative_bouyancy'][mask],
                                       data['age'][mask],
                                       data['thickness'][mask],
                                       data['area'][mask],
                                       data['frac_coverage'][mask])

        # update thickness per the new area
        data['thickness'][mask] = data['init_volume'][mask]/data['area'][mask]

    def _set_relative_bouyancy(self, rho_oil):
        '''
        relative bouyancy of oil: (rho_water - rho_oil) / rho_water
        only 3 lines but made it a function for easy testing
        '''
        rho_h2o = self.water.get('density', 'kg/m^3')
        return (rho_h2o - rho_oil)/rho_h2o
