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


class FayGravityViscous(AddLogger):
    '''
    Model the FayGravityViscous spreading of the oil. This assumes all LEs
    released together spread as a blob. The blob can be partitioned into 'N'
    LEs and the assumption is that the thickness and initial volume of the
    blob applies to all LEs in it. As such, instead of computing area, lets
    compute thickness - whether 1 or 10 LEs is used to model the blob, the
    thickness remains the same.
    '''
    def __init__(self):
        self.spreading_const = (1.53, 1.21)
        self.thickness_limit = .0001

    @lru_cache(1)
    def _gravity_spreading_t0(self,
                              water_viscosity,
                              relative_bouyancy,
                              blob_init_vol):
        # time to reach a0
        t0 = ((self.spreading_const[1]/self.spreading_const[0]) ** 4.0 *
              (blob_init_vol/(water_viscosity * constants.gravity *
                              relative_bouyancy))**(1./3))
        return t0

    def init_area(self,
                  water_viscosity,
                  relative_bouyancy,
                  blob_init_vol,
                  time_step):
        '''
        This takes scalars inputs since water_viscosity, init_volume and
        relative_bouyancy for a bunch of LEs released together will be the same
        It

        :param water_viscosity: viscosity of water
        :type water_viscosity: float
        :param init_volume: total initial volume of all LEs released together
        :type init_volume: float
        :param relative_bouyancy: relative bouyance of oil wrt water:
            (rho_water - rho_oil)/rho_water where rho defines density
        :type relative_bouyancy: float
        :param time_step: age of particle at the end of this model step. If
            is greater than the time for gravity spreading, then return initial
            area due to gravity spreading. If this is greater, then set age
            = time_step - gravity_spreading_time and invoke update_area().
        :type time_step: float

        Equation for gravity spreading:
        ::
            A0 = np.pi*(k2**4/k1**2)*(((n_LE*V0)**5*g*dbuoy)/(nu_h2o**2))**(1./6.)
        '''
        self._check_relative_bouyancy(relative_bouyancy)
        t0 = self._gravity_spreading_t0(water_viscosity,
                                        relative_bouyancy,
                                        blob_init_vol)
        if t0 >= time_step:
            a0 = (np.pi*(self.spreading_const[1]**4/self.spreading_const[0]**2)
                  * (((blob_init_vol)**5*constants.gravity*relative_bouyancy) /
                     (water_viscosity**2))**(1./6.))
            return a0
        else:
            area = self._update_blob_area(water_viscosity, relative_bouyancy,
                                          blob_init_vol, time_step - t0)
            return area

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

    def _update_blob_area(self, water_viscosity, relative_bouyancy,
                          blob_init_volume, age):
        area = (np.pi * self.spreading_const[1]**2 *
                (blob_init_volume**2 * constants.gravity * relative_bouyancy /
                 np.sqrt(water_viscosity)) ** (1./3) * np.sqrt(age))

        return area

    def update_area(self,
                    water_viscosity,
                    relative_bouyancy,
                    blob_init_volume,
                    area,
                    age):
        '''
        update area array in place, also return area array
        not including frac_coverage at present
        each blob is defined by its age. This updates the area of each blob,
        as such, use the mean relative_bouyancy for each blob. Still check
        and ensure relative bouyancy is > 0 for all LEs
        '''
        if np.any(age == 0):
            msg = "use init_area for age == 0"
            raise ValueError(msg)

        self._check_relative_bouyancy(relative_bouyancy)

        # update area for each blob of LEs
        for b_age in np.unique(age):
            # within each age blob_init_volume should also be the same
            m_age = b_age == age

            # now update area of old LEs
            blob_thickness = blob_init_volume[m_age][0]/area[m_age].sum()
            if blob_thickness > self.thickness_limit:

                self.logger.debug(self._pid + "Before update: ")
                msg = ("\n\trel_bouy: {0}\n"
                       "\tblob_i_vol: {1}\n"
                       "\tage: {2}\n"
                       "\tarea: {3}".
                       format(relative_bouyancy, blob_init_volume[m_age][0],
                              age[m_age][0], area[m_age].sum()))
                self.logger.debug(msg)

                # update area
                blob_area = \
                    self._update_blob_area(water_viscosity,
                                           relative_bouyancy,
                                           blob_init_volume[m_age][0],
                                           age[m_age][0])
                area[m_age] = blob_area/m_age.sum()
                self.logger.debug(self._pid +
                                  "\tarea after update: {0}".format(blob_area))

        return area


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
                            'bulk_init_volume',
                            'init_mass', 'frac_water', 'frac_lost',
                            'fay_area',
                            'frac_coverage', 'age',
                            'spill_num'}

        # following used to update viscosity
        self.visc_curvfit_param = 1.5e3     # units are sec^0.5 / m
        self.visc_f_ref = 0.84

        # relative_bouyancy - use density at release time. For now
        # temperature is fixed so just compute once and store. When temperature
        # varies over time, may want to do something different
        self._init_relative_bouyancy = None

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

    def update(self, num_new_released, sc, time_step):
        '''
        Uses 'substance' properties together with 'water' properties to update
        'density', 'bulk_init_volume', etc
        The 'bulk_init_volume' is not updated at each step; however, it depends on
        the 'density' which must be set/updated first and this depends on
        water object. So it was easiest to initialize the 'bulk_init_volume' for
        newly released particles here.
        '''
        if len(sc) > 0:
            self._update_intrinsic_props(sc, time_step)
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
        # todo: move weighted average to utilities
        # also added a check for 'mass' == 0, edge case
        if len(sc.substances) > 1:
            self.logger.warning(self._pid + "current code isn't valid for "
                                "multiple weathering substances")
        elif len(sc.substances) == 0:
            # should not happen with the Web API. Just log a warning for now
            self.logger.warning(self._pid + "weathering is on but found no"
                                "weatherable substances.")
        else:
            # avg_density, avg_viscosity applies to elements that are on the
            # surface and being weathered
            data = sc.substancefatedata(sc.substances[0],
                                        {'mass', 'density', 'viscosity'})
            if data['mass'].sum() > 0.0:
                sc.weathering_data['avg_density'] = \
                    np.sum(data['mass']/data['mass'].sum() * data['density'])
                sc.weathering_data['avg_viscosity'] = \
                    np.sum(data['mass']/data['mass'].sum() * data['viscosity'])
            else:
                self.logger.info(self._pid + "sum of 'mass' array went to 0.0")

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

    def _update_intrinsic_props(self, sc, time_step):
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
                self._init_new_particles(new_LEs_mask, data, substance,
                                         time_step)
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
            mask = data['fate_status'] & fate.non_weather == fate.non_weather
            self._init_fate_status(mask, data)

        sc.update_from_fatedataview(fate='all')

    def _init_new_particles(self, mask, data, substance, time_step):
        '''
        initialize new particles released together in a given timestep

        :param mask: mask gives only the new LEs in data arrays
        :type mask: numpy bool array
        :param data: dict containing numpy arrays
        :param substance: OilProps object defining the substance spilled
        :param time_step: timestep for this step
        '''
        water_temp = self.water.get('temperature', 'K')
        data['density'][mask] = substance.get_density(water_temp)

        if self._init_relative_bouyancy is None:
            rho_h2o = self.water.get('density', 'kg/m^3')
            self._init_relative_bouyancy = \
                (rho_h2o - data['density'][mask][0])/rho_h2o

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

        # initialize bulk_init_volume and fay_area for new particles per spill
        # other properties must be set (like 'mass', 'density')
        self._init_data_by_spill(mask, data, substance, time_step)

        # initialize the fate_status array based on positions and status_codes
        self._init_fate_status(mask, data)

    def _init_data_by_spill(self, mask, data, substance, time_step):
        '''
        set bulk_init_volume and fay_area. These are set on a per spill bases
        in addition to per substance.
        '''
        # do this once incase there are any unit conversions, it only needs to
        # happen once - for efficiency
        water_kvis = self.water.get('kinematic_viscosity',
                                    'square meter per second')
        rho_h2o = self.water.get('density', 'kg/m^3')

        '''
        Sets relative_bouyancy, bulk_init_volume, init_area, thickness all of
        which are required when computing the 'thickness' of each LE
        '''
        for s_num in np.unique(data['spill_num'][mask]):
            s_mask = np.logical_and(mask,
                                    data['spill_num'] == s_num)
            # do the sum only once for efficiency
            num = s_mask.sum()

            data['bulk_init_volume'][s_mask] = \
                (data['mass'][s_mask][0]/data['density'][s_mask][0]) * num
            init_blob_area = \
                self.spreading.init_area(water_kvis,
                                         self._init_relative_bouyancy,
                                         data['bulk_init_volume'][s_mask][0],
                                         time_step)

            data['fay_area'][s_mask] = init_blob_area/num

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

    @lru_cache(1)
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

    @lru_cache(1)
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

        water_kvis = self.water.get('kinematic_viscosity',
                                    'square meter per second')

        # must update intrinsic properties per spill. Same substance but
        # multiple spills - update intrinsic for each spill.
        for s_num in np.unique(data['spill_num'][mask]):
            s_mask = np.logical_and(mask,
                                    data['spill_num'] == s_num)
            # sub-select mass_components array by substance.num_components.
            # Currently, the physics for modeling multiple spills with different
            # substances is not being correctly done in the same model. However,
            # let's put some basic code in place so the data arrays can infact
            # contain two substances and the code does not raise exceptions. The
            # mass_components are zero padded for substance which has fewer
            # psuedocomponents. Subselecting mass_components array by
            # [mask, :substance.num_components] ensures numpy operations work
            mass_frac = \
                (data['mass_components'][s_mask, :substance.num_components] /
                 data['mass'][s_mask].reshape(np.sum(s_mask), -1))
            # check if density becomes > water, set it equal to water in this
            # case
            new_rho = k_rho*(substance.component_density * mass_frac).sum(1)
            if np.any(new_rho > self.water.density):
                new_rho[new_rho > self.water.density] = self.water.density
                self.logger.info(self._pid + "during update, density is larger"
                                 " than water density - set to water density")

            data['density'][s_mask] = new_rho

            # following implementation results in an extra array called
            # fw_d_fref but is easy to read
            v0 = substance.get_viscosity(self.water.get('temperature', 'K'))
            if v0 is not None:
                kv1 = self._get_kv1_weathering_visc_update(v0)
                fw_d_fref = data['frac_water'][s_mask]/self.visc_f_ref
                data['viscosity'][s_mask] = \
                    (v0 * np.exp(kv1 *
                                 data['frac_lost'][s_mask]) *
                     (1 + (fw_d_fref/(1.187 - fw_d_fref)))**2.49)
            data['fay_area'][s_mask] = \
                self.spreading.update_area(water_kvis,
                                           self._init_relative_bouyancy,
                                           data['bulk_init_volume'][s_mask],
                                           data['fay_area'][s_mask],
                                           data['age'][s_mask])
