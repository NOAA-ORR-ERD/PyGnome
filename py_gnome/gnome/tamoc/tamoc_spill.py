#!/usr/bin/env python

"""
Assorted code for working with TAMOC
"""

from datetime import timedelta

import copy
import random
import numpy as np
import unit_conversion as uc

from netCDF4 import date2num, num2date
from datetime import datetime

import gnome
from gnome.utilities.projections import FlatEarthProjection
from gnome.cy_gnome.cy_rise_velocity_mover import rise_velocity_from_drop_size

from tamoc import ambient, seawater
from tamoc import chemical_properties as chem
from tamoc import dbm, sintef, dispersed_phases, params
from tamoc import bent_plume_model as bpm
from tamoc import chemical_properties as chem

__all__ = []


class TamocDroplet():
    """
    Dummy class to show what we need from the TAMOC output

    :param mass_flux=1.0: Measured in kg/s
    :param radius=1e-6:   Measured in meters
    :param density=900.0: Measured in kg/m^3 at 15degC
    :param position=(10, 20, 100): (x, y, z) in meters
    :param flag_phase_insitu='Mixture': Flag for the phase of the particle
                                        at plumetermination
    :param flag_phase_surface='Mixture': Flag for the phase of the particle
                                         at 1 atm and 15 degC
    """
    def __init__(self,
                 mass_flux=1.0,
                 radius=1e-6,
                 density=900.0,
                 position=(10, 20, 100),
                 flag_phase_insitu='Mixture',
                 flag_phase_surface='Mixture'):

        self.mass_flux = mass_flux
        self.radius = radius
        self.density = density
        self.position = np.asanyarray(position)
        self.flag_phase_insitu = flag_phase_insitu
        self.flag_phase_surface = flag_phase_surface

    def __repr__(self):
        return ('[flux = {0}, radius = {1}, density = {2}, position = {3}]'
                .format(self.mass_flux,
                        self.radius,
                        self.density,
                        self.position))


class TamocDissMasses():
    """
        Dummy class to show what we need from the TAMOC output

        :param mass_flux=1.0: Measured in kg/s
        :param position=(10, 20, 100): (x, y, z) in meters
        :param chem_name='x': The name of the chemical
    """
    def __init__(self,
                 mass_flux=1.0,
                 position=(10, 20, 100),
                 chem_name='x'):
        self.mass_flux = mass_flux
        self.position = np.asanyarray(position)
        self.chem_name = chem_name

    def __repr__(self):
        return ('[flux = {0}, position = {1}, chem_name = {2}]'
                .format(self.mass_flux, self.position, self.chem_name))


def log_normal_pdf(x, mean, std):
    """
        Utility  to compute the log normal CDF
        - used to get a "realistic" distribution of droplet sizes
    """
    sigma = np.sqrt(np.log(1 + std ** 2 / mean ** 2))
    mu = np.log(mean) + sigma ** 2 / 2

    return ((1 / (x * sigma * np.sqrt(2 * np.pi))) *
            np.exp(-((np.log(x) - mu) ** 2 / (2 * sigma ** 2))))


def fake_tamoc_results(num_droplets=10):
    """
        utility for providing a tamoc result set

        Returns a simple list of TamocDroplet objects
    """

    # sizes from 10 to 1000 microns
    radius = np.linspace(10, 300, num_droplets) * 1e-6  # meters

    mass_flux = log_normal_pdf(2 * radius, 2e-4, 1.5e-4) * 0.1
    # normalize to 10 kg/s (about 5000 bbl per day)
    mass_flux *= 10.0 / mass_flux.sum()

    # give it a range, why not?
    density = np.linspace(900, 850, num_droplets)  # kg/m^3 at 15degC

    # linear release
    position = np.empty((num_droplets, 3), dtype=np.float64)
    position[:, 0] = np.linspace(10, 50, num_droplets)  # x
    position[:, 1] = np.linspace(5, 25, num_droplets)  # y
    position[:, 2] = np.linspace(20, 100, num_droplets)  # z

    results = [TamocDroplet(*params) for params in zip(mass_flux,
                                                       radius,
                                                       density,
                                                       position)]

    return results


class TamocSpill(gnome.spill.spill.BaseSpill):
    """
    Models a spill

    TODO: we should not be using complex multidemensional values as
          parameter defaults such as the one used for 'tamoc_parameters'
    """
    def __init__(self,
                 release_time=None,
                 start_position=None,
                 num_elements=None,
                 end_release_time=None,
                 name='TAMOC plume',
                 TAMOC_interval=None,
                 on=True,
                 tamoc_parameters={'depth': 2000.,
                                   'diameter': 0.3,
                                   'release_flowrate': 20000.,
                                   'release_temp': 273.15 + 150,
                                   'release_phi': (-np.pi / 2.),
                                   'release_theta': 0.,
                                   'discharge_salinity': 0.,
                                   'tracer_concentration': 1.,
                                   'hydrate': True,
                                   'dispersant': False,
                                   'sigma_fac': np.array([[1.], [1. / 200.]]),
                                   'inert_drop': False,
                                   'd50_gas': 0.008,
                                   'd50_oil': 0.0038,
                                   'nbins': 20,
                                   'nc_file': './Input/case_01',
                                   'fname_ctd': './Input/ctd_api.txt',
                                   'ua': np.array([0.05, 0.05]),
                                   'va': np.array([0.06, 0.06]),
                                   'wa': np.array([0.01, 0.01]),
                                   'depths': np.array([0, 1])},
                 data_sources={'currents': None,
                               'salinity': None,
                               'temperature': None}
                 ):
        super(TamocSpill, self).__init__()

        self.release_time = release_time
        self.start_position = start_position
        self.num_elements = num_elements
        self.end_release_time = end_release_time
        self.num_released = 0
        self.amount_released = 0.0

        if TAMOC_interval is not None:
            self.tamoc_interval = timedelta(hours=TAMOC_interval)
        else:
            self.tamoc_interval = None

        self.last_tamoc_time = release_time
        self.droplets = None
        self.on = on  # spill is active or not
        self.name = name
        self.tamoc_parameters = tamoc_parameters
        self.data_sources = data_sources

    def update_environment_conditions(self, current_time):
        ds = self.data_sources
        if ds['currents'] is not None:
            currents = ds['currents']
            u_data = currents.variables[0].data
            v_data = currents.variables[1].data
            source_idx = None

            try:
                source_idx = currents.grid.locate_faces(np.array(self.start_position)[0:2], 'node')
            except TypeError:
                source_idx = currents.grid.locate_faces(np.array(self.start_position)[0:2])

            if currents.grid.node_lon.shape[0] == u_data.shape[-1]:
                # lon/lat are inverted in data so idx must be reversed
                source_idx = source_idx[::-1]

            print source_idx
            time_idx = currents.time.index_of(current_time, False)
            print time_idx
            u_conditions = u_data[time_idx, :, source_idx[0], source_idx[1]]
            max_depth_ind = np.where(u_conditions.mask)[0].min()
            u_conditions = u_conditions[0:max_depth_ind]
            v_conditions = v_data[time_idx, 0:max_depth_ind,
                                  source_idx[0], source_idx[1]]

            self.tamoc_parameters['ua'] = u_conditions
            self.tamoc_parameters['va'] = v_conditions
            print 'getdepths'

            try:
                self.tamoc_parameters['depths'] = u_data._grp['depth_levels'][0:max_depth_ind]
            except IndexError:
                self.tamoc_parameters['depths'] = u_data._grp['depth'][0:max_depth_ind]

        if ds['salinity'] is not None:
            pass

        if ds['temperature'] is not None:
            pass

    def run_tamoc(self, current_time, time_step):
        """
        runs TAMOC if no droplets have been initialized or if current_time has
        reached last_tamoc_run + interval
        """
        if self.on:
            if self.tamoc_interval is None:
                if self.last_tamoc_time is None:
                    self.last_tamoc_time = current_time
                    self.droplets, self.diss_components = self._run_tamoc()
                return self.droplets

            if (current_time >= self.release_time and
                    (self.last_tamoc_time is None or self.droplets is None) or
                    current_time >= self.last_tamoc_time + self.tamoc_interval and
                    current_time < self.end_release_time):
                self.last_tamoc_time = current_time
                self.droplets, self.diss_components = self._run_tamoc()

        return self.droplets

    def _run_tamoc(self):
        """
        this is the code that actually calls and runs tamoc_output

        it returns a list of TAMOC droplet objects
        """
        # Release conditions

        tp = self.tamoc_parameters

        # Release depth (m)
        z0 = tp['depth']
        # Release diameter (m)
        D = tp['diameter']
        # Release flowrate (bpd)
        Q = tp['release_flowrate']
        # Release temperature (K)
        T0 = tp['release_temp']
        # Release angles of the plume (radians)
        phi_0 = tp['release_phi']
        theta_0 = tp['release_theta']
        # Salinity of the continuous phase fluid in the discharge (psu)
        S0 = tp['discharge_salinity']
        # Concentration of passive tracers in the discharge (user-defined)
        c0 = tp['tracer_concentration']

        # List of passive tracers in the discharge
        chem_name = 'tracer'
        # Presence or abscence of hydrates in the particles

        hydrate = tp['hydrate']
        # Presence or absence of dispersant
        dispersant = tp['dispersant']
        # Reduction in interfacial tension due to dispersant
        # sigma_fac[0] - for gas; sigma_fac[1] - for liquid
        sigma_fac = tp['sigma_fac']
        # Define liquid phase as inert
        inert_drop = tp['inert_drop']
        # d_50 of gas particles (m)
        d50_gas = tp['d50_gas']
        # d_50 of oil particles (m)
        d50_oil = tp['d50_oil']
        # number of bins in the particle size distribution
        nbins = tp['nbins']

        # Create the ambient profile needed for TAMOC
        # name of the nc file
        nc_file = tp['nc_file']

        # Define and input the ambient ctd profiles
        fname_ctd = tp['fname_ctd']

        # Define and input the ambient velocity profile
        ua = tp['ua']
        va = tp['va']
        wa = tp['wa']
        depths = tp['depths']

        profile = self.get_profile(nc_file, fname_ctd, ua, va, wa, depths)

        # Get the release fluid composition
        fname_composition = './Input/API_2000.csv'
        composition, mass_frac = self.get_composition(fname_composition)

        # Read in the user-specified properties for the chemical data
        data, units = chem.load_data('./Input/API_ChemData.csv')
        oil = dbm.FluidMixture(composition, user_data=data)

        # oil.delta = self.load_delta('./Input/API_Delta.csv',oil.nc)

#        if np.sum(oil.delta==0.):
#            print 'Binary interaction parameters are zero, estimating them.'
#            # Estimate the values of the binary interaction parameters
#            oil.delta = self.estimate_binary_interaction_parameters(oil)

        # Get the release rates of gas and liquid phase
        md_gas, md_oil = self.release_flux(oil, mass_frac, profile, T0, z0, Q)
        print 'md_gas, md_oil', np.sum(md_gas), np.sum(md_oil)
        # Get the particle list for this composition
        particles = self.get_particles(composition, data,
                                       md_gas, md_oil, profile,
                                       d50_gas, d50_oil,
                                       nbins, T0, z0,
                                       dispersant, sigma_fac, oil, mass_frac,
                                       hydrate, inert_drop)
        print len(particles)
        print particles

        # Run the simulation
        jlm = bpm.Model(profile)
        jlm.simulate(np.array([0., 0., z0]),
                     D, None, phi_0, theta_0, S0, T0, c0,
                     chem_name, particles,
                     track=False, dt_max=60., sd_max=6000.)

        # Update the plume object with the nearfiled terminal level answer
        jlm.q_local.update(jlm.t[-1], jlm.q[-1],
                           jlm.profile, jlm.p, jlm.particles)

        Mp = np.zeros((len(jlm.particles), len(jlm.q_local.M_p[0])))
        gnome_particles = []
        gnome_diss_components = []
#        print jlm.particles
        m_tot_nondiss = 0.
        for i in range(len(jlm.particles)):
            nb0 = jlm.particles[i].nb0
            Tp = jlm.particles[i].T
            Mp[i, 0:len(jlm.q_local.M_p[i])] = (jlm.q_local.M_p[i][:] /
                                                jlm.particles[i].nbe)

            mass_flux = np.sum(Mp[i, :] * jlm.particles[i].nb0)
            density = jlm.particles[i].rho_p

            radius = (jlm.particles[i].diameter(Mp[i, 0:len(jlm.particles[i].m)], Tp,
                                                jlm.q_local.Pa,
                                                jlm.q_local.S,
                                                jlm.q_local.T)) / 2.

            position = np.array([jlm.particles[i].x,
                                 jlm.particles[i].y,
                                 jlm.particles[i].z])

            # Calculate the equlibrium and get the particle phase
            Eq_parti = dbm.FluidMixture(composition=jlm.particles[i].composition[:],
                                        user_data=data)

            # Get the particle equilibrium at the plume termination conditions
            print 'Insitu'
            flag_phase_insitu = self.get_phase(jlm.profile,
                                               Eq_parti,
                                               Mp[i, :] / np.sum(Mp[i, :]),
                                               Tp,
                                               jlm.particles[i].z)

            # Get the particle equilibrium at the 15 C and 1 atm
            print 'Surface'
            flag_phase_surface = self.get_phase(jlm.profile,
                                                Eq_parti,
                                                Mp[i, :] / np.sum(Mp[i, :]),
                                                273.15 + 15.,
                                                0.)
            gnome_particles.append(TamocDroplet(mass_flux, radius, density,
                                                position))

        for p in gnome_particles:
            print p

        m_tot_diss = 0.

        # Calculate the dissolved particle flux
        for j in range(len(jlm.chem_names)):
            diss_mass_flux = (jlm.q_local.c_chems[j] *
                              np.pi *
                              jlm.q_local.b ** 2 *
                              jlm.q_local.V)
            m_tot_diss += diss_mass_flux

            position = np.array([jlm.q_local.x, jlm.q_local.y, jlm.q_local.z])
            chem_name = jlm.q_local.chem_names[j]
            gnome_diss_components.append(TamocDissMasses(diss_mass_flux,
                                                         position,
                                                         chem_name))

        print ('total dissolved mass flux at plume termination {}\n'
               'total non-dissolved mass flux at plume termination {}\n'
               'total mass flux tracked at plume termination {}\n'
               'total mass flux released at the orifice {}\n'
               'percentage_error {}'
               .format(m_tot_diss,
                       m_tot_nondiss,
                       m_tot_diss + m_tot_nondiss,
                       np.sum(md_gas) + np.sum(md_oil),
                       ((np.sum(md_gas) + np.sum(md_oil) -
                         m_tot_diss - m_tot_nondiss) /
                        (np.sum(md_gas) + np.sum(md_oil)) * 100.)))

        # Now, we will generate the GNOME properties for a weatherable particle
        # For now, computed at the release location:
        # The pressure at release:
        P0 = profile.get_values(z0, ['pressure'])
        K_ow, json_oil = self.translate_properties_gnome_to_tamoc(md_oil,
                                                                  composition,
                                                                  oil,
                                                                  P0, S0,
                                                                  T=288.15)

        return gnome_particles, gnome_diss_components

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}()'
                .format(self))

    def _get_mass_distribution(self, mass_fluxes, time_step):
        ts = time_step
        delta_masses = []
        for flux in mass_fluxes:
            delta_masses.append(flux * ts)
        total_mass = sum(delta_masses)
        proportions = [d_mass / total_mass for d_mass in delta_masses]

        return (delta_masses, proportions, total_mass)

    @property
    def units(self):
        """
        Default units in which amount of oil spilled was entered by user.
        The 'amount' property is returned in these 'units'
        """
        return self._units

    @units.setter
    def units(self, units):
        """
        set default units in which volume data is returned
        """
        self._check_units(units)  # check validity before setting
        self._units = units

    def _check_units(self, units):
        """
        Checks the user provided units are in list of valid volume
        or mass units
        """

        if (units in self.valid_vol_units or
                units in self.valid_mass_units):
            return True
        else:
            msg = ('Units for amount spilled must be in volume or mass units. '
                   'Valid units for volume: {0}, for mass: {1}'
                   .format(self.valid_vol_units, self.valid_mass_units))
            ex = uc.InvalidUnitError(msg)
            self.logger.exception(ex, exc_info=True)

            raise ex  # this should be raised since run will fail otherwise

    # what is this for??
    def get_mass(self, units=None):
        '''
        Return the mass released during the spill.
        User can also specify desired output units in the function.
        If units are not specified, then return in 'SI' units ('kg')
        If volume is given, then use density to find mass. Density is always
        at 15degC, consistent with API definition
        '''
        # first convert amount to 'kg'
        if self.units in self.valid_mass_units:
            mass = uc.convert('Mass', self.units, 'kg', self.amount_released)

        if units is None or units == 'kg':
            return mass
        else:
            self._check_units(units)
            return uc.convert('Mass', 'kg', units, mass)

    def uncertain_copy(self):
        """
        Returns a deepcopy of this spill for the uncertainty runs

        The copy has everything the same, including the spill_num,
        but it is a new object with a new id.

        Not much to this method, but it could be overridden to do something
        fancier in the future or a subclass.

        There are a number of python objects that cannot be deepcopied.
        - Logger objects

        So we copy them temporarily to local variables before we deepcopy
        our Spill object.
        """
        u_copy = copy.deepcopy(self)
        self.logger.debug(self._pid + "deepcopied spill {0}".format(self.id))

        return u_copy

    def rewind(self):
        """
        rewinds the release to original status (before anything has been
        released).
        """
        self.num_released = 0
        self.amount_released = 0

        # don't want to run tamoc on every rewind!
        self.last_tamoc_time = None

    def num_elements_to_release(self, current_time, time_step):
        """
        Determines the number of elements to be released during:
        current_time + time_step

        It invokes the num_elements_to_release method for the the unerlying
        release object: self.release.num_elements_to_release()

        :param current_time: current time
        :type current_time: datetime.datetime
        :param int time_step: the time step, sometimes used to decide how many
            should get released.

        :returns: the number of elements that will be released. This is taken
            by SpillContainer to initialize all data_arrays.
        """
        if not self.on:
            return 0

        if (current_time < self.release_time or
                current_time > self.end_release_time):
            return 0

        self.droplets = self.run_tamoc(current_time, time_step)

        duration = (self.end_release_time - self.release_time).total_seconds()
        if duration is 0:
            duration = 1
        LE_release_rate = self.num_elements / duration
        num_to_release = int(LE_release_rate * time_step)
        if self.num_released + num_to_release > self.num_elements:
            num_to_release = self.num_elements - self.num_released

        return num_to_release

        # return self.release.num_elements_to_release(current_time, time_step)

    def set_newparticle_values(self, num_new_particles, current_time,
                               time_step, data_arrays):
        """
        SpillContainer will release elements and initialize all data_arrays
        to default initial value. The SpillContainer gets passed as input and
        the data_arrays for 'position' get initialized correctly by the release
        object: self.release.set_newparticle_positions()

        If a Spill Amount is given, the Spill object also sets the 'mass' data
        array; else 'mass' array remains '0'

        :param int num_new_particles: number of new particles that were added.
            Always greater than 0
        :param current_time: current time
        :type current_time: datetime.datetime
        :param time_step: the time step, sometimes used to decide how many
            should get released.
        :type time_step: integer seconds
        :param data_arrays: dict of data_arrays provided by the SpillContainer.
            Look for 'positions' array in the dict and update positions for
            latest num_new_particles that are released
        :type data_arrays: dict containing numpy arrays for values

        Also, the set_newparticle_values() method for all element_type gets
        called so each element_type sets the values for its own data correctly
        """
        mass_fluxes = [tam_drop.mass_flux for tam_drop in self.droplets]
        delta_masses, proportions, total_mass = self._get_mass_distribution(mass_fluxes, time_step)

        # set up LE distribution,
        # the number of particles in each 'release point'
        LE_distribution = [int(num_new_particles * p) for p in proportions]
        diff = num_new_particles - sum(LE_distribution)
        for i in range(0, diff):
            LE_distribution[i % len(LE_distribution)] += 1

        # compute release point location for each droplet
        positions = [self.start_position +
                     FlatEarthProjection.meters_to_lonlat(d.position,
                                                          self.start_position)
                     for d in self.droplets]

        for p in positions:
            p[0][2] -= self.start_position[2]

        # for each release location, set the position and mass
        # of the elements released at that location
        total_rel = 0
        for mass_dist, n_LEs, pos, droplet in zip(delta_masses,
                                                  LE_distribution,
                                                  positions, self.droplets):
            start_idx = -num_new_particles + total_rel

            if start_idx == 0:
                break
            end_idx = start_idx + n_LEs
            if end_idx == 0:
                end_idx = None

            if start_idx == end_idx:
                continue

            data_arrays['positions'][start_idx:end_idx] = pos
            data_arrays['mass'][start_idx:end_idx] = mass_dist / n_LEs
            data_arrays['init_mass'][start_idx:end_idx] = mass_dist / n_LEs
            data_arrays['density'][start_idx:end_idx] = droplet.density
            data_arrays['droplet_diameter'][start_idx:end_idx] = \
                np.random.normal(droplet.radius * 2,
                                 droplet.radius * 0.15,
                                 (n_LEs))

            v = data_arrays['rise_vel'][start_idx:end_idx]
            rise_velocity_from_drop_size(v,
                                         data_arrays['density'][start_idx:end_idx],
                                         data_arrays['droplet_diameter'][start_idx:end_idx],
                                         0.0000013, 1020)
            data_arrays['rise_vel'][start_idx:end_idx] = v
            total_rel += n_LEs

        self.num_released += num_new_particles
        self.amount_released += total_mass

    def get_profile(self, nc_name, fname, u_a, v_a, w_a, depths):
        """
        Read in the ambient CTD data

        Read in the CTD data specified by API for all test cases.  Append the
        velocity information to the CTD file.

        Parameters
        ----------
        nc_name : str
        Name to call the netCDF4 dataset.
        u_a : float
        Crossflow velocity for this test case (m/s).

        Returns
        -------
        profile : `ambient.Profile` object
        Returns an `ambient.Profile` object of the ambient CTD and velocity
        information

        """
        # Get the ambient CTD data
        names = ['z', 'temperature', 'salinity', 'oxygen']
        units = ['m', 'deg C', 'psu', 'mmol/m^3']
        data = np.loadtxt(fname, comments='%')

        # Convert the data to standard units
        M_o2 = 31.9988 / 1000.  # kg/mol
        data[:, 3] = data[:, 3] / 1000. * M_o2
        units[3] = 'kg/m^3'
        data, units = ambient.convert_units(data, units)

        # Create an empty netCDF4 dataset to store the CTD dat
        summary = 'Global horizontal mean hydrographic and oxygen data'
        source = 'Taken from page 226 of Sarmiento and Gruber'
        sea_name = 'Global'
        p_lat = 0.
        p_lon = 0.
        p_time = date2num(datetime(1998, 1, 1, 1, 0, 0),
                          units='seconds since 1970-01-01 00:00:00 0:00',
                          calendar='julian')
        nc = ambient.create_nc_db(nc_name, summary, source, sea_name, p_lat,
                                  p_lon, p_time)

        # Insert the data into the netCDF dataset
        comments = ['average', 'average', 'average', 'average']
        nc = ambient.fill_nc_db(nc, data, names, units, comments, 0)

        # Compute the pressure and insert into the netCDF dataset
        P = ambient.compute_pressure(data[:, 0], data[:, 1], data[:, 2], 0)
        P_data = np.vstack((data[:, 0], P)).transpose()
        nc = ambient.fill_nc_db(nc, P_data,
                                ['z', 'pressure'],
                                ['m', 'Pa'],
                                ['average', 'computed'], 0)

        # Create an ambient.Profile object from this dataset
        profile = ambient.Profile(nc, chem_names='all')

        # Force the max depth to model
        # depths[-1] = profile.z_max

        # Add the crossflow velocity

        print '******************'
        print depths
        print '******************'

        u_crossflow = np.zeros((len(depths), 2))
        u_crossflow[:, 0] = depths

        if u_a.shape != depths.shape:
            u_crossflow[:, 1] = np.linspace(u_a[0], u_a[-1], len(depths))
        else:
            u_crossflow[:, 1] = u_a

        symbols = ['z', 'ua']
        units = ['m', 'm/s']
        comments = ['provided', 'provided']
        profile.append(u_crossflow, symbols, units, comments, 0)

        v_crossflow = np.zeros((len(depths), 2))
        v_crossflow[:, 0] = depths

        if v_a.shape != depths.shape:
            v_crossflow[:, 1] = np.linspace(v_a[0], v_a[-1], len(depths))
        else:
            v_crossflow[:, 1] = v_a

        symbols = ['z', 'va']
        units = ['m', 'm/s']
        comments = ['provided', 'provided']
        profile.append(v_crossflow, symbols, units, comments, 0)

        w_crossflow = np.zeros((len(depths), 2))
        w_crossflow[:, 0] = depths

        if w_a.shape != depths.shape:
            w_crossflow[:, 1] = np.linspace(w_a[0], w_a[-1], len(depths))
        else:
            w_crossflow[:, 1] = w_a

        symbols = ['z', 'wa']
        units = ['m', 'm/s']
        comments = ['provided', 'provided']
        profile.append(w_crossflow, symbols, units, comments, 0)

        # Finalize the profile (close the nc file)
        profile.close_nc()

        # Return the final profile
        return profile

    def get_composition(self, fname):

        composition = []
        mass_frac = []

        with open(fname) as datfile:
            for line in datfile:
                # Get a line of data
                entries = line.strip().split(',')
                print entries

                # Excel sometimes addes empty columns...remove them.
                if len(entries[len(entries) - 1]) is 0:
                    entries = entries[0:len(entries) - 1]

                if line.find('%') >= 0:
                    # This is a header line...ignore it
                    pass

                else:
                    composition.append(entries[0])
                    print type(entries[1])
                    mass_frac.append(np.float64(entries[1]))

        # Return the release composition data
        return (composition, mass_frac)

    def release_flux(self, oil, mass_frac, profile, T0, z0, Q):
        """
        Calulate the release flux

        """
        # Compute the phase equilibrium at the surface
        m0, xi, K = oil.equilibrium(mass_frac, 273.15 + 15., 101325.)

        # Get the mass flux of oil
        rho_o = oil.density(m0[1, :], 273.15 + 15., 101325.)[1, 0]
        md_o = Q * 0.15899 * rho_o / 24. / 60. / 60.

        # The amount of gas coming with that volume flux of oil is determined
        # by the equilibrium
        rho_g = oil.density(m0[0, :], 273.15 + 15., 101325.)[0, 0]
        md_g = np.sum(m0[0, :]) / np.sum(m0[1, :]) * md_o

        # Get the total mass flux of each component in the mixture
        m_tot = mass_frac / np.sum(mass_frac) * (md_o + md_g)

        # Compute the GOR as a check
        V_o = md_o / rho_o / 0.15899  # bbl/s
        V_g = md_g / rho_g * 35.3147  # ft^3/s

        # Determine the mass fluxes at the release point
        P = profile.get_values(z0, ['pressure'])
        m0, xi, K = oil.equilibrium(m_tot, T0, P)
        md_gas = m0[0, :]
        md_oil = m0[1, :]

        # Return the total mass flux of gas and oil at the release
        return (md_gas, md_oil)

    def get_particles(self, composition, data,
                      md_gas0, md_oil0, profile,
                      d50_gas, d50_oil, nbins,
                      T0, z0, dispersant, sigma_fac,
                      oil, mass_frac, hydrate, inert_drop):
        """
        docstring for get_particles

        """

        # Reduce surface tension if dispersant is applied
        if dispersant is True:
            sigma = np.array([[1.], [1.]]) * sigma_fac
        else:
            sigma = np.array([[1.], [1.]])

        # Create DBM objects for the bubbles and droplets
        bubl = dbm.FluidParticle(composition, fp_type=0,
                                 sigma_correction=sigma[0], user_data=data)
        drop = dbm.FluidParticle(composition, fp_type=1,
                                 sigma_correction=sigma[1], user_data=data)

        # Get the local ocean conditions
        T, S, P = profile.get_values(z0,
                                     ['temperature', 'salinity', 'pressure'])
        rho = seawater.density(T, S, P)

        # Get the mole fractions of the released fluids
        molf_gas = bubl.mol_frac(md_gas0)
        molf_oil = drop.mol_frac(md_oil0)
        print molf_gas
        print molf_oil

        # Use the Rosin-Rammler distribution to get the mass flux in each
        # size class
        # de_gas, md_gas = sintef.rosin_rammler(nbins, d50_gas,
        #                                       np.sum(md_gas0),
        #                                       bubl.interface_tension(md_gas0,
        #                                                              T0,
        #                                                              S, P),
        #                                       bubl.density(md_gas0, T0, P),
        #                                       rho)
        # de_oil, md_oil = sintef.rosin_rammler(nbins, d50_oil,
        #                                       np.sum(md_oil0),
        #                                       drop.interface_tension(md_oil0,
        #                                                              T0,
        #                                                              S, P),
        #                                       drop.density(md_oil0, T0, P),
        #                                       rho)

        # Get the user defined particle size distibution
        de_oil, vf_oil, de_gas, vf_gas = self.userdefined_de()
        md_gas = np.sum(md_gas0) * vf_gas
        md_oil = np.sum(md_oil0) * vf_oil

        # Define a inert particle to be used if inert liquid particles are use
        # in the simulations
        molf_inert = 1.
        isfluid = True
        iscompressible = True
        rho_o = drop.density(md_oil0, T0, P)
        inert = dbm.InsolubleParticle(isfluid, iscompressible,
                                      rho_p=rho_o, gamma=40., beta=0.0007,
                                      co=2.90075e-9)

        # Create the particle objects
        particles = []
        t_hyd = 0.

        # Bubbles
        for i in range(nbins):
            if md_gas[i] > 0.:
                m0, T0, nb0, P, Sa, Ta = dispersed_phases.initial_conditions(
                    profile, z0, bubl, molf_gas, md_gas[i], 2, de_gas[i], T0)
                # Get the hydrate formation time for bubbles
                if hydrate is True and dispersant is False:
                    t_hyd = dispersed_phases.hydrate_formation_time(bubl,
                                                                    z0, m0, T0,
                                                                    profile)
                    if np.isinf(t_hyd):
                        t_hyd = 0.
                else:
                    t_hyd = 0.

                particles.append(bpm.Particle(0., 0., z0, bubl,
                                              m0, T0, nb0,
                                              1.0, P, Sa, Ta,
                                              K=1., K_T=1., fdis=1.e-6,
                                              t_hyd=t_hyd))

        # Droplets
        for i in range(len(de_oil)):
            # Add the live droplets to the particle list
            if md_oil[i] > 0. and not inert_drop:
                m0, T0, nb0, P, Sa, Ta = dispersed_phases.initial_conditions(
                    profile, z0, drop, molf_oil, md_oil[i], 2, de_oil[i], T0)
                # Get the hydrate formation time for bubbles
                if hydrate is True and dispersant is False:
                    t_hyd = dispersed_phases.hydrate_formation_time(drop,
                                                                    z0, m0, T0,
                                                                    profile)
                    if np.isinf(t_hyd):
                            t_hyd = 0.
                else:
                    t_hyd = 0.

                particles.append(bpm.Particle(0., 0., z0, drop,
                                              m0, T0, nb0, 1.0, P, Sa, Ta,
                                              K=1., K_T=1., fdis=1.e-6,
                                              t_hyd=t_hyd))

            # Add the inert droplets to the particle list
            if md_oil[i] > 0. and inert_drop is True:
                m0, T0, nb0, P, Sa, Ta = dispersed_phases.initial_conditions(
                    profile, z0, inert, molf_oil, md_oil[i], 2, de_oil[i], T0)

                particles.append(bpm.Particle(0., 0., z0, inert,
                                              m0, T0, nb0, 1.0, P, Sa, Ta,
                                              K=1., K_T=1., fdis=1.e-6,
                                              t_hyd=0.))

        # Define the lambda for particles
        model = params.Scales(profile, particles)

        for j in range(len(particles)):
            particles[j].lambda_1 = model.lambda_1(z0, j)

        # Return the particle list
        return particles

    def userdefined_de(self):

        # Load the particle sizes
        fname = './Input/Particles_de.csv'

        de_details = np.zeros([100, 4])
        k = 0
        with open(fname, 'rU') as datfile:
            datfile.readline()
            datfile.readline()
            datfile.readline()
            for row in datfile:
                row = row.strip().split(",")
                for i in range(len(row)):
                    de_details[k, i] = float(row[i])
                k += 1

        de_oil = np.zeros([100, 1])
        de_gas = np.zeros([100, 1])
        vf_oil = np.zeros([100, 1])
        vf_gas = np.zeros([100, 1])

        de_oil = de_details[:, 0] / 1000.
        de_gas = de_details[:, 2] / 1000.
        vf_oil = de_details[:, 1]
        vf_gas = de_details[:, 3]

        return (de_oil[de_oil > 0.],
                vf_oil[vf_oil > 0.],
                de_gas[de_gas > 0.],
                vf_gas[vf_gas > 0.])

    def get_phase(self, profile, particle, Mp, T, z):
        """
        This get the equilibrium composition of the particle at adefined
        temperature T and pressure P

        """
        # Get the pressure at particle location
        P = profile.get_values(z, ['pressure'])
        print 'Pressure', P

        # Get the equilibrium composition
        m0, xi, K = particle.equilibrium(Mp, T, P)

        print 'liquid fraction', np.sum(m0[1, :])
        print 'gas fraction', np.sum(m0[0, :])

        if np.sum(m0[1, :]) == 1.0:
            print ' Particle is complete liquid'
            flag_phase = 'Liquid'
        elif np.sum(m0[0, :]) == 1.0:
            print 'particle is complete gas'
            flag_phase = 'Gas'
        else:
            print 'particle is a mixture of gas and liquid'
            flag_phase = 'Mixture'

        return (flag_phase)

    def estimate_binary_interaction_parameters(self, oil):
        '''
        Estimates values of the binary interaction parameters.

        Parameters
        ----------
        oil : dbm.FluidMixture
            a TAMOC oil object

        Returns
        -------
        delta : ndarray, size (nc,nc)
            a matrix containing the estimated binary interaction parameters

        Notes
        -----
        Valid for hydrocarbon-hydrocarbon interaction.

        Uses the Pedersen method for the binary interaction parameters:
        Pedersen et al. "On the danger of "tuning" equation of state
        parameters", 1985. Eqs. 2 and 3.
        (Note: Riazi's ASTM book cite the method but rounds the coefficient to
        one significant digit without explanation. Here the original value
        from Pedersen et al. is used (0.00145).)

        '''
        # Initialize the matrix
        delta = np.zeros((len(oil.M), len(oil.M)))
        # Populate the matrix with the estimates:
        for yy in range(len(oil.M)):
            for tt in range(len(oil.M)):
                if not (tt == yy):
                    delta[yy, tt] = 0.00145 * np.max((oil.M[tt]/oil.M[yy],
                                                      oil.M[yy]/oil.M[tt]))

        return delta

    def load_delta(self, file_name, nc):
        """
        Loads the binary interaction parameters.

        Parameters
        ----------
        file_name : string
            file name
        nc : int
            number of components in the mixture

        Returns
        -------
        delta : ndarray, size (nc,nc)
           a matrix containing the loaded binary interaction parameters
        """
        delta = np.zeros([nc, nc])
        k = 0
        with open(file_name, 'r') as datfile:
            for row in datfile:
                row = row.strip().split(",")
                for i in range(len(row)):
                    delta[k, i] = float(row[i])
                k += 1

        return (delta)

    def translate_properties_gnome_to_tamoc(self, md_oil, composition, oil,
                                            P, Sa, T=288.15):
        '''
        Translates properties from TAMOC components to GNOME components.

        Generates a GNOME weatherable substance, and computes the oil-water
        partition coefficients.

        Parameters
        ----------
        md_oil : ndarray, size (nc)
            masses of each component in a mixture (kg)
        composition : list of strings, size (nc)
            names of the components in TAMOC
        oil: a dbm.FluidMixture
            the oil of interest
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)
        Sa : float
            water salinity of the ambient seawater (psu)

        Returns
        -------
        K_ow : ndarray, (size (nc)
            the oil-water partition coefficients according to TAMOC
        json_oil : GNOME oil substance
            the GNOME substance generated using the estimates of properties
            from tamoc.

        Notes
        -----
        When exiting a TAMOC simulation, each droplet size has its own
        composition, hence its own properties if computed at local conditions.
        It is likely the best to provide the function with the composition
        at the emission source, same for T and P.

        BEWARE: we compute key properties (e.g. densities) at
        288.15 K because this is the GNOME default. Except if the user inputs
        a lower T.

        '''

        print '- - - - - - - - - -'

        # Let's get the partial densities in liquid for each component:
        # (Initialize the array:)
        densities = np.zeros(len(composition))

        # We will compute component densities at 288.15 K_T, except if the
        # user has input a lower T. A higher T is not allowed.
        # (In deep waters, droplets should cool very fast, it is not a
        # reasonable assumption to compute at a high T.)
        T_rho = np.min([288.15, T])

        # Check that we have no gas phase at this conditions:
        m_, xi, K = oil.equilibrium(md_oil, T_rho, P)

        if np.sum(m_, 1)[0] > 0.:
            # The mixture would separate in a gas and a liquid phase at
            # equilibrium. Let's use the composition of the liquid phase:
            md_oil = m_[1]

        # density of the bulk oil at release conditions:
        rho_0 = oil.density(md_oil, T_rho, P)[1]

        # Now, we will remove/add a little mass of a component, and get its
        # partial density as the ratio of the change of mass divided by
        # change of oil volume.
        for ii in range(len(densities)):  # (We do a loop over each component)
            # We will either remove 1% or add 1% mass (and we choose the one
            # that keeps the mixture as a liquid):
            add_or_remove = np.array([.99, 1.01])
            for tt in range(len(add_or_remove)):
                # Factor used to remove/add mass of just component i:
                m_multiplication_factors = np.ones(len(densities))

                # We remove or add 1% of the mass of component i:
                m_multiplication_factors[ii] = add_or_remove[tt]
                m_i = md_oil * m_multiplication_factors

                # Make an equilibrium calculation to check that we did not
                # generate a gas phase:
                m_ii, xi, K = oil.equilibrium(m_i, T_rho, P)
                print T_rho, P

                # If we did not generate a gas phase, stop here. Else we will
                # do the for loop a second time using the second value in
                # 'add_or_remove'
                if np.sum(m_ii, 1)[0] == 0.:
                    break

            # We compute the density of the new mixture:
            rho_i = oil.density(m_i, T_rho, P)[1]

            # we get the partial density of each component as:
            # (DELTA(Mass) / DELTA(Volume)):
            densities[ii] = ((np.sum(md_oil) - np.sum(m_i)) /
                             (np.sum(md_oil) / rho_0 - np.sum(m_i) / rho_i))

        print ('TAMOC density: {} '
               'and estimated from component densities: {}'
               .format(rho_0,
                       (np.sum(md_oil) / np.sum(md_oil / densities))))

        # Note: the (np.sum(md_oil)/np.sum(md_oil/densities)) makes sense
        # physically: density = SUM(MASSES) / SUM(VOLUMES) (Assuming volume
        # of mixing is zero, which is a very good assumption for petroleum
        # liquids)
        # This is the GNOME-way, though less physically-grounded.
        print ('However GNOME would somehow estimate the density as '
               'm_i * rho_i: {}'
               .format(np.sum(md_oil * densities / np.sum(md_oil))))
        print 'densities: ', densities

        # Normalize densities so that the GNOME-way to compute density gives
        # the TAMOC density for the whole oil:
        densities = (densities *
                     rho_0 /
                     np.sum(md_oil * densities / np.sum(md_oil)))
        print ('GNOME value after normalizing densities: {}'
               .format(np.sum(md_oil * densities / np.sum(md_oil))))

        print composition
        print 'densities: ', densities
        print 'MW: ', oil.M
        print 'Tb: ', oil.Tb
        print 'delta: ', oil.delta

        # Now oil properties:
        oil_viscosity = oil.viscosity(md_oil, T_rho, P)[1]
        oil_density = oil.density(md_oil, T_rho, P)[1]
        oil_interface_tension = oil.interface_tension(md_oil, T_rho, Sa, P)[1]

        # Compute the oil-water partition coefficients, K_ow:
        C_oil = md_oil / (np.sum(md_oil) / oil.density(md_oil, T_rho, P)[1])
        C_water = oil.solubility(md_oil, T, P, Sa)[1]

        K_ow = C_oil / C_water
        print 'K_ow : {}'.format(K_ow)

        # Below, we will assume that any component having a K_ow that is not
        # inf is a 'Aromatics' (it may not be a component corresponding to
        # aromatics compounds. But it contains soluble compounds. Labeling it
        # as 'Aromatics' should enable GNOME to deal with it.)

        # Now, create a GNOME substance with these data:
        json_object = dict()
        # We need to create a list of dictionaries containing the molecular
        # weights:
        molecular_weights_dict_list = []

        for i in range(len(oil.M)):
            # This is the dictionary for the current component:
            current_dict = dict()

            # Populate the keys of the dictionary with corresponding values:
            if not np.isinf(K_ow[i]):
                current_dict['sara_type'] = 'Aromatics'
            else:
                current_dict['sara_type'] = 'Saturatess'

            # BEWARE: GNOME wants g/mol and TAMOC has kg/mol.
            current_dict['g_mol'] = oil.M[i] * 1000.
            current_dict['ref_temp_k'] = oil.Tb[i]

            # append each dictionary to the list of dictionarries:
            molecular_weights_dict_list.append(current_dict)

        json_object['molecular_weights'] = molecular_weights_dict_list

        # Now do the same for the cuts:
        cuts_dict_list = []

        for i in range(len(oil.M)):
            # This is the dictionary for the current component:
            current_dict = dict()

            # Populate the keys of the dictionary with corresponding values:
            current_dict['vapor_temp_k'] = oil.Tb[i]
            current_dict['fraction'] = md_oil[i]

            # append each dictionary to the list of dictionarries:
            cuts_dict_list.append(current_dict)

        json_object['cuts'] = cuts_dict_list
        json_object['oil_seawater_interfacial_tension_ref_temp_k'] = T_rho
        json_object['oil_seawater_interfacial_tension_n_m'] = oil_interface_tension[0]

        # Now do the same for the densities:
        densities_dict_list = []

        for i in range(len(oil.M)):
            # This is the dictionary for the current component:
            current_dict = dict()
            # Populate the keys of the dictionary with corresponding values:
            current_dict['density'] = densities[i]
            if not np.isinf(K_ow[i]):
                current_dict['sara_type'] = 'Aromatics'
            else:
                current_dict['sara_type'] = 'Saturatess'

            current_dict['ref_temp_k'] = oil.Tb[i]
            # append each dictionary to the list of dictionarries:
            densities_dict_list.append(current_dict)

        json_object['sara_densities'] = densities_dict_list

        # This one is for the density of the oil as a whole:
        oil_density_dict = dict()
        oil_density_dict['ref_temp_k'] = T_rho  # a priori 288.15
        oil_density_dict['kg_m_3'] = oil_density[0]
        oil_density_dict['weathering'] = 0.
        json_object['densities'] = [oil_density_dict]

        # This one is for the viscosity of the oil as a whole:
        # Note: 'dvis' in GNOME is the dynamic viscosity
        #       called 'viscosity' in TAMOC
        oil_viscosity_dict = dict()
        oil_viscosity_dict['ref_temp_k'] = T_rho  # a priori 288.15
        oil_viscosity_dict['kg_ms'] = oil_viscosity[0]
        oil_viscosity_dict['weathering'] = 0.

        json_object['dvis'] = [oil_viscosity_dict]
        json_object['name'] = 'test TAMOC oil'

        # Now do the same for the sara dractions:
        SARA_dict_list = []

        for i in range(len(oil.M)):
            # This is the dictionary for the current component:
            current_dict = dict()

            # Populate the keys of the dictionary with corresponding values:
            if not np.isinf(K_ow[i]):
                current_dict['sara_type'] = 'Aromatics'
            else:
                current_dict['sara_type'] = 'Saturatess'

            current_dict['ref_temp_k'] = oil.Tb[i]
            current_dict['fraction'] = md_oil[i]

            # append each dictionary to the list of dictionarries:
            SARA_dict_list.append(current_dict)

        json_object['sara_fractions'] = SARA_dict_list
        # print json_object

        from oil_library.models import Oil
        json_oil = Oil.from_json(json_object)

        print json_oil.densities

        # print json_oil.dvis
        # Hum. Oil has no attribute 'dvis', but 'kvis' is empty. Is that a bug?
        print ('interfacial tension: ({}, {})'
               .format(json_oil.oil_seawater_interfacial_tension_n_m,
                       oil_interface_tension))
        print json_oil.molecular_weights
        print json_oil.sara_fractions
        print json_oil.cuts
        print json_oil.densities

        # TO ELUCIDATE: IS IT NORMAL THAT THE FIELDS OF json_oil ARE NOT
        #               THE SAME AS WHEN AN OIL IS IMPORTED FROM THE
        #               OIL DATABASE USING get_oil??

        # NOTE: I CANNOT DO THIS BELOW, THIS IS ONLY FOR OILS IN THE DATABASE.
        # from oil_library import get_oil, get_oil_props
        # uuu = get_oil_props(json_oil.name)
        # print ('oil density from our new created substance: {} or same: {}'
        #        .format(np.sum(uuu.mass_fraction * uuu.component_density),
        #                uuu.density_at_temp()))
        # print 'component densities: ',uuu.component_density
        # print 'component mass fractions: ',uuu.mass_fraction
        # print 'component molecular weights: ',uuu.molecular_weight
        # print 'component boiling points: ',uuu.boiling_point
        # print 'API: ',uuu.api
        # print 'KINEMATIC viscosity: ',uuu.kvis_at_temp()

        # tested the K_ow with benzene and toluene and ethylbenzene
        # oil = dbm.FluidMixture(['benzene','toluene','ethylbenzene'])
        # md_oil = np.array([1.,1.,1.])
        # C_oil = md_oil / (np.sum(md_oil) / oil.density(md_oil, T_rho, P)[1])
        # C_water = oil.solubility(md_oil, T_rho, P, Sa)[1]
        # K_ow = C_oil / C_water
        # from gnome.utilities.weathering import BanerjeeHuibers
        # K_ow2 = BanerjeeHuibers.partition_coeff(oil.M * 1000.,
        #                                         oil.density(md_oil,
        #                                                     T_rho, P)[1])
        # print 'K_ow :'
        # print K_ow
        # print K_ow2

        return (K_ow, json_oil)
