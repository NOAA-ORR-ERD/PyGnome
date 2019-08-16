"""
gnome_objs.py
------------

This script defines a new Blowout object that is intended to provide TAMOC
integration to PyGNOME.  This object is built from the tools already defined
in blowout.py, a script created for the MPRI project to quickly simulate a 
blowout in TAMOC.  

Where possible, this module will import tools that already exist in GNOME to
make the integration as seemless as possible.  

S. Socolofsky
socolofs@tamu.edu
August 5, 2019

"""

from tamoc import dbm, bent_plume_model, dispersed_phases

import tamoc_utilities as tu
import adios_utilities as au

from datetime import datetime
from netCDF4 import date2num

import numpy as np

class TamocBlowout(object):
    """
    Object to simulate blowouts using TAMOC integrated with GNOME
        
    This object creates the inputs required to run a blowout simulation 
    in TAMOC, provides methods to conduct and analyze the simulation, and
    includes methods to pass data to GNOME for simulation of the far
    field.  
    
    Attributes
    ----------
    z0 : float
        Depth of the blowout source (m)
    d0 : float
        Diameter of the equivalent circular area describing the cross-
        section of the blowout orifice (m)
    adios_id : str
        Adios OilLibrary unique string ID number
    q_oil : float
        Oil flow rate expressed as flow rate of liquid petroleum compounds
        at standard conditions (bbl/d)
    gor : float
        Gas-to-oil ratio at standard conditions expressed in standard
        petroleum engineering units of standard cubic feet per stock 
        barrel of oil (std ft^3/bbl)
    
    """
    
    def __init__(self, 
                 z0,
                 d0,
                 adios_id,
                 q_oil,
                 gor, 
                 x0=0.,
                 y0=0.,
                 u0=0.,
                 phi_0=-np.pi / 2.,
                 theta_0 = 0.
                ):
        """
        Initializer for a tamoc_blowout object.        
        
        """
        super(TamocBlowout, self).__init__()
        
        # Store required class attributes passed to initializer
        self.z0 = z0
        self.d0 = d0
        self.adios_id = adios_id
        self.q_oil = q_oil
        self.gor = gor
        
        # Store optional class attributes
        self.x0 = x0
        self.y0 = y0
        self.u0 = u0
        self.phi_0 = phi_0
        self.theta_0 = theta_0
        self.ca = ['nitrogen', 'oxygen', 'argon', 'carbon_dioxide']
        
        # Create the remaining object attributes needed to made a TAMOC
        # simulation
        self._update()
    
    
    def _update(self):
        """
        Generate input data for a TAMOC model run
        
        Creates the input data required for a TAMOC model simulation based on
        the present object attributes.  This function updates values for each
        of the following object attributes.
        
        Attributes
        ----------
        profile : `tamoc.ambient.Profile`
            A `tamoc.ambient.Profile` object with the ambient ocean water 
            column data
        oil : `tamoc.dbm.FluidMixture`
            A discrete bubble model `tamoc.dbm.FluidMixture` object that 
            contains the property data for the desired live oil mixture.
        mass_flux : np.array
            An array of gas and liquid mass fluxes for each chemical 
            component in the mixture (kg/s) required to achieve the desired 
            flow rate of dead oil at the surface
        T0 : float
            Ambient ocean temperature at the release (K)
        S0 : float
            Ambient ocean salinity at the release (psu)
        P0 : float
            Ambient ocean presure at the release (Pa)
        Sj : float
            Salinity of the fluids discharging with the blowout (psu)
        Tj : float
            Temperature of the fluids discharging with the blowout (K)
        cj : float
            Concentration of passive tracers in the jet (kg/m^3)
        tracers : str
            Name of the passive tracers in the jet
        gas : `tamoc.dbm.FluidParticle` object
            A `tamoc.dbm.FluidParticle` object that contains the properties
            for the free gas phase at the release
        liq : `tamoc.dbm.FluidParticle` object
            A `tamoc.dbm.FluidParticle` object that contains the properties
            of the liquid phase petroleum at the release
        d_gas : np.array
            Array of gas bubble sizes (m)
        vf_gas : np.array
            Array of volume fractions for each size in the gas bubble size 
            array.  This array should sum to 1.0.
        d_liq : np.array
            Array of oil droplet sizes (m)
        vf_liq : np.array
            Array of volume fractions for each size in the oil droplet size 
            array.  This array should sum to 1.0.        
        disp_phases : list
            A Python list that contains `tamoc.disp_phases.Particle` 
            objects.  These objects contain each bubble and droplet class
            in the simulations.
        track : bool
            Parameter of the `tamoc.bent_plume_model.Model` that sets 
            whether or not the particles should be tracked sub-surface
            by TAMOC (True), or whether sub-surface particle tracking will
            be done by GNOME (False).
        dt_max : float
            Maximum number of seconds to simulate for the plume simulation
        sd_max : float
            Maximum distance to simulate along the plume centerline 
            measured as number of orifice diameters.
        bpm : `tamoc.bent_plume_model.Model`
            A `tamoc.bent_plume_model.Model` object that is ready to 
            simulate the present conditions.
        
        """
        # Create the profile object
        fname = '../data/CTD_data.dat'
        summary = 'Sample blowout simulation offshore Newfoundland'
        source = 'Model output from north Atlantic model by Youyu Lu'
        sea_name = 'Atlantic'
        p_lat = 46.7624
        p_lon = -48.0762
        p_time = date2num(datetime(2018, 11, 6, 12, 0, 0), 
        units = 'seconds since 1970-01-01 00:00:00 0:00', 
        calendar = 'julian')
        self.profile = tu.get_ctd_from_txt(fname, summary, source, 
                                           sea_name, p_lat, p_lon, p_time, 
                                           self.ca)
        
        # Import the oil with the desired gas-to-oil ratio
        self.oil, self.mass_flux = tu.get_adios_oil(self.adios_id, 
                                                    self.q_oil, 
                                                    self.gor,
                                                    self.ca)
        
        # Use the profile object to get the ocean conditions at the release
        self.T0, self.S0, self.P0 = self.profile.get_values(self.z0, 
                                       ['temperature',
                                        'salinity',
                                        'pressure'])
        
        # Define the remaining initial conditions
        self.Sj = 0.
        self.Tj = self.T0
        self.cj = 1.
        self.tracers = 'tracer'
        
        # Compute the equilibrium mixture properties at the release
        m, xi, K = self.oil.equilibrium(self.mass_flux, self.Tj, self.P0)
        
        # Create dbm.FluidParticle objects for gas and liquid petroleum
        self.gas = dbm.FluidParticle(self.oil.composition, 
                                     fp_type=0, 
                                     delta=self.oil.delta, 
                                     user_data=self.oil.user_data)
        self.liq = dbm.FluidParticle(self.oil.composition, 
                                     fp_type=1, 
                                     delta=self.oil.delta,
                                     user_data=self.oil.user_data)

        # Get the bubble and droplet volume size distributions
        self.d_gas, self.vf_gas, self.d_liq, self.vf_liq = psd()
        
        # Initialize an empty particle list
        self.disp_phases = []
        
        # Add the bubbles to the particles in the simulation
        self.disp_phases += particles(np.sum(m[0,:]), self.d_gas, 
                                      self.vf_gas, self.profile, self.gas, 
                                      xi[0,:], 0., 0., self.z0, self.Tj, 
                                      0.9, False)
        
        # Add the oil droplets to the particles in the simulation
        self.disp_phases += particles(np.sum(m[1,:]), self.d_liq, 
                                      self.vf_liq, self.profile, self.liq, 
                                      xi[1,:], 0., 0., self.z0, self.Tj, 
                                      0.98, False)
        
        # Set some of the default parameters
        self.track = True
        self.dt_max = 3. * 3600.
        self.sd_max = 3 * self.z0 / self.d0
        
        # Create the `tamoc.bent_plume_model.Model` object
        self.bpm = bent_plume_model.Model(self.profile)
        
        # Record flag that update is complete
        self.update = True
    
    
    def update_release_depth(self, z0):
        """
        Change the release depth (m) to use in a model simulation
        """
        self.z0 = z0
        self.update = False
        self.bpm.sim_stored = False
    
    def update_orifice_diameter(self, d0):
        """
        Change the orifice diametr (m) to use in a model simulation
        """
        self.d0 = d0
        self.update = False
        self.bpm.sim_stored = False
    
    def update_adios_id(self, adios_id):
        """
        Change the OilLibrary ID number to use in a model simulation
        """
        self.adios_id = adios_id
        self.update = False
        self.bpm.sim_stored = False
    
    def update_q_oil(self, adios_id):
        """
        Change the oil flow rate (bbl/d) to use in a model simulation
        """
        self.q_oil = q_oil
        self.update = False
        self.bpm.sim_stored = False
    
    def update_gor(self, gor):
        """
        Change the gas-to-oil ratio (std ft^3/bbl) to use in a model 
        simulation
        """
        self.gor = gor
        self.update = False
        self.bpm.sim_stored = False
    
    def simulate(self):
        """
        Run a `tamoc.bent_plum_model.Model` simulation of the present
        parameter set.
        """
        # Update the model attributes if they have changed
        if not self.update:
            self._update()
        
        # Run the new simulation
        self.bpm.simulate(np.array([self.x0, self.y0, self.z0]), 
                          self.d0, 
                          self.u0,
                          self.phi_0, 
                          self.theta_0, 
                          self.Sj,
                          self.Tj,
                          self.cj,
                          self.tracers,
                          self.disp_phases,
                          self.track,
                          self.dt_max,
                          self.sd_max)
    
    def plot_state_space(self, fignum=1):
        """
        Plot the `tamoc.bent_plume_model` state space solution
        """
        if self.bpm.sim_stored is False:
            print 'No simulation results available to analyze...'
            print 'Run TamocBlowout.simulate() first.\n'
            return
        
        self.bpm.plot_state_space(fignum)
    
    def plot_all_variables(self, fignum=2):
        """
        Plot all variables for the `tamoc.bent_plume_model` solution
        """
        if self.bpm.sim_stored is False:
            print 'No simulation results available to analyze...'
            print 'Run TamocBlowout.simulate() first.\n'
            return
        
        self.bpm.plot_all_variables(fignum)
    
    def analyze_results(self):
        """
        Report a table of post-processed information for the blowout
        """
        if self.bpm.sim_stored is False:
            print 'No simulation results available to analyze...'
            print 'Run TamocBlowout.simulate() first.\n'
            return
        
        analyze(self.bpm, self.oil, self.mass_flux)
        
    
""" Helper functions """

def particles(m_tot, d, vf, profile, oil, yk, x0, y0, z0, Tj, lambda_1, 
              lag_time):
    """
    Create particles to add to a bent plume model simulation
    
    Creates bent_plume_model.Particle objects for the given particle 
    properties so that they can be added to the total list of particles
    in the simulation.
    
    Parameters
    ----------
    m_tot : float
        Total mass flux of this fluid phase in the simulation (kg/s)
    d : np.array
        Array of particle sizes for this fluid phase (m)
    vf : np.array
        Array of volume fractions for each particle size for this fluid 
        phase (--).  This array should sum to 1.0.
    profile : ambient.Profile
        An ambient.Profile object with the ambient ocean water column data
    oil : dbm.FluidParticle
        A dbm.FluidParticle object that contains the desired oil database 
        composition
    yk : np.array
        Mole fractions of each compound in the chemical database of the oil
        dbm.FluidParticle object (--).
    x0, y0, z0 : floats
        Initial position of the particles in the simulation domain (m)
    Tj : float
        Initial temperature of the particles in the jet (K)
    lambda_1 : float
        Value of the dispersed phase spreading parameter of the jet integral
        model (--).
    lag_time : bool
        Flag that indicates whether (True) or not (False) to use the
        biodegradation lag times data.
              
    """
    # Create an empty list of particles
    disp_phases = []
    
    # Add each particle in the distribution separately
    for i in range(len(d)):
        
        # Get the total mass flux of this fluid phase for the present 
        # particle size
        mb0 = vf[i] * m_tot
        
        # Get the properties of these particles at the source
        (m0, T0, nb0, P, Sa, Ta) = dispersed_phases.initial_conditions(
            profile, z0, oil, yk, mb0, 2, d[i], Tj)
        
        # Append these particles to the list of particles in the simulation
        disp_phases.append(bent_plume_model.Particle(x0, y0, z0, oil, m0, T0, 
            nb0, lambda_1, P, Sa, Ta, K=1., K_T=1., fdis=1.e-6, t_hyd=0., 
            lag_time=lag_time))
    
    # Return the list of particles
    return disp_phases


def analyze(bpm, oil, masses):
    """
    Analyzes the results of a bent plume model simulation
    
    Parameters
    ----------
    bpm : bent_plume_model.Model object
    
    """
    # Compute the initial fluxes of oil and gas
    T = 273.15 + 15.
    P = 101325.
    m, xi, K = oil.equilibrium(masses, T, P)
    p_gas = oil.density(m[0,:], T, P)[0,0]
    md_gas = np.sum(m[0,:])
    q0_gas = md_gas / p_gas / 0.0283168 * 86400.
    p_oil = oil.density(m[1,:], T, P)[1,0]
    md_oil = np.sum(m[1,:])
    q0_oil = md_oil / p_oil / 0.158987 * 86400.
    
    # Create empty lists to store the gas and liquid particle results
    r = [[], []]
    md = [[], []]
    q = [[], []]
    
    # Loop through each particle in the simulation
    for particle in bpm.particles:
        
        # Get properties at the final position of particles that were
        # simulated by the single bubble model above an intruding plume
        if particle.farfield == True:
            
            pos = -1
            
            # Position and mass
            x = particle.sbm.y[pos,0:3]
            t = particle.sbm.t[pos]
            m = particle.sbm.y[pos,3:-1]
            
            # Ambient conditions
            Ta, Sa, P = bpm.profile.get_values(x[2], ['temperature', 
                'salinity', 'pressure'])
            
            # Particle properties
            (us, rho_p, A, Cs, beta, beta_T, T) = \
                particle.sbm.particle.properties(m, Ta, P, Sa, Ta, t)
            
            # Check if the particle dissolved
            if us == 0.:
                dissolved = True
            else:
                dissolved = False
            
            # Mass flux of particle
            m_p = np.sum(m) * particle.nb0
            v_p = m_p / rho_p
            print rho_p, particle.nb0, m_p
            
            # Type of particle
            fp_type = particle.sbm.particle.particle.fp_type
            
            # Record the tracked statistics
            if not dissolved:
                r[fp_type].append(np.sqrt(x[0]**2 + x[1]**2))
                md[fp_type].append(m_p)
                q[fp_type].append(v_p)
        
        # Also get properties of particles that stayed inside the plume if 
        # the plume surfaced.
        if particle.z <= -0.1:
            
            # Position and mass
            x = np.array([particle.x, particle.y, particle.z])
            t = particle.t
            m = particle.m
            
            # Ambient conditions
            Ta, Sa, P = bpm.profile.get_values(x[2], ['temperature', 
                'salinity', 'pressure'])
            
            # Mass flux of particle
            m_p = np.sum(m) * particle.nb0
            v_p = m_p / particle.rho_p
            
            # Type of particle
            fp_type = particle.particle.fp_type
            
            # Record the tracked statistics
            r[fp_type].append(np.sqrt(x[0]**2 + x[1]**2))
            md[fp_type].append(m_p)
            q[fp_type].append(v_p)
            
    
    r_gas = np.array(r[0])
    r_oil = np.array(r[1])
    m_gas = np.array(md[0])
    m_oil = np.array(md[1])
    q_gas = np.array(q[0])
    q_oil = np.array(q[1])
    
    # Report the results
    print '\nSimulation Results Summary:'
    print '---------------------------'
    print '\nRelease Conditions: \n'
    print '    Oil flow rate (bbl/d)  : ', q0_oil
    print '    GOR (ft^3/bbl)         : ', q0_gas / q0_oil
    print '\nGas Bubbles: \n'
    if len(r_gas) > 0:
        print '    Release rate at STP (kg/s)        : ', md_gas
        print '    Surfacing region (m)              : ', np.min(r_gas), \
            ' to '
        print '                                        ', np.max(r_gas)
        print '    Mass flux to atmosphere (kg/s)    : ', np.sum(m_gas)
        print '    Volume flux to atmosphere (m^3/s) : ', np.sum(q_gas)
        print '    Fraction surfacing (--)           : ', \
            np.sum(m_gas) / md_gas
    else:
        print '    None of the gas surfaced.'
    print '\nOil Droplets: \n'
    if len(r_oil) > 0:
        print '    Release rate at STP (kg/s)     : ', md_oil
        print '    Surfacing region (m)           : ', np.min(r_oil), ' to '
        print '                                     ', np.max(r_oil)
        print '    Mass flux to surface (kg/s)    : ', np.sum(m_oil)
        print '    Volume flux to surface (m^3/s) : ', np.sum(q_oil)
        print '    Fraction surfacing (--)        : ', \
            np.sum(m_oil) / md_oil
    else:
        print '    None of the oil surfaced.'


def psd():
    """
    Get the particle size distributions
    
    Gets the gas bubble and oil droplet volume size distributions and 
    returns the sizes and volume fractions for each.
    
    Returns
    -------
    d_gas : np.array
        Array of gas bubble sizes (m)
    vf_gas : np.array
        Array of volume fractions for each size in the gas bubble size array.
        This array should sum to 1.0.
    d_liq : np.array
        Array of oil droplet sizes (m)
    vf_liq : np.array
         Array of volume fractions for each size in the oil droplet size 
        array.  This array should sum to 1.0.
    
    """
    # Gas bubble size distribution
    d_gas = np.linspace(0.1, 10., 100) / 1000.
    vf_gas = np.array([4.67E-04, 1.43E-03, 2.78E-03, 4.29E-03, 5.92E-03, 
        7.56E-03, 9.03E-03, 1.05E-02, 1.17E-02, 1.29E-02, 1.38E-02, 1.47E-02,
        1.54E-02, 1.60E-02, 1.67E-02, 1.70E-02, 1.73E-02, 1.76E-02, 1.76E-02,
        1.78E-02, 1.79E-02, 1.81E-02, 1.82E-02, 1.82E-02, 1.81E-02, 1.80E-02,
        1.77E-02, 1.76E-02, 1.75E-02, 1.76E-02, 1.75E-02, 1.75E-02, 1.75E-02,
        1.72E-02, 1.71E-02, 1.69E-02, 1.66E-02, 1.63E-02, 1.59E-02, 1.55E-02,
        1.49E-02, 1.42E-02, 1.36E-02, 1.29E-02, 1.21E-02, 1.14E-02, 1.07E-02,
        9.95E-03, 9.27E-03, 8.75E-03, 8.26E-03, 7.88E-03, 7.65E-03, 7.51E-03,
        7.51E-03, 7.64E-03, 7.83E-03, 8.18E-03, 8.57E-03, 9.02E-03, 9.50E-03,
        1.00E-02, 1.05E-02, 1.09E-02, 1.13E-02, 1.17E-02, 1.18E-02, 1.20E-02,
        1.19E-02, 1.18E-02, 1.16E-02, 1.12E-02, 1.07E-02, 1.01E-02, 9.49E-03,
        8.80E-03, 8.02E-03, 7.23E-03, 6.44E-03, 5.66E-03, 4.88E-03, 4.16E-03,
        3.49E-03, 2.86E-03, 2.31E-03, 1.83E-03, 1.41E-03, 1.06E-03, 7.80E-04,
        5.54E-04, 3.79E-04, 2.49E-04, 1.56E-04, 9.18E-05, 5.03E-05, 2.52E-05,
        1.12E-05, 4.32E-06, 1.44E-06, 9.11E-07])
    
    # Fix round-off error
    err = 1.0 - np.sum(vf_gas)
    nmax = np.argmax(vf_gas)
    vf_gas[nmax] = vf_gas[nmax] + err
    
    # Oil droplet size distribution
    d_liq = np.linspace(0.1, 10., 100) / 1000.
    vf_liq = np.array([1.26E-03, 3.15E-03, 5.62E-03, 8.14E-03, 1.07E-02, 
        1.31E-02, 1.50E-02, 1.68E-02, 1.81E-02, 1.93E-02, 2.01E-02, 2.09E-02,
        2.14E-02, 2.17E-02, 2.21E-02, 2.21E-02, 2.19E-02, 2.20E-02, 2.16E-02,
        2.16E-02, 2.14E-02, 2.12E-02, 2.10E-02, 2.06E-02, 2.01E-02, 1.98E-02,
        1.94E-02, 1.91E-02, 1.88E-02, 1.87E-02, 1.83E-02, 1.81E-02, 1.78E-02,
        1.72E-02, 1.68E-02, 1.64E-02, 1.57E-02, 1.51E-02, 1.44E-02, 1.37E-02,
        1.28E-02, 1.20E-02, 1.13E-02, 1.05E-02, 9.80E-03, 9.18E-03, 8.65E-03,
        8.20E-03, 7.82E-03, 7.64E-03, 7.49E-03, 7.46E-03, 7.55E-03, 7.67E-03,
        7.88E-03, 8.14E-03, 8.36E-03, 8.65E-03, 8.88E-03, 9.07E-03, 9.20E-03,
        9.28E-03, 9.30E-03, 9.17E-03, 9.01E-03, 8.76E-03, 8.41E-03, 8.02E-03,
        7.53E-03, 7.02E-03, 6.47E-03, 5.88E-03, 5.30E-03, 4.70E-03, 4.13E-03,
        3.59E-03, 3.06E-03, 2.58E-03, 2.15E-03, 1.76E-03, 1.41E-03, 1.12E-03,
        8.70E-04, 6.62E-04, 4.94E-04, 3.60E-04, 2.55E-04, 1.76E-04, 1.18E-04,
        7.58E-05, 4.69E-05, 2.77E-05, 1.55E-05, 8.06E-06, 3.87E-06, 1.68E-06,
        6.36E-07, 2.01E-07, 5.10E-08, 1.67E-08])
    
    # Fix round-off error
    err = 1.0 - np.sum(vf_liq)
    nmax = np.argmax(vf_liq)
    vf_liq[nmax] = vf_liq[nmax] + err
    
    # Return the particle size distributions
    return (d_gas, vf_gas, d_liq, vf_liq)

    