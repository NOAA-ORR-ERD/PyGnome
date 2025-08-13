"""
TAMOC Spill Module
==================

This module creates a `gnome.spill.Spill` object that supports simulations
using the ``TAMOC`` `bent_plume_model` to simulate a blowout or subsea oil
spill. This module handles all of the work of a normal `gnome.spill.Spill`
object, which includes initializing the spill, releasing new Lagrangian
elements each model time step (while the spill is still active) and, in the
present case, re-running the ``TAMOC`` plume simulations periodically as
ambient ocean conditions change.

To handle the ``TAMOC`` side of the simulation, this module relies mostly on
the `blowout` module of ``TAMOC``, which provides an interface to the
`bent_plume_model` that is optimized for subsea blowouts. This module
provides for computing the gas-oil equilibrium at the source, creating an
initial gas bubble and oil droplet size distribution, and generating all of
the necessary initial conditions for a bent plume model simulation.

See Also
--------
tamoc.blowout, tamoc.dbm_utilities, tamoc.particle_size_models,
gnome.spill.Spill, gnome.tamoc.scripting

Notes
-----
The idea behind the ``TAMOC`` `blowout` module and the present
`gnome.tamoc.spill` module is to define oil and gas properties using normal
units and terminology in petroleum engineering. Hence, the oil flow rate is
specified as a dead oil (all gas removed) flow rate at surface conditions --
this is generally the production rate of crude oil for a well -- and the gas
flow rate is specified as the gas produced (standard cubic feed) per barrel
of dead oil -- this is called the GOR in units of ft^3/bbl. Then, ``TAMOC``
does the work of creating this oil and gas mixture and computing the in situ
flow rates of oil and gas at the release.

Currently, initial oil droplet size distributions are computed using the
SINTEF equations (Johansen et al., MPB, 2013) updated with new fit
coefficients (Johansen et al. API Technical Reports) and the d_50 rule
(reducing the estimated d_50 such that the 95th percentile droplet size does
not exceed the maximum stable droplet size). The initial gas bubble size
distributions are computed using the method in Wang et al. (2018). Both of
these procedures use empirical equations to compute the volume median
particle size and then estimate a distribution from an assumed probability
distribution and spreading coefficient; spreading parameters are selected
based on the measurements in the above references and assumed to be scale
invariant between the laboratory and the field.

Oil property data is specified using an Adios ID name to an oil in the NOAA
``OilLibrary`` module. Gas is added to the oil following the specified GOR by
assuming a standard natural gas composition dominated by methane. For
different gas compositions, the `mix_gas_for_gor()` method in the
`tamoc.dbm_utilities` module would need to be updated.

"""
# S. Socolofsky, March 25, 2020, Texas A&M University, <socolofs@tamu.edu>

from __future__ import (absolute_import, division, print_function)

from tamoc import ambient
from tamoc.blowout import Blowout

#import unit_conversion as uc
import nucos
import gnome.scripting as gs

from gnome.spills.spill import Spill
from gnome.array_types import gat
from gnome.utilities import projections

from datetime import datetime, timedelta
import numpy as np

class TamocSpill(Spill):
    """
    A GNOME Spill object that manages a TAMOC `bent_plume_model` simulation
    
    Defines a Spill class for GNOME that manages simulations of a blowout
    using the TAMOC `bent_plume_model` through the `tamoc.blowout.Blowout`
    class interface.
    
    Attributes
    ----------
    num_elements_per_oil_element : float, default=2
        Number of GNOME Lagrangian Element to release each model time step
        per liquid droplet element in the TAMOC solution. 
    spill_position : tuple, default=(0.0, 0.0, 100.)
        Position of the spill in longitude (deg), latitude (deg) and depth
        (m).
    release_time : datetime.datetime object, default=datetime.now()
        Start time of the blowout release
    release_duration : datetime.timedelta, default=timedelta(hours=12)
        Duration of the blowout release
    gnome_model_time_step : float, default=900
        Time step used for particle tracking in the GNOME model
    tamoc_time_delta : datetime.timedelta, default=timedelta(hours=1)
        Interval between repeated TAMOC simulations.  TAMOC is a steady-state
        jet integral model.  The solution from TAMOC should be updated 
        whenever the ambient water column conditions or spill flow rate 
        changes enough that the previous model solution becomes out-of-date
    substance : str, default='AD01554' (Louisiana Light Sweet)
        Description of the dead-oil composition in the spill.  See Notes 
        below for details.  
    release_rate : float, default=20000.
        Release rate of the dead oil composition at the release point.  This
        is generally in stock barrels of oil per day.  Other options are,
        e.g., m^3/s, kg/s
    release_units : str, default='bbl/day'
        Units for the release variable.
    gor : float, default=0.
        Gas to oil ratio at standard surface conditions in standard cubic 
        feet per stock barrel of oil
    d0 : float, default=0.1
        Equivalent circular diameter of the release (m)
    phi_0 : float, default=-np.pi / 2. (vertical release)
        Vertical angle of the release relative to the horizontal plane; z is
        positive down so that -pi/2 represents a vertically upward flowing
        release (rad)
    theta_0 : float, default=0.
        Horizontal angle of the release relative to the x-direction (rad)
    num_gas_elements : int, default=10
        Number of gas bubble sizes to include in the gas bubble size
        distribution
    num_oil_elements : int, default=25
        Number of oil droplet sizes to include in the oil droplet size
        distribution
    water : various, default=None
        Ambient water conditions as represented in GNOME.  Several options
        are available.  See Notes below for details.  
    currents : ndarray or GridCurrent, default=np.array([0.1, 0., 0.])
        Ambient water velocity data as represented in GNOME.  See Notes 
        below for details.  
    on : bool, default=True
        Boolean variable indicating whether the spill object is active
    name : str, default='TAMOC Blowout'
        Name for this spill release
    
    Notes
    -----
    The spilled substance can either be taken from the NOAA OilLibrary or
    can be created from individual pseudo-components in TAMOC.  The user may 
    define the `substance` in one of two ways:
    
    substance : str
        Provide a unique OilLibrary ID number from the NOAA Python 
        OilLibrary package
    substance : dict
        Use the chemical properties database provided with TAMOC.  In this
        case, use the dictionary keyword `composition` to pass a list 
        of chemical property names and the keyword `masses` to pass a 
        list of mass fractions for each component in the composition
        list.  If the masses variable does not sum to unity, this function
        will compute an equivalent mass fraction that does.
    
    Likewise, the ambient water column data can be provided through several
    different options.  The `water` variable contains temperature and salinity
    data.  The user may define the `water` in the following ways:
    
    water : None
        Indicates that we have no information about the ambient temperature
        or salinity.  In this case, the model will import data for the 
        world-ocean average.
    water : dict
        If we only know the water temperature and salinity at the surface, 
        this may be passed through a dictionary with keywords `temperature`
        and `salinity`.  In this case, the model will import data for the
        world-ocean average and adjust the data to have the given temperature
        and salinity at the surface.
    water : str
        If we stored the water column profile in a text file, we may provide
        the file path to this text file via the string stored in water. This
        file should contain columns in the following order: depth (m),
        temperature (deg C), salinity (psu), velocity in the x-direction
        (m/s), velocity in the y-direction (m/s). Since this option includes
        the currents, the current variable will be ignored in this case.  A
        comment string of `#` may be used in the text file.
    water : gs.Water
        Use a gnome.environment Water object to pass surface temperature
        and salinity.  This object will convert the data to a `dict` and
        will be used as for a dictionary above.
    water : list
        If water column data are available as GridTemperature and GridSalinity
        gnome.environment objects, these may be contained in a list and
        passed as input.  This object uses the .at() method to extract the
        profile data and then builds an array of depth, temperature, and
        salinity data.
    water : ndarray
        Finally, profile data may be passed as a numpy array containing 
        depth (m), temperature (K), and salinity (psu).
    
    Finally, current profile data can be provided through several different
    options.  The user may define the `current` in the following ways:
    
    current : float
        This is assumed to be the current velocity along the x-axis and will
        be uniform over the depth
    current : ndarray
        This is assumed to contain the current velocity in the x- and y- (and
        optionally also z-) directions. If this is a one-dimensional array,
        then these currents will be assumed to be uniform over the depth. If
        this is a multi-dimensional array, then these values are assumed to
        contain a profile of data, with the depth (m) as the first column of
        data.
    current : list
        Use a list to contain a gnome.environment GridCurrent object.  This
        provides flexibility in case an IceAwareCurrent is used.  As with the
        GridTemperature and GridCurrent data, this object uses the .at()
        method to extract the profile data and then builds an array of 
        depth and velocity data.
    
    """
    def __init__(self, 
                 num_elements_per_oil_element=2,
                 spill_position=(0.0, 0.0, 100.), 
                 release_time=datetime.now(),
                 release_duration=timedelta(hours=12),
                 gnome_model_time_step=900, 
                 tamoc_time_delta=timedelta(hours=1),
                 substance='AD01554',
                 release_rate=20000.,
                 release_units='bbl/day',
                 gor=0.,
                 d0=0.1,
                 phi_0=-np.pi / 2.,
                 theta_0=0.,
                 num_gas_elements=10,
                 num_oil_elements=25,
                 water=None,
                 current=np.array([0.1, 0., 0.]),
                 on=True,
                 name='TAMOC Blowout',
                 **kwargs):
        
        # Compute the total amount spilled from the release rate and duration
        duration = release_duration.total_seconds()
        try:
            amount = nucos.convert(release_units, 'm^3/s', release_rate) * \
                        duration
            units = 'm^3'
        except uc.NotSupportedUnitError:
            amount = nucos.convert(release_units, 'kg/s', release_rate) * \
                        duration
            units = 'kg'
        
        # Compute the total number of elements to release
        frac_releases = float(release_duration.total_seconds() / \
            gnome_model_time_step)
        if frac_releases == int(frac_releases):
            num_releases = int(frac_releases)
        else:
            num_releases = int(frac_releases) + 1
        num_elements = num_elements_per_oil_element * num_oil_elements * \
                       num_releases
        
        # Create a GNOME oil with the given .json database
        # TODO: figure out how to initialize the substance...assume substance
        #       is a GnomeOil
        #gnome_oil = gs.GnomeOil(filename=substance)
        gnome_oil = substance
        
        # Send required data to the base Spill class for instantiation
        super(TamocSpill, self).__init__(num_elements=num_elements,
                                         amount=amount,
                                         units=units,
                                         substance=gnome_oil,
                                         release=None,
                                         water=water,
                                         on=on,
                                         name=name,
                                         **kwargs)
        
        # Store object attributes unique to TamocBlowout
        self.release_time = release_time
        self.end_release_time = gs.asdatetime(release_time) + \
                                   release_duration
        self.release_rate = release_rate
        self.release_units = release_units
        self.water = water
        self.current = current
        self.tamoc_time_delta = tamoc_time_delta
        self.gnome_oil = gnome_oil
        
        # Compute some of the attributes related to the number of elements
        self.num_elements_per_oil_element = num_elements_per_oil_element
        self.num_elements = num_elements
        total_time = (self.end_release_time - 
                      self.release_time).total_seconds()
        self.num_elements_per_second = float(self.num_elements) / total_time
        
        # Initialize a TAMOC model blowout simulation object
        self.spill_position = spill_position
        x0, y0, z0 = spill_position
        # Use the GNOME flat-earth projection
        self.project = projections.FlatEarthProjection()
        self.ref_pt = np.array([x0, y0, 0.])
        # Set TAMOC to (0,0) in meters relative to ref_pt
        x0 = 0.
        y0 = 0.
        
        # Format the ambient water and current data for TAMOC
        self.update_tamoc_profile(self.water,
                                  self.current,
                                  self.release_time)
        
        # Create the `tamoc` Blowout object to manage `tamoc` simualtions
        self.z0 = z0
        self.d0 = d0
        self.tamoc_oil = substance
        self.release_rate = release_rate
        self.gor = gor
        self.x0 = x0
        self.y0 = y0
        self.phi_0 = phi_0
        self.theta_0 = theta_0
        self.num_gas_elements = num_gas_elements
        self.num_oil_elements = num_oil_elements
        self.tamoc_sim = Blowout(z0=self.z0,
                                 d0=self.d0,
                                 substance=self.tamoc_oil,
                                 q_oil=self.release_rate,
                                 gor=self.gor,
                                 x0=self.x0,
                                 y0=self.y0,
                                 u0=None,
                                 phi_0=self.phi_0,
                                 theta_0=self.theta_0,
                                 num_gas_elements=self.num_gas_elements,
                                 num_oil_elements=self.num_oil_elements, 
                                 water=self.tamoc_water,
                                 current=self.tamoc_current)
        
        # Tell the spill container what array types need to be included
        # when receiving a TAMOC Lagrangian element into a GNOME model
        # simulation
        self.array_types.update(
            {'positions' : gat('positions'),
             'init_mass' : gat('init_mass'),
             'mass' : gat('mass'),
             'mass_components' : gat('mass_components'),
             'density' : gat('density'),
             'viscosity' : gat('density'),
             'droplet_diameter': gat('droplet_diameter'), 
             'rise_vel' : gat('rise_vel')
            }
        )
    
    def rewind(self):
        """
        Rewinds the release to its original status (before anything has been 
        released).
        
        """
        # Reset the GNOME elements array
        self.array_types.update(
            {'positions' : gat('positions'),
             'init_mass' : gat('init_mass'),
             'mass' : gat('mass'),
             'mass_components' : gat('mass_components'),
             'density' : gat('density'),
             'viscosity' : gat('density'),
             'droplet_diameter': gat('droplet_diameter'), 
             'rise_vel' : gat('rise_vel')
            }
        )
        self._num_released = 0
    
    def prepare_for_model_run(self, timestep):
        """
        Do anything that needs to happen before the first time-step... last 
        thing to happen before the simulation starts.
        
        """
        # <currently, there is nothing to do>
        pass
    
    def update_tamoc_profile(self, water, current, current_time):
        """
        Update the environmental forcing for a TAMOC simulation
        
        Use the water and current data defined in GNOME and and format the
        data as needed by the `tamoc.blowout.Blowout` object.  
        
        Parameters
        ----------
        water : various, default=None
            Ambient water conditions as represented in GNOME.  Several options
            are available.  See Notes at start of class for details.  
        currents : ndarray or GridCurrent, default=np.array([0.1, 0., 0.])
            Ambient water velocity data as represented in GNOME.  Several 
            options are available.  See Notes at start of class for details.
        current_time : : datetime.datetime
            Current real time in the model simulation as a datetime.datetime
            object
        
        Notes
        -----
        This method sets the values of two class attributes self.tamoc_water
        and self.tamoc_current.  These can be used by the `tamoc.blowout.   
        Blowout` class.  This method does not update the Blowout object, but
        relies on other methods of this class to do that when needed.  
        
        If GridTemperature, GridSalinity, or GridCurrent data are used, the
        .at() method requires a time.  This method compares the `current_time`
        to the available times in the Grid data.  If `current_time` is outside
        the bounds of the Grid data, then the closest time in the Grid data 
        is used to extract a profile.  
        
        """
        # Format ambient temperature and salinity for TAMOC
        if isinstance(water, gs.Water):
            
            # Get the constant, surface data from the Water object
            Ts = water.get('temperature')
            Ss = water.get('salinity')
            self.tamoc_water = {}
            self.tamoc_water['temperature'] = Ts
            self.tamoc_water['salinity'] = Ss
            
        elif isinstance(water, list):
            
            # Extract the gridded data
            grid_T = water[0]
            grid_S = water[1]
            
            # Get the water depth at the release location
            x0 = self.spill_position[0]
            y0 = self.spill_position[1]
            h = self.spill_position[2]
            
            # Create an array of depths to include in the profile
            n_points = h // 2
            z = np.linspace(0, h, num=n_points)
            points = np.zeros((len(z), 3))
            points[:,0] = x0
            points[:,1] = y0
            points[:,2] = z
            
            # Determine the time to extract
            if current_time > grid_T.data_stop:
                t0 = grid_T.data_stop
            elif current_time < grid_T.data_start:
                t0 = grid_T.data_start
            else:
                t0 = current_time
            
            # Extract the data
            T_data = grid_T.at(points, t0)
            T_data, T_units = ambient.convert_units(T_data, grid_T.units)
            S_data = grid_S.at(points, t0)
            S_data, S_units = ambient.convert_units(S_data, grid_S.units)
            P_data = ambient.compute_pressure(z, T_data, S_data, 0)
            
            # Build the water data array for the ztsp-data
            self.tamoc_water = np.zeros((len(z), 4))
            self.tamoc_water[:,0] = z
            self.tamoc_water[:,1] = T_data
            self.tamoc_water[:,2] = S_data
            self.tamoc_water[:,3] = P_data
            
        else:
            self.tamoc_water = water
        
        # Format the ambient current profile for TAMOC
        if isinstance(current, list):
            
            # Extract the gridded data
            grid_U = current[0]
            
            # Get the water depth at the release location
            x0 = self.spill_position[0]
            y0 = self.spill_position[1]
            h = self.spill_position[2]
            
            # Create an array of depths to include in the profile
            n_points = int(h // 2)
            z = np.linspace(0, h, num=n_points)
            points = np.zeros((len(z), 3))
            points[:,0] = x0
            points[:,1] = y0
            points[:,2] = z
            
            # Determine the time to extract
            if current_time > grid_U.data_stop:
                t0 = grid_U.data_stop
            elif current_time < grid_U.data_start:
                t0 = grid_U.data_start
            else:
                t0 = current_time
            
            # Extract the data
            print('Points are:\n', points)
            U_data = grid_U.at(points, t0)
            U_units = [grid_U.units, grid_U.units, grid_U.units]
            U_data, U_units = ambient.convert_units(U_data, U_units)
            
            # Build the water data array for the ztsp-data
            self.tamoc_current = np.zeros((len(z), 4))
            self.tamoc_current[:,0] = z
            self.tamoc_current[:,1] = U_data[:,0]
            self.tamoc_current[:,2] = U_data[:,1]
            self.tamoc_current[:,3] = U_data[:,2]
            
        else:
            self.tamoc_current = current
    
    def update_tamoc_parameters(self, current_time):
        """
        Update the environmental forcing for a TAMOC simulation
        
        Update the `ambient.Profile` object and then re-compute the initial
        conditions for a `tamoc` Blowout simulation
        
        Parameters
        ----------
        current_time : datetime.datetime
            Current real time in the model simulation as a datetime.datetime
            object
        
        """
        # Update the ambient water column data
        self.update_tamoc_profile(self.water, self.current, current_time)
        self.tamoc_sim.update_water_data(self.tamoc_water)
        self.tamoc_sim.update_current_data(self.tamoc_current)
        
        # Update anything else to change about the TAMOC simulation
        # <nothing to do currently>
    
    def run_tamoc(self, current_time):
        """
        Run a `tamoc` simulation for the present conditions
        
        Updates the `tamoc` Blowout object with the present water conditions
        and then runs a simulation.
        
        Parameters
        ----------
        current_time : : datetime.datetime
            Current real time in the model simulation as a datetime.datetime
            object
        
        """
        # Decide whether to update the `tamoc` Profile object
        elapsed_time = (current_time - self.release_time).total_seconds()
        if elapsed_time > 0:
            self.update_tamoc_parameters(current_time)
        
        # Run the `tamoc` Blowout simulation
        self.tamoc_sim.simulate()
        print('\n Used currents of:')
        print('    u = %3.3f' % 
            self.tamoc_sim.profile.get_values(0., ['ua']))
        print('    v = %3.3f' % 
            self.tamoc_sim.profile.get_values(0., ['va']))
        print('\n -- GNOME Simulation\n')
    
    def release_elements(self, sc, start_time, end_time, 
        environment=None):
        """
        Initialize and release Lagrangian Elements into the GNOME Model
        
        Initializes GNOME Lagrangian Elements into the Model simulation 
        using initial conditions from the most recent TAMOC simulation.
        
        Parameters
        ----------
        sc : list
            Spill container list containing each of the Lagrangian elements
            currently in the GNOME simulation.  This method appends new 
            elements to this list.
        start_time : datetime.datetime
            Current real time in the model simulation as a datetime.datetime
            object
        end_time : float
            End of the release time step as a datetime.datetime object
        environment : dict, default=None
            New parameter needed by `release_elements`, but not used here
        
        Returns
        -------
        to_rel : int
            Number of Lagrangian elements released during the present call
            to this method.
        
        Notes
        -----
        This method updates the spill container `sc` with new Lagrangian
        elements using the mutable property of Python lists.
        
        """
        # Compute the time step in seconds
        time_step = (end_time - start_time).total_seconds()
        
        # Only perform action if the spill is active
        if not self.on or time_step == 0.:
            return 0
        
        # Check whether TAMOC needs to be run
        elapsed_time = (start_time - self.release_time).total_seconds()
        if start_time < self.end_release_time and \
            elapsed_time % self.tamoc_time_delta.total_seconds() == 0:
            # Run tamoc
            self.run_tamoc(start_time)
        
        # Release the LEs needed for this time step.
        print('Releasing Elements for Simulation Time: ', start_time, '...')
        print('   Time step = ', time_step)
        
        # Get the dictionary keyword for the present spill
        idx = sc.spills.index(self)
        
        # Compute the expected number of elements after release
        expected_num_release = self.num_elements_after_time(start_time, 
                                                            time_step)
        actual_num_release = self._num_released
        
        # Determine how many elements will be released this time
        to_rel = expected_num_release - actual_num_release
        if to_rel <= 0:
            print('    All', len(sc), 'elements have been released.', 
                  ' Tracking...')
            return 0 #nothing to release, so end early
        else:
            print('   Plan to have released: ', expected_num_release, 'of', 
                  self.num_elements, 'elements')
            print('   --> Will release', to_rel, ' elements this time step')
        
        # Add blank elements to the spill container
        sc._append_data_arrays(to_rel)
        self._num_released += to_rel
        
        # Associate these elements with the present spill object
        sc['spill_num'][-to_rel:] = idx
        
        # Add the substance attributes to the present elements
        self.substance.initialize_LEs(to_rel, sc)
        
        # Update the LE properties with output from TAMOC
        self.initialize_LEs(to_rel, sc, time_step)
        
        # Return the number of elements released
        return to_rel
    
    def num_elements_after_time(self, current_time, time_step):
        """
        Compute the number of elements that should exist for this spill
        after current_time + time_step
        
        Parameters
        ----------
        current_time : datetime.datetime
            Current real time in the model simulation as a datetime.datetime
            object
        time_step : float
            Simulation time step (s) in the GNOME Model simulation
        
        Returns
        -------
        num_released : int
            The number of Lagrangian elements released this time step
        
        """
        if current_time > self.end_release_time:
            # If release has ended, all elements will exist
            num_released = self.num_elements
        
        else:
            # Compute total number released from release rate
            num_released = int(((current_time - 
                                 self.release_time).total_seconds() + 
                                 time_step) * self.num_elements_per_second)
        
        # Return the total
        return num_released
    
    def initialize_LEs(self, to_rel, sc, time_step):
        """
        Initialize the properties of each new LE to be released
        
        Initializes all of the Lagrangian element properties for each new
        Lagrangian element released during a model time step using initial
        conditions taken from the results of the present `tamoc` Blowout
        simulation.
        
        Parameters
        ----------
        to_rel : int
            Number of Lagrangian elements to release this time step
        sc : list
            Spill container list containing each of the Lagrangian elements
            currently in the GNOME simulation.  This method appends new 
            elements to this list.
        time_step : float
            Simulation time step (s) in the GNOME Model simulation
        
        Notes
        -----
        This function uses the fact that Python lists are pointers to 
        memory locations.  When values in the spill container lists are 
        updated in this functions, those values are updated in the spill
        container throughout the model using the mutability of Python lists.
        
        """
        # Get the indicies in the spill container that need to be updated
        sl = slice(-to_rel, None, 1)
        
        # Get the particle data from the stored TAMOC simulation
        tp, nparticles = self.get_tamoc_particles(to_rel, time_step)
        
        # Broadcast the `tamoc` Blowout particles to the present number of 
        # Lagrianian elements to release...start by making a copy of tp
        rp = {}
        rp['positions'] = tp['positions'][:]
        rp['init_mass'] = tp['init_mass'][:]
        rp['mass'] = tp['mass'][:]
        rp['mass_components'] = tp['mass_components'][:]
        rp['density'] = tp['density'][:]
        rp['viscosity'] = tp['viscosity'][:]
        rp['droplet_diameter'] = tp['droplet_diameter'][:]
        rp['rise_vel'] = tp['rise_vel'][:]
        
        # Next, store the correct total mass to release
        m0 = np.sum(np.array(tp['init_mass']))
        
        # Return the correct set of particles
        cycles = int(to_rel / nparticles)
        extras = to_rel % (nparticles * cycles)
        
        # Decide how to distribute the `tamoc` particle (tp) to the spill 
        # container Lagrangian elements
        if cycles == 0:
            # Return the largest particles only...overwrite the copy in rp
            rp['positions'] = tp['positions'][sl]
            rp['init_mass'] = tp['init_mass'][sl]
            rp['mass'] = tp['mass'][sl]
            rp['mass_components'] = tp['mass_components'][sl]
            rp['density'] = tp['density'][sl]
            rp['viscosity'] = tp['viscosity'][sl]
            rp['droplet_diameter'] = tp['droplet_diameter'][sl]
            rp['rise_vel'] = tp['rise_vel'][sl]
        elif extras == 0:
            # Return the right number of cycles through all particles
            if cycles > 1:
                for i in range(cycles - 1):
                    rp['positions'] += tp['positions'][:]
                    rp['init_mass'] += tp['init_mass'][:]
                    rp['mass'] += tp['mass'][:]
                    rp['mass_components'] += tp['mass_components'][:]
                    rp['density'] += tp['density'][:]
                    rp['viscosity'] += tp['viscosity'][:]
                    rp['droplet_diameter'] += tp['droplet_diameter'][:]
                    rp['rise_vel'] += tp['rise_vel'][:]
        else:
            # First, walk through the whole number of cycles
            if cycles > 1:
                for i in range(cycles - 1):
                    rp['positions'] += tp['positions'][:]
                    rp['init_mass'] += tp['init_mass'][:]
                    rp['mass'] += tp['mass'][:]
                    rp['mass_components'] += tp['mass_components'][:]
                    rp['density'] += tp['density'][:]
                    rp['viscosity'] += tp['viscosity'][:]
                    rp['droplet_diameter'] += tp['droplet_diameter'][:]
                    rp['rise_vel'] += tp['rise_vel'][:]
            # Second, add the remaining particles from the largest ones
            sub_sl = slice(-extras, None, 1)
            rp['positions'] += tp['positions'][sub_sl]
            rp['init_mass'] += tp['init_mass'][sub_sl]
            rp['mass'] += tp['mass'][sub_sl]
            rp['mass_components'] += tp['mass_components'][sub_sl]
            rp['density'] += tp['density'][sub_sl]
            rp['viscosity'] += tp['viscosity'][sub_sl]
            rp['droplet_diameter'] += tp['droplet_diameter'][sub_sl]
            rp['rise_vel'] += tp['rise_vel'][sub_sl]
        
        # Scale the masses to conserve mass...this is needed if we release
        # a different number of particles in GNOME than were simulated in 
        # `tamoc`...this is normal and not correcting for an error.
        m1 = np.sum(np.array(rp['init_mass']))
        m_frac = m0 / m1
        rp['init_mass'] = list(np.array(rp['init_mass']) * m_frac)
        rp['mass'] = list(np.array(rp['mass']) * m_frac)
        for i in range(to_rel):
            rp['mass_components'][i] = \
                list(np.array(rp['mass_components'][i]) * m_frac)
        
        # Create the LEs in the spill container
        sc['positions'][sl] = rp['positions']
        sc['init_mass'][sl] = rp['init_mass']
        sc['mass'][sl] = rp['mass']
        sc['mass_components'][sl] = rp['mass_components']
        sc['density'][sl] = rp['density']
        sc['viscosity'][sl] = rp['viscosity']
        sc['droplet_diameter'][sl] = rp['droplet_diameter']
        sc['rise_vel'][sl] = rp['rise_vel']
    
    def get_tamoc_particles(self, to_rel, time_step):
        """
        Get the particle information for each droplet in a TAMOC simulation
        
        Gets the initial condition data for each liquid droplet in a `tamoc`
        Blowout simulation.  
        
        Parameters
        ----------
        to_rel : int
            Number of Lagrangian elements to release this time step
        time_step : float
            Simulation time step (s) in the GNOME Model simulation
        
        Returns
        -------
        tp : list of spill container variables
            List of Lagrangian Element Properties for each `tamoc` particle
            (e.g., the liquid oil droplets) organized in the same way as the
            GNOME spill container.  This list only contains one entry for
            each `tamoc` particle.
        
        nparticles : int
            Number of `tamoc` particles passed to the spill container in 
            GNOME
            
        """
        # Extract TAMOC solution at the end of the simulation
        t = self.tamoc_sim.bpm.t[-1]
        q = self.tamoc_sim.bpm.q[-1,:]
        particles = self.tamoc_sim.bpm.particles
        
        # Initialize a dictionary to hold the particle information in the 
        # format of a GNOME spill container
        tp = {
            'positions' : [],
            'init_mass' : [],
            'mass' : [],
            'mass_components' : [],
            'density' : [],
            'viscosity' : [],
            'droplet_diameter' : [],
            'rise_vel' : []
        }
        
        # Create a list of Oil Library chemical names
        adios_chems = ['Saturates', 'Aromatics', 'Resins', 
                       'Asphaltenes']
        
        # Get the information for each liquid particle (ignore gas bubbles)
        nparticles = 0
        for particle in particles:
            
            if particle.particle.fp_type == 1:
                
                nparticles += 1
                if not particle.farfield:
                    # This particle stayed in the plume
                    tp['positions'].append(positions(particle.x, particle.y,
                                                     particle.z, 
                                                     self.project, 
                                                     self.ref_pt))
                    
                    # Get the masses
                    mi_droplet = particle.m
                    ndot = particle.nb0
                    comp = particle.particle.composition
                    mass_components, init_mass, mass = masses(mi_droplet,
                        ndot, comp, time_step, adios_chems)
                    tp['mass_components'].append(mass_components)
                    tp['init_mass'].append(init_mass)
                    tp['mass'].append(mass)
                    
                    # Get the particle age
                    t0 = particle.t
                
                else:
                    # This particle was tracked in the intermediate field
                    x, y, z = particle.sbm.y[-1,0:3]
                    tp['positions'].append(positions(x, y, z, self.project,
                        self.ref_pt))
                    
                    # Get the masses
                    mi_droplet = particle.sbm.y[-1,3:-1]
                    ndot = particle.nb0
                    comp = particle.particle.composition
                    mass_components, init_mass, mass = masses(mi_droplet,
                        ndot, comp, time_step, adios_chems)
                    tp['mass_components'].append(mass_components)
                    tp['init_mass'].append(init_mass)
                    tp['mass'].append(mass)
                    
                    # Get the particle age
                    t0 = particle.t + particle.sbm.t[-1]
                
                # Get the remaining particle properties
                z0 = tp['positions'][-1][2]
                if z0 < self.tamoc_sim.bpm.profile.z_min:
                    z0 = self.tamoc_sim.bpm.profile.z_min
                elif z0 > self.tamoc_sim.bpm.profile.z_max:
                    z0 = self.tamoc_sim.bpm.profile.z_max
                Ta, Sa, Pa = self.tamoc_sim.bpm.profile.get_values(z0, 
                    ['temperature', 'salinity', 'pressure'])
                us, rho_p, A, Cs, beta, beta_T, T = particle.properties(
                    mi_droplet, Ta, Pa, Sa, Ta, t0)
                
                # Add these properties to the `tamoc` particle dictionary
                tp['density'].append(rho_p)
                tp['viscosity'].append(particle.particle.viscosity(
                    mi_droplet, T, Pa))
                tp['droplet_diameter'].append(particle.diameter(mi_droplet, 
                    T, Pa, Sa, Ta))
                tp['rise_vel'].append(us)
        
        # Return the particle list
        return tp, nparticles


# ----------------------------------------------------------------------------
# Helper functions to get and format particle properties
# ----------------------------------------------------------------------------

def positions(x, y, z, projection, ref_pt):
    """
    Compute positions in the GNOME coordinate system
    
    Convert the `tamoc` (x, y, z) coordinate system to the GNOME (longitude,
    latitude, depth) coordinate system using a GNOME flat-Earth projection
    tool.
    
    Parameters
    ----------
    x : float
        Position along the x-axis (East) in m
    y : float
        Position along the y-axis (North) in m
    z : float
        Depth (m)
    projection : FlatEarthProjection object
        Tool from gnome.utilities.projections that implements a flat-Earth
        (e.g., local Cartesian) coordinate system.
    ref_pt : ndarray
        Reference point in longitude (deg), latitude (deg), and depth (m) for 
        the flat-Earth projection.
    
    Returns
    -------
    list of positions
        Returns a list of positions in longitude (deg), latitude (deg) and 
        depth (m)
    
    """
    # Use the gnome projections tool
    x0, y0, z0 = projection.meters_to_lonlat(
        np.array([x, y, z]), ref_pt)[0] + ref_pt
    
    # Make sure particles do not pass through the free surface
    if z < 0:
        z0 = 0.
    else:
        z0 = z
    
    return([x0, y0, z0])


def masses(mi_droplet, ndot, comp, time_step, adios_chems):
    """
    Return the masses for the components in a GNOME Lagrangian Element
    
    Computes the correct total mass for each pseudocomponent for a GNOME
    Lagrangian element and returns the masses for the same pseudocomponents
    and in the same order as imported from the Oil Library.
    
    Parameters
    ----------
    mi_droplet : ndarray
        Array of mass fluxes (kg/s) for each pseudocomponent in each droplet
        in the `tamoc` blowout simulation.
    ndot : float
        Number flux (#/s) of droplets for this Lagrangian element in `tamoc`
    comp : list of str
        List of pseudocomponent names for each pseudocomponent of the `tamoc`
        particle
    time_step : float
        The GNOME Model time step (s)
    adios_chems : list of str
        List of string names for pseudocomponents defined in the Oil Library
    
    Returns
    -------
    mass_components : list of float
        List of masses (kg) for each pseudocomponent in an Oil Library 
        substance
    init_mass : float
        The total mass contained in the present GNOME Lagrangian element 
        (kg); this mass should not change with weathering.
    mass : float
        The total mass contained in the present GNOME Lagrangian element (kg);
        this mass will change with weathering.
    
    Notes
    -----
    The Oil Library names pseudocomponent with the same name multiple times:
    e.g., the name Saturates may occur several times in a composition file.
    When this data is imported to `tamoc`, each repetition is given a unique
    name by appending an integer to the name (e.g., Satures1).  Because 
    these are stored in `tamoc` in the same order as they occurred in the 
    Oil Library, here we unpack the `tamoc` pseudocomponents and add them
    to the spill container as long as the `tamoc` chemical name includes one
    of the root words from the Oil Library database (e.g., Satures) a not
    another compound (e.g., oxygen).
    
    """
    m_comp = mi_droplet * ndot * time_step
    m0 = np.sum(m_comp)
    
    # Find the appropriate pseudocomponents
    mi = []
    for i in range(len(comp)):
        keep = False
        for psuedo_component in adios_chems:
            if psuedo_component in comp[i]:
                keep = True
        if keep:
            mi.append(m_comp[i])
            
    # Scale the passed components to conserve total mass
    mi = np.array(mi)
    mi = mi * m0 / np.sum(mi)
    
    # Format the data with the necessary data types
    mass_components = list(mi)
    init_mass = m0
    mass = m0
    
    return (mass_components, init_mass, mass)

