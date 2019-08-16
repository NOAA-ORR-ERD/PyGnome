"""
tamoc_utilities.py
------------------

This module contains functions to help create an accidental subsea blowout
simulation using the TAMOC bent plume model.  These utilities:

    Create ambient.Profile objects from text CTD data
    Read in bubble and droplet size data
    Create the initial conditions needed by the bent plume model
    Analyze the results of a bent plume model simulation
    Plot and print diagnostic information

S. Socolofsky 
socolofs@tamu.edu
February 26, 2019

"""
from tamoc import seawater, ambient, dbm

import adios_utilities as au

import numpy as np
import matplotlib.pyplot as plt

def get_ctd_from_txt(fname, summary, source, sea_name, p_lat, p_lon, 
    p_time, ca=[]):
    """
    Create an ambient.Profile object from a text file of ocean property data
    
    Read the CTD and current data in the given filename (fname) and use that
    data to create an ambient.Profile object for use in TAMOC.  This function
    is built to work with an ascii file created from the model output by
    Youyu Lu. The output are stored in columns that report depth (m),
    temperature (deg C), salinity (psu), u-component of velocity (m/s) and
    v-component of velocity (m/s).
    
    Parameters
    ----------
    fname : str
        String containing the relative path to the water column data file.
    summary : str
        String describing the simulation for which this data will be used.
    source : str
        String documenting the source of the ambient ocean data provided.
    sea_name : str
        NC-compliant name for the ocean water body as a string.
    p_lat : float
        Latitude (deg)
    p_lon : float
        Longitude, negative is west of 0 (deg)
    p_time : netCDF4 time format
        Date and time of the CTD data using netCDF4.date2num().
    ca : list, default=[]
        List of dissolved atmospheric gases to include in the ambient ocean
        data as a derived concentration; choices are 'nitrogen', 'oxygen', 
        'argon', and 'carbon_dioxide'.  
    
    Returns
    -------
    profile : ambient.Profile
        Returns an ambient.Profile object for manipulating ambient water 
        column data in TAMOC.
    
    """
    # Read in the data
    data = np.loadtxt(fname, comments='#')
    
    # Describe what should be stored in this dataset
    units = ['m', 'deg C', 'psu', 'm/s', 'm/s']
    labels = ['z', 'temperature', 'salinity', 'ua', 'va']
    comments = ['modeled', 'modeled', 'modeled', 'modeled', 'modeled']
    
    # Extract a file name for the netCDF4 dataset that will hold this data
    # based on the name of the text file.
    nc_name = '.'.join(fname.split('.')[:-1])  # remove text file .-extension
    nc_name = nc_name + '.nc'
    
    # Create the ambient.Profile object
    profile = create_ambient_profile(data, labels, units, comments, nc_name, 
        summary, source, sea_name, p_lat, p_lon, p_time, ca)
    
    return profile
    
def create_ambient_profile(data, labels, units, comments, nc_name, summary,
    source, sea_name, p_lat, p_lon, p_time, ca=[]):
    """
    Create an ambient Profile object from given data
    
    Create an ambient.Profile object using the given CTD and current data.
    This function performs some standard operations to this data (unit
    conversion, computation of pressure, insertion of concentrations for
    dissolved gases, etc.) and returns the working ambient.Profile object.
    The idea behind this function is to separate data manipulation and
    creation of the ambient.Profile object from fetching of the data itself.
    
    Parameters
    ----------
    data : np.array
        Array of the ambient ocean data to write to the CTD file.  The 
        contents and dimensions of this data are specified in the labels 
        and units lists, below.
    labels : list
        List of string names of each variable in the data array.
    units : list
        List of units as strings for each variable in the data array.
    comments : list
        List of comments as strings that explain the types of data in the 
        data array.  Typical comments include 'measured', 'modeled', or
        'computed'.
    nc_name : str
        String containing the file path and file name to use when creating
        the netCDF4 dataset that will contain this data.
    summary : str
        String describing the simulation for which this data will be used.
    source : str
        String documenting the source of the ambient ocean data provided.
    sea_name : str
        NC-compliant name for the ocean water body as a string.
    p_lat : float
        Latitude (deg)
    p_lon : float
        Longitude, negative is west of 0 (deg)
    p_time : netCDF4 time format
        Date and time of the CTD data using netCDF4.date2num().
    ca : list, default=[]
        List of gases for which to compute a standard dissolved gas profile;
        choices are 'nitrogen', 'oxygen', 'argon', and 'carbon_dioxide'.
    
    Returns
    -------
    profile : ambient.Profile
        Returns an ambient.Profile object for manipulating ambient water 
        column data in TAMOC.
    
    """
    # Convert the data to standard units
    data, units = ambient.convert_units(data, units)
    
    # Create an empty netCDF4-classic datast to store this CTD data
    nc = ambient.create_nc_db(nc_name, summary, source, sea_name, p_lat,
                              p_lon, p_time)
    
    # Put the CTD and current profile data into the ambient netCDF file
    nc = ambient.fill_nc_db(nc, data, labels, units, comments, 0)
    
    # Compute and insert the pressure data
    z = nc.variables['z'][:]
    T = nc.variables['temperature'][:]
    S = nc.variables['salinity'][:]
    P = ambient.compute_pressure(z, T, S, 0)
    P_data = np.vstack((z, P)).transpose()
    nc = ambient.fill_nc_db(nc, P_data, ['z', 'pressure'], ['m', 'Pa'], 
                            ['measured', 'computed'], 0)
    
    # Use this netCDF file to create an ambient object
    profile = ambient.Profile(nc, ztsp=['z', 'temperature', 'salinity', 
                 'pressure', 'ua', 'va'])
    
    # Compute dissolved gas profiles to add to this dataset
    if len(ca) > 0:
        
        # Create a gas mixture object for air
        gases = ['nitrogen', 'oxygen', 'argon', 'carbon_dioxide']
        air = dbm.FluidMixture(gases)
        yk = np.array([0.78084, 0.20946, 0.009340, 0.00036])
        m = air.masses(yk)
        
        # Set atmospheric conditions
        Pa = 101325.
        
        # Compute the desired concentrations
        for i in range(len(ca)):
            
            # Initialize a dataset of concentration data
            conc = np.zeros(len(profile.z))
            
            # Compute the concentrations at each depth
            for j in range(len(conc)):
                
                # Get the local water column properties
                T, S, P = profile.get_values(profile.z[j], ['temperature',
                    'salinity', 'pressure'])
                
                # Compute the gas solubility at this temperature and salinity 
                # at the sea surface
                Cs = air.solubility(m, T, Pa, S)[0,:]
                
                # Adjust the solubility to the present depth
                Cs = Cs * seawater.density(T, S, P) / \
                    seawater.density(T, S, 101325.)
                
                # Extract the right chemical
                conc[j] = Cs[gases.index(ca[i])]
            
            # Add this computed dissolved gas to the Profile dataset
            data = np.vstack((profile.z, conc)).transpose()
            symbols = ['z', ca[i]]
            units = ['m', 'kg/m^3']
            comments = ['measured', 'computed from CTD data']
            profile.append(data, symbols, units, comments, 0)
    
    # Close the netCDF dataset
    profile.close_nc()
    
    # Return the profile object
    return profile


def get_adios_oil(adios_id, q_oil, gor, ca):
    """
    Create a dbm.FluidMixture object for this oil and given flow rate
    
    Create a dbm.FluidMixture object that NOAA Oil Library oil given by 
    the unique adios_id number, mixed with the requested amount of natural
    gas given by the gas to oil ratio (gor), and return a matrix of oil
    component mass fluxes to achieve the given oil flow rate (q_oil).
    
    Parameters
    ----------
    adios_id : str
        The unique ADIOS ID number of an oil in the NOAA Oil Library as a 
        string.
    q_oil : float
        Flow rate of oil (bbl/d at standard conditions). 
    gor : float
        Gas to oil ratio desired for a given live-oil release. 
    ca : list, default=[]
        List of dissolved atmospheric gases to track as part of the oil;
        choices are 'nitrogen', 'oxygen', 'argon', and 'carbon_dioxide'.
    
    Returns
    -------
    oil : dbm.FluidMixture
        A discrete bubble model FluidMixture object that contains the 
        property data for the desired live oil.
    mass_flux : np.array
        An array of gas and liquid mass fluxes for each chemical component
        in the mixture (kg/s) required to achieve the desired flow rate of 
        dead oil at the surface, q_oil.
    
    """
    # Import the dead oil properties from the NOAA Oil Library and convert
    # them to the required TAMOC properties
    composition, mass_frac, user_data, delta, units = au.load_adios_oil(
        adios_id)
    
    # Add the atmospherica gases to the oil library if desired
    if len(ca) > 0:
        # Update the composition
        composition = composition + ca
        # Update the mass fractions assuming zero atmospheric gases in 
        # petroleum
        new_mf = np.zeros(len(composition))
        new_mf[0:len(mass_frac)] = mass_frac
        mass_frac = new_mf
        # Update the binary interaction parameters
        oil = dbm.FluidMixture(composition, user_data=user_data)
        delta = au.pedersen(oil.M, composition)
    
    # Create a live oil mixture for this oil that has the given GOR
    composition, mass_frac, delta = au.mix_gas_for_gor(composition, 
        mass_frac, user_data, delta, gor)
    
    # Get the mass flux for the desired oil flow rate
    mass_flux = au.set_mass_fluxes(composition, mass_frac, user_data, delta, 
        q_oil)
    
    # Create the dbm.FluidMixture object
    oil = dbm.FluidMixture(composition, delta=delta, user_data=user_data)
    
    # Return the results
    return oil, mass_flux


def plot_profile(profile, fignum=1):
    """
    Plot profiles from an ambient.Profile object
    
    Plot ambient water column data for an ambient.Profile object.  This 
    function displays all data that is found in the Profile object.  It uses
    the standard variable names in TAMOC to decide whether data is present.
    
    Parameters
    ----------
    profile : ambient.Profile
        Profile object that contains the water column data for a TAMOC 
        simulation
    fignum : int, default=1
        Figure number to use to plot the data
    
    """
    # Create the depth axis to cover the available data
    z = np.linspace(profile.z_min, profile.z_max, 500)
    
    # Extract the standard, required water column data
    ztsp = profile.ztsp
    if 'temperature' not in ztsp:
        ztsp.append('temperature')
    if 'salinity' not in ztsp:
        ztsp.append('salinity')
    if 'pressure' not in ztsp:
        ztsp.append('pressure')
    if 'ua' not in ztsp:
        ztsp.append('ua')
    if 'va' not in ztsp:
        ztsp.append('va')
    ztsp_data = np.zeros((len(z), len(ztsp)))
    for i in range(len(z)):
        # get_values will return zero for anything not in the profile
        ztsp_data[i,:] = profile.get_values(z[i], ztsp)
    
    # Extract any dissolved chemical data available
    chem_names = profile.chem_names
    if len(chem_names) > 0:
        chem_data = np.zeros((len(z), len(chem_names)))
        for i in range(len(z)):
            chem_data[i,:] = profile.get_values(z[i], chem_names)
    
    # Compute the density profile
    rho_data = np.zeros(len(z))
    for i in range(len(z)):
        T, S, P = profile.get_values(z[i], ['temperature', 'salinity', 
            'pressure'])
        rho_data[i] = seawater.density(T, S, P)
    
    # Open a figure for plotting
    fig = plt.figure(fignum)
    fig.clf()
    
    # Plot the temperature, salinity, and density
    ax1 = plt.subplot(231)
    ax1.plot(ztsp_data[:,ztsp.index('temperature')] - 273.15, z)
    ax1.set_xlabel('Temperature, (deg C)')
    ax1.set_ylabel('Depth, (z)')
    ax1.invert_yaxis()
    
    ax2 = plt.subplot(232)
    ax2.plot(ztsp_data[:,ztsp.index('salinity')], z)
    ax2.set_xlabel('Salinity, (psu)')
    ax2.invert_yaxis()
    
    ax3 = plt.subplot(233)
    ax3.plot(rho_data, z)
    ax3.set_xlabel('Density (kg/m^3)')
    ax3.invert_yaxis()
    
    ax4 = plt.subplot(234)
    ax4.plot(ztsp_data[:,ztsp.index('ua')], z)
    ax4.set_xlabel('Easterly Currents (ua in m/s)')
    ax4.set_ylabel('Depth, (z)')
    ax4.invert_yaxis()
    
    ax5 = plt.subplot(235)
    ax5.plot(ztsp_data[:,ztsp.index('va')], z)
    ax5.set_xlabel('Northerly Currents (va in m/s)')
    ax5.invert_yaxis()
    
    # Plot the available chemistry data
    ax6 = plt.subplot(236)
    if len(chem_names) > 0:    
        for i in range(len(chem_names)):
            ax6.plot(chem_data[:,i], z, label=chem_names[i])
        ax6.legend()
    ax6.set_xlabel('Dissolved Gases (kg/m^3)')
    ax6.invert_yaxis()
    
    fig.show()
    fig.canvas.draw_idle()


def print_fluid_ic(profile, oil, mass_flux, d0, z0=None):
    """
    Print the fluid properties at the release
    
    Print the fluid chemical, thermodynamic, and physical properties at the
    blowout release point.
    
    Parameters
    ----------
    profile : ambient.Profile
        Profile object that contains the water column data for a TAMOC 
        simulation
    oil : dbm.FluidMixture
        Discrete bubble model FluidMixture object that contains the present
        petroleum mixture
    mass_flux : np.array
        An array of gas and liquid mass fluxes for each chemical component
        in the mixture (kg/s) required to achieve the desired flow rate of 
        dead oil at the surface, q_oil.
    d0 : float
        Diameter of the equivalent circular orifice of the release (m).
    z0 : float, default=None
        Release depth (m); if None, then use the bottom of the CTD profile
        as the release depth
    
    """
    # Get the ambient water column conditions at the release
    if z0 == None:
        z0 = profile.z_max
    T, S, P = profile.get_values(z0, ['temperature', 'salinity', 'pressure'])
    
    # Print the fluid properties
    au.print_petroleum_props(oil.composition, mass_flux, oil.user_data,     
        oil.delta, T, S, P)
    
    # Also report the diameter of the release
    print '\n    orifice diameter (m)    : ', d0, '\n'


