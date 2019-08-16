"""
adios_utilities.py
------------------

This module contains functions to load an oil from the NOAA Oil Library
(ADIOS) and convert the properties listed there into the properties and
format needed to create a discrete bubble model FluidMixture or FluidParticle
object. These utilities are translations of code written by Jonas Gros and
use the methods published in:

    Gros, J., A. L. Dissanayake, M. M. Daniels, C. H. Barker, W. Lehr, and S.
    A. Socolofsky (2018), Oil spill modeling in deep waters: Estimation of
    pseudo-component properties for cubic equations of state from
    distillation data, Mar Pollut Bull, 137, 627-637,
    doi:https://doi.org/10.1016/j.marpolbul.2018.10.047.

S. Socolofsky
socolofs@tamu.edu
February 26, 2019

"""
from tamoc import seawater
from tamoc import chemical_properties as chem
from tamoc import dbm

import numpy as np
from scipy.optimize import fsolve, fmin


def load_csv_oil(chem_file, composition_file):
    """
    Load an oil from an external chemical properties file
    
    Load the chemical properties and pseudo-component compositions from 
    a .csv file created by Jonas Gros from the ADIOS Oil Library.  
    
    Parameters
    ----------
    chem_file : str
        Name of a .csv file that contains the chemical properties in the 
        format expected by the tamoc.dbm module.  This file was created by
        Jonas based on information in the Adios database
    composition_file : str
        Name of another .csv file that contains the composition of the dead
        oil as it was reported in the Adios database.  This file is also 
        provided by Jonas
    
    Returns
    -------
    composition : list
        List of strings containing the names of the oil components in the 
        dead oil from the Adios database.
    mass_frac : np.array
        An array of mass fractions for all compounds in the dead oil from 
        the Adios database (kg).
    user_data : dict
        A dictionary of chemical property data in the format expected by 
        the tamoc.dbm module FluidMixture or FluidParticle objects.
    delta : np.array (len M, len M)
        Array of binary interaction coefficients
    
    """
    # Use the load_data function in chem to read in a user-defined chemical 
    # properties database
    user_data, units = chem.load_data(chem_file)
    
    # Load in composition and mass fractions for each component of this oil
    composition = []
    mass_frac = []
    with open(composition_file) as datfile:
        
        # Read each line separately
        for line in datfile:
            
            # Only consider lines that are not header lines
            if line.find('%') < 0:
                
                # Extract the data and append to the lists
                values = line.strip().split(',')
                composition.append(values[0])
                mass_frac.append(float(values[1]))
    
    # Convert mass fractions to numpy array
    mass_frac = np.array(mass_frac)
    
    # Create a dbm.FluidMixture object with this composition, but not yet
    # knowing the binary interaction coefficients
    oil = dbm.FluidMixture(composition, user_data=user_data)
    
    # Use the Pedersen method to estimate the binary interaction parameters
    delta = pedersen(oil.M, composition)

    return (composition, mass_frac, user_data, delta, units)


def load_adios_oil(adios_id):
    """
    Load an oil from the NOAA Oil Library (formerly, ADIOS)
    
    Create the chemical property data for a TAMOC discrete bubble model
    FluidMixture of FluidParticle object from data for an oil in the ADIOS
    library. This function reads in the oil properties from the NOAA Oil
    Library and computes the inputs needed by TAMOC using methods published
    in Gros et al. (2018) MPB, 137:627-637.
    
    Parameters
    ----------
    adios_id : str
        The unique ADIOS ID number of an oil in the NOAA Oil Library as a 
        string.
    
    Returns
    -------
    composition : list
        List of strings containing the names of the oil components in the 
        dead oil from the Adios database.
    mass_frac : np.array
        An array of mass fractions for all compounds in the dead oil from 
        the Adios database (kg).
    user_data : dict
        A dictionary of chemical property data in the format expected by 
        the tamoc.dbm module FluidMixture or FluidParticle objects.
    delta : np.array (len M, len M)
        Array of binary interaction coefficients
        
    """
    # Import the tools needed from the NOAA Oil Library
    from oil_library import get_oil_props
    
    # Read in the desired oil from the ADIOS Oil Library
    gnome_oil = get_oil_props(adios_id)
    print '\n----------'
    print 'Loading NOAA Oil Library Oil:  ' + adios_id
    print 'Record name:  ' + gnome_oil.record.name
    print '----------\n'
    
    # Extract properties of this oil from the gnome_oil object
    molecular_weight = gnome_oil.molecular_weight         # g/mol
    mass_frac = gnome_oil.mass_fraction                   # --
    boiling_point = gnome_oil.boiling_point               # K
    vapor_pressure_5C = gnome_oil.vapor_pressure(278.15)  # Pa
    vapor_pressure_25C = gnome_oil.vapor_pressure(298.15) # Pa
    
    # Extract the densities of each pseudocomponent
    density = gnome_oil.component_density
    
    # Convert these densities so that the new densities will give the same 
    # oil density using the formula density = 1. / sum(mass_frac / density)
    density2 = density * (np.sum(density * mass_frac)) / (1. / 
        np.sum(mass_frac / density))
    
    # Read in the names of each of the oil pseudocomponents
    composition = list(gnome_oil.component_types)
    
    # TAMOC requires unique names for each pseudocomponent; we add a counter
    # to each SARA analysis type (e.g., Saturates1, Saturates2, etc.)
    sequence_names(composition, 'Saturates')
    sequence_names(composition, 'Aromatics')
    
    # Report any error messages or warnings
    if np.any(boiling_point < 231):
        # The below methods do not work for compounds more volatile than 
        # propane:
        print '\nWARNING:  This oil entry has compounds more volatile than'
        print '          propane.  Current property estimation methods are'
        print '          not designed for this situation.  Errors may occur'
        print '          using this oil.\n'
    
    # Estimate oil properties using methods in Gros et al. (2018)
    solubility = get_solubility(molecular_weight, density)
    k_h_0 = get_henry_constant(solubility, vapor_pressure_25C, 
        molecular_weight)
    (Tc, Pc, Vc, M, omega, delta) = get_preos_params(boiling_point, 
        molecular_weight, density)
    
    nu_bar = (-2.203e-5 * Pc + 518.6 * M + 143.4) * 1.e-6
    neg_delta_H_sol_R = 2.637 * Tc + 22.48e6 * nu_bar + 314.6
    K_salt = (-1.345 * M + 2799.4 * nu_bar +  0.083556) / 1000.
              
    # Estimate Vb based on the Tyn and Calus formula (see dbm.py)
    Vb = compute_Vb(Vc)
    
    # Turn off solubility of non-aromatic hydrocarbons
    for i in range(len(composition)):
        if composition[i].find('Aromatics') < 0:
            # Use -9999. flag expected in dbm module of TAMOC
            neg_delta_H_sol_R[i] = -9999.
            k_h_0[i] = -9999.
            nu_bar[i] = -9999.
            K_salt[i] = -9999.
    
    # Format these data as they are normally used in the dbm module of TAMOC
    user_data, units = format_dbm_data(composition, M, Pc, Tc, omega, k_h_0, 
        neg_delta_H_sol_R, nu_bar, K_salt, Vc, boiling_point, Vb, 
        B=None, dE=None)
        
    # Extract the measurements of the whole oil density
    rho_data = gnome_oil.culled_densities()
    
    # Perform tuning of Vc to get better densities
    user_data = Vc_tuning(mass_frac, composition, rho_data, density2,
        delta, user_data)
    
    # Update the value of Vb in the user_data database using the final 
    # value of Vc after tuning.
    for i in range(len(composition)):
        Vc = user_data[composition[i]]['Vc']
        Vb = compute_Vb(Vc)
        user_data[composition[i]]['Vb'] = Vb
    
    # Return the results
    return (composition, mass_frac, user_data, delta, units)


def compute_Vb(Vc):
    """
    Compute the molar volume at the boiling point
    
    Compute the specific molar volume at the boiling point from the 
    specific molar volume at the critical point using the Tyn and Calus 
    formula (see dbm.py for details).
    
    Parameters
    ----------
    Vc : float or np.array
        Specific molar volume at the critical point (m^3/mol)
    
    Returns
    -------
    Vb : float or np.array
        Specific molar volume at the boiling point (m^3/mol)
    
    """
    return (0.285 * (Vc * 1.e6)**1.048) * 1.e-6


def get_solubility(molecular_weight, density):
    """
    Estimate the solubility of each oil pseudo-component
    
    Estimate the solubility (mol/L) of each oil pseudo-component using the
    method from Huibers and Lehr given in the huibers_lehr.py module of
    py_gnome in the director gnome/utilities/weathering/. This method is from
    Huibers & Katrisky in a 2012 EPA report and was further modified by Lehr
    to better match measured values. The equation used here is adapted to
    return results in mol/L.
    
    Parameters
    ----------
    molecular_weight : np.array
        Molecular weights of each pseudo-component as recorded in the NOAA
        Oil Library (g/mol)
    density : np.array
        Density of each pseudo-component as recorded in the NOAA Oil Library
        (kg/m^3)
    
    Returns
    -------
    solubility : np.array
        Array of solubilities (mol/L) for each pseudo-component of the oil.
    
    """
    return 46.4 * 10. ** (-36.7 * molecular_weight / density)


def get_henry_constant(solubility, vapor_pressure_25C, molecular_weight):
    """
    Estimate the Henry's Law constant for each pseudo-component
    
    Estimate the Henry's Law constant at standard conditions for each
    pseudo-component in an oil. This method uses the solubility together with
    the definition of the Henry's Law constant to estimate k_h_0 at standard
    conditions.
    
    Parameters
    ----------
    solubility : np.array
        Array of solubilities (mol/L) for each pseudo-component of the oil.
    vapor_pressure_25C : np.array
        Array of vapor pressures (Pa) for each pseudo-component of the oil
        at 25 deg C and atmospheric pressure.
    molecular_weight : np.array
        Array of molecular weights (g/mol) for each pseudo-component of the 
        oil as recorded in the NOAA Oil Library.
    
    Returns
    -------
    k_h_0 : np.array
        Henry's Law constant at standard conditions (kg/(m^3 Pa)).
    
    """
    # Use the definition of Henry's law constant to estimate its value in
    # mol / (L Pa)
    k_h_0 = solubility / vapor_pressure_25C
    
    # Return the result in units required by TAMOC (kg / (m^3 Pa))
    return k_h_0 * (molecular_weight / 1000.) * 1000.


def get_preos_params(Tb, M, rho):
    """
    Get the parameters required by the Ping Robinson Equation of State
    
    We estimate the parameters required by the Ping Robinson equation of
    state using the methods Twu (9184) and by Kesler and Lee reported in Chen
    et al. (1993). These are the critical point temperature, pressure, and
    specific volume and the acentric factor. We also estimate the binary
    interaction parameters using a method in Pedersen et al. (2014).
    
    In theory, the Twu and Kesler and Lee correlations should only be used
    for pseudo-components above n-C6, and literature property values should
    be used for pseudo-component below n-C6. When these lighter compounds are
    present, it is best for each molecule to have its own component
    properties.
    
    Parameters:
    -----------
    Tb : np.array
        Array of boiling point temperatures for each pseudo-component in the
        oil (K).
    M : np.array
        Array of molecular weights for each pseudo-component in the oil 
        (g/mol).
    rho : np.array
        Density of each pseudo-component in the oil (kg/m^3).
    
    Returns:
    --------
    Tc : np.array
        Array of critical point temperatures for each pseudo-component in the
        oil (K).
    Pc : np.array
        Array of critical point pressures for each pseudo-component in the
        oil (Pa).
    Vc : np.array
        Array of critical point specific volumes for each pseudo-component in 
        the oil (m^3/mol).
    M : np.array
        Array of the molecular weights of each pseudo-component in the oil 
        (kg/mol).
    omega : np.array
        Array of acentric factors (--) for each pseudo-component in the oil.
    delta : np.array
        Two-dimensional array of the binary interaction coefficients (--) 
        for the interactions of each pseudo-component in the oil.
    
    Notes
    -----
    The references used here are:
    
    Chen, D.H., Dinivahi, M.V., Jeng, C.Y., 1993. New acentric factor
    correlation based on the Antoine equation. Ind. Eng. Chem. Res. 32,
    241-244.
    
    Pedersen, K.S., Christensen, P.L., Shaikh, J.A., 2014. Phase Behavior of
    Petroleum Reservoir Fluids, 2nd ed. CRC Press, Boca Raton, Florida.
    
    Twu, C.H., 1984. An internally consistent correlation for predicting the
    critical properties and molecular weights of petroleum and coal-tar
    liquids. Fluid Phase Equilib. 16, 137-150.
    
    """
    # Compute the specific gravity using the density of water at 60 deg F = 
    # 15.555... def C
    rho_0 = 999.# 999.0632006915614
    sg_adios = rho / rho_0
    
    # First, we apply correlations in Twu (1983) -----------------------------
    
    # These correlations are in English units...perform some unit conversions
    Tb = Tb * 9. / 5. # K to R
    
    # Use Equation (1) in Twu (1983) to estimate the critical temperature in 
    # Rankine
    Tc_0 = Tb / (0.533272 + (Tb * 0.191017e-3) + (Tb**2 * 0.779681e-7) +
        - (Tb**3 * 0.284376e-10) + 0.959468e28 / Tb**13)
    
    # Equation (5) in Twu (1983) estimates a parameter alpha
    alpha = 1. - Tb / Tc_0
    
    # Use Equation (8) in Twu (1983) to estimate the critical pressure in 
    # psia
    Pc_0 = (3.83354 + 1.19629 * alpha**(0.5) + 34.8888 * alpha 
        + 36.1952 * alpha**2 + 104.193 * alpha**4)**2
    
    # Use Equation (2) in Twu (1983) to estimate the critical specific volume
    # in ft^3 / (lb mol)
    Vc_0 = (1. - (0.419869 - 0.505839 * alpha - 1.56436 * alpha**3 +
        - 9481.70 * alpha**14))**(-8)
    
    # Use Equation (3) in Twu (1983) to estimate the specific gravity at 60
    # deg F relative to that of water at the same temperature (unitless)
    sg_twu = 0.843593 - 0.128624 * alpha - 3.36159 * alpha**3 + \
        - 13749.5 * alpha**12
    
    # Equation (4) in Twu (1983) is used to estimate the molecular weight, 
    # but this equation cannot be solved directly.  Write a function that 
    # can be used with a root-finding method to obtain the molecular
    # weight
    def twu_eq4(theta, Tb):
        """
        Function containing Equation (4) in Twu (1983)
        
        This function is used to compute the molecular weight using a root-
        finding algorithm.
        
        Parameters
        ----------
        theta : float
            Estimate for the log of the molecular weight, ln(M).
        Tb : float
            Estimate for the boiling point (R).
        
        Returns
        -------
        Residual of Equation (4) in Twu (1983) appropriate for use to compute
        theta from a root-finding algorithm.
        
        """
        # Compute Equation (4) in Twu (1983) in a form useful for a root-
        # finding algorithm.
        residual = np.exp(5.71419 + 2.71579 * theta - 0.286590 * 
            theta**2 - 39.8544 / theta - 0.122488 / theta**2) - 24.7522 * \
            theta + 35.3155 * theta**2 - Tb
        
        return residual
    
    # Initialize an array to hold the results
    M_twu = np.zeros(M.shape)
    
    # Estimate the molecular weight of each pseudocomponent
    for i in range(len(M)):
        
        # Equation (7) in Twu (1983) gives an appropriate initial guess
        M_0 = Tb[i] / (10.44 - 0.0052 * Tb[i])
        
        # Use fsolve() to solve Equation (4) in Two (1983)
        theta = fsolve(twu_eq4, np.log(M_0), args=(Tb[i],))
        
        # Convert theta to molecular weight
        M_twu[i] = np.exp(theta)
    
    # Apply Equations (11) - (13) in Twu (1983)
    delta_sg_t = np.exp(5. * (sg_twu - sg_adios)) - 1.
    f_T = delta_sg_t * (-0.362456 / np.sqrt(Tb) + 
        (0.0398285 - 0.948125 / np.sqrt(Tb)) * delta_sg_t)
    Tc = Tc_0 * ((1. + 2. * f_T) / (1. - 2. * f_T))**2
    
    # Apply Equations (14) - (16) in Twu (1983)
    delta_sg_v = np.exp(4. * (sg_twu**2 - sg_adios**2)) - 1.
    f_V = delta_sg_v*(0.466590 / np.sqrt(Tb) +
        (-0.182421 + 3.01721 / np.sqrt(Tb)) * delta_sg_v)
    Vc = Vc_0 * ((1. + 2. * f_V) / (1. - 2. * f_V))**2
    
    # Apply Equations (17) - (19) in Twu (1983)
    delta_sg_p = np.exp(0.5 * (sg_twu - sg_adios)) - 1.
    f_P = delta_sg_p * (2.53262 - 46.1955 / np.sqrt(Tb) - 0.00127885*Tb +
        (-11.4277 + 252.140 / np.sqrt(Tb) + 0.00230535 * Tb) * delta_sg_p)
    Pc = Pc_0 * Tc / Tc_0 * Vc_0 / Vc * ((1. + 2. * f_P) / (1. - 2. * f_P))**2
    
    # Second, we apply correlations in Chen et al. (1993) --------------------
    
    # Now, work on equations in Kesler-Lee (taken from Chen (1993))...need
    # to convert over the SI units.
    Pc_atm = Pc * 6.8046e-2   # (atm)
    Pc_bar = Pc *  6894.76e-5 # (bar)
    Tc_K = Tc * 5. / 9.       # (K)
    Tb_K = Tb * 5. / 9.       # (K)
    
    # Compute acentric factor from Equation (4) in Chen (1993)
    theta = Tb_K / Tc_K
    omega = (-np.log(Pc_atm) - 5.92714 + 6.09648 / theta + 1.28862 *
        np.log(theta) - 0.169347 * theta**6) / (15.2518 - 15.6875 / theta 
        -  13.4721 * np.log(theta) + 0.43577 * theta**6)
    
    # Thrid, put results in the right units ----------------------------------
    
    # Convert all results to SI base units of TAMOC
    Tc = 5./ 9. * (Tc - 459.67 - 32) + 273.15   # (K)
    Pc = Pc * 6894.76                           # (Pa)
    M = M / 1000.                               # [kg/mol)
    Vc = Vc / 453.59237 * 0.3048**3             # (m^3/mol)
    
    # Fourth, get the binary interaction coefficients ------------------------
    
    # Use the Pedersen method to estimate the binary interaction 
    # coefficients, i.e., delta.
    delta = pedersen(M)
    
    return (Tc, Pc, Vc, M, omega, delta)


def pedersen(M, composition=[]):
    """
    Estimate the binary interaction coefficients
    
    Use the method in Pedersen et al. (2014) to estimate the binary
    interaction coefficients and return them as a matrix, delta.
        
    This method is reported in Pedersen, K.S., Christensen, P.L., Shaikh,
    J.A., 2014. Phase Behavior of Petroleum Reservoir Fluids, 2nd ed. CRC
    Press, Boca Raton, Florida.  This method is valid for hydrocarbon-
    hydrocarbon interactions.  For atmospheric gases, there needs to be 
    a correction.  We use the guidance in Table 4.2 in the Pedersen et al.
    book for CO2 and N2.  We further assume N2 and O2 function the same.
    Any other gas is not treated; hence, we set the binary interaction
    parameters to zero.
    
    Parameters
    ----------
    M : np.array (len M)
        Array of molecular weights (mol/kg)
    composition : list, default=[]
        List of chemical components in the mixture.  If this list is not 
        provided, we assume that all compounds are hydrocarbons.  If the list
        is provided, we use it only to find the atmospheric gases nitrogen, 
        oxygen, and carbon dioxide and correct these.  We also search for 
        other non-hydrocarbons and set their parameters to zero.
        
    Returns
    -------
    delta : np.array (len M, len M)
        Array of binary interaction coefficients
    
    """
    # Initialize a matrix to hold delta
    delta = np.zeros((len(M), len(M)))
    
    # Compute each off-diagonal term using the correlation to hydrocarbon-
    # hydrocarbon mixtures.
    for i in range(len(M)):
        for j in range(len(M)):
            if i != j:
                # Take the appropriate ratio
                delta[i,j] = 0.00145 * np.max(np.array([M[j] / M[i], 
                    M[i] / M[j]]))
    
    # Next, make corrections based on the given composition
    if len(composition) > 0:
        
        # Get the compositions of air and natural gas
        air = ['nitrogen', 'oxygen', 'carbon_dioxide']
        gas = ['methane', 'ethane', 'propane', 'isobutane', 'n-butane']
        
        # Get the values of delta between air (rows) and the natural gases
        # (columns)
        delta_air_gas = np.array([
            [0.0311, 0.0515, 0.0852, 0.1033, 0.08], 
            [0.0311, 0.0515, 0.0852, 0.1033, 0.08],
            [0.12, 0.12, 0.12, 0.12, 0.12]
        ])
        
        # Get the values of data between air and longer-chain hydrocarbons
        delta_air_hydro = np.array([0.08, 0.08, 0.01])
        
        # For each atmospheric gas, correct the delta values
        for i in range(len(air)):
            
            # Only correct components in the composition
            if air[i] in composition:
                
                # Get the index to this gas in the composition
                air_idx = composition.index(air[i])
                
                # Set all binary interaction parameters to the value between
                # this gas and a hydrocarbon
                delta[air_idx,:] = delta_air_hydro[i]
                delta[:,air_idx] = delta_air_hydro[i]
                delta[air_idx,air_idx] = 0.
                
                # Correct the values for interaction between this atmospheric
                # gas and a light hydrocarbon (natural gas)
                for j in range(len(gas)):
                    
                    # Only correct components in the composition
                    if gas[j] in composition:
                        
                        # Get the  index to this natural gas compound in 
                        # the composition
                        gas_idx = composition.index(gas[j])
                        
                        # Set these binary interaction coefficients to the 
                        # correct values
                        delta[air_idx,gas_idx] = delta_air_gas[i,j]
                        delta[gas_idx,air_idx] = delta_air_gas[i,j]
        
        # Set the names of known compounds
        chems = ['Saturates', 'Aromatics', 'Resins', 'Asphaltenes'] + air + \
            gas
        
        # For any other components of the mixture, set the delta values to 
        # zero
        for i in range(len(composition)):
            
            # Check if we recognize this chemical
            known = False
            for j in range(len(chems)):
                # Note that component names look like 'Aromatics5', so
                # we have to check whether the base name 'Aromatics' is
                # in this compound name
                if chems[j] in composition[i]:
                    known = True
            
            # Set delta to zero for unknown chemicals
            if not known:
                # This compound does not have a base name common with any
                # compound in chems...set these binary interaction parameters 
                # to zero
                delta[i,:] = 0.
                delta[:,i] = 0.
    
    # Return the result
    return delta


def Vc_tuning(mass_frac, composition, rho_data, rho_i, delta, user_data):
    """
    Tune Vc to get better density estimates
    
    Tune the molar specific volume at the critical point in order to get 
    better agreement between the measured and modeled densities.
    
    Parameters
    ----------
    mass_frac : np.array
        Array of mass fractions of the dead oil compounds in the mixture.
        All of the numbers in this array also sum to 1.
    composition : list
        List of strings containing unique names for each chemical in the 
        present oil composition.
    rho_data : np.array
        List of NOAA Oil Library models.Density objects.  These objects 
        contain measurements of the whole oil density at different
        temperature and weathering conditions.
    rho_i : np.array
        Array of densities of each pseudo-component in the Adios database,  
        re-scaled such that:
            density whole oil = (1./np.sum(mass_frac/rho_i))
    delta : np.array
        Array of binary interaction coefficients
    user_data : dict
        A dictionary of chemical property data in the format expected by 
        the tamoc.dbm module objects.
    
    Returns
    -------
    user_data : dict
        A dictionary of updated chemical property data in the format expected 
        by the tamoc.dbm module objects.
    
    """
    # Count the number of oil components
    nc = len(mass_frac)
    
    # The component densities in Adios are specified at the following 
    # conditions
    T = 288.15
    P = 101325.
        
    # Create a function for optimizing Vc such that the difference between
    # the TAMOC pseudocomponent density and the reported pseudocomponent
    # density is minimized
    def delta_rho(Vc, component, user_data, rho_i, T, P):
        """
        Compute the density difference between TAMOC and a measurement
        
        Compute the density of a single oil component in TAMOC and compare
        this density to a reported measured value.
        
        Parameters
        ----------
        Vc : float
            Value of the molar specific volume at the critical point 
            (m^3/mol)
        component : str
            String name of the oil pseudocomponent to compute.
        user_data : dict
            A dictionary of chemical property data in the format expected by 
            the tamoc.dbm module objects.
        rho_i : float
            Measured value of the density of this pseudocomponent (kg/m^3)
        T : float
            Temperature at which to compute properties (K)
        P : float
            Pressure at which to compute properties (Pa)
        
        Returns
        -------
        delta_rho : float
            Absolute value of the difference between the density computed in 
            TAMOC and the density measured in the Adios database, re-scaled 
            such that:
                density whole oil = (1./np.sum(mass_fraction/rho_i))
        
        Notes
        -----
        Since this function computes the density of a single oil component, 
        the binary interaction coefficient is zero and the parameter `delta`
        is not needed by this function.
        
        """
        # Update the value of Vc in the user_data for the present oil 
        # component
        user_data[component]['Vc'] = Vc
        
        # Create a dbm.FluidMixture object for this oil component with the
        # present user data.
        oil_comp = dbm.FluidMixture(component, user_data=user_data)
        
        # Compute the density in TAMOC and select the liquid density
        rho_tamoc = oil_comp.density(np.array([1.]), T, P)[1,0]
        
        # Return the absolute value of the difference between the TAMOC and 
        # Adios densities
        return np.abs(rho_tamoc - rho_i)
    
    # Find the optimal value of Vc for each component of the oil
    for i in range(nc):
        
        # Extract the original value of Vc as initial value for the search
        Vc_0 = user_data[composition[i]]['Vc']
        
        # Minimize the error between the model and measurement
        Vc = fmin(delta_rho, Vc_0, args=(composition[i], user_data, rho_i[i],
            T, P), disp=0)[0]
        
        # Store the optimized value in user_data
        user_data[composition[i]]['Vc'] = Vc
    
    # Extract the density information reported in the Adios database for the
    # whole oil.
    T_0 = np.zeros(len(rho_data))
    rho_0 = np.zeros(len(rho_data))
    w_0 = np.zeros(len(rho_data))
    for i in range(len(rho_data)):
        T_0[i] = rho_data[i].ref_temp_k
        rho_0[i] = rho_data[i].kg_m_3
        w_0[i] = rho_data[i].weathering
    
    # Compare the TAMOC predictions to each of the measurements in Adios that
    # ignore weathering.
    num = 0.
    sum_abs = 0.
    sum_rel = 0.
    oil = dbm.FluidMixture(composition, user_data=user_data, delta=delta)
    for i in range(len(rho_0)):
        if w_0[i] == 0.:
            # Compute the whole-oil density in TAMOC
            rho_tamoc = oil.density(mass_frac, T_0[i], P)[1,0]
            rho_adios = rho_0[i]
            
            # Update statistics
            sum_abs += np.abs(rho_tamoc - rho_adios)
            sum_rel += np.abs(rho_tamoc - rho_adios) / rho_adios
            num += 1.
    
    # Compute statistics
    err_abs = sum_abs / num
    err_rel = sum_rel / num
    
    # Report the statistics
    print '\n----------'
    print 'Average absolute error for density:  %6.3f (kg/m^3) ' % err_abs
    print 'Average relative error for density:  %6.4f (--)' % err_rel
    print '----------\n'
    
    # Return the optimized user_data dictionary
    return user_data


def sequence_names(sara_names, name):
    """
    Add sequential counters to the pseudo-component name in sara_names
    
    Parameters
    ----------
    sara_names : list
        List of strings containing the names of each pseudo-component in an 
        oil based on SARA analysis.  
    name : str
        Name of the pseudo-component to edit ('Saturates' or 'Aromatics')
    
    Notes
    -----
    Since the sara_names are stored in a list, this function makes use of the
    fact that Python lists are mutable; hence, changes made to sara_names in
    this function will also be reflected for that variable in the calling
    function or program.
    
    """
    id_num = 1
    for i in range(len(sara_names)):
        if sara_names[i] == name:
            sara_names[i] = sara_names[i] + str(id_num)
            id_num += 1


def format_dbm_data(composition, M, Pc, Tc, omega, kh_0, neg_dH_solR, nu_bar, 
    K_salt, Vc=None, Tb=None, Vb=None, B=None, dE=None):
    """
    Format the chemical property data for use by tamoc.dbm
    
    Format the chemical property data into the dictionaries expected by the
    discrete bubble model (dbm) module FluidMixture and FluidParticle objects
    of TAMMOC.
    
    Parameters
    ----------
    composition : list
        List of strings containing unique names for each chemical in the 
        present oil composition.
    M : ndarray, size (nc)
        Molecular weights (kg/mol)
    Pc : ndarray, size (nc)
        Critical pressures (Pa)
    Tc : ndarray, size (nc)
        Critical temperatures (K)
    omega : ndarray, size (nc)
        Acentric factors (--)
    kh_0 : ndarray, size (nc)
        Henry's law constants at 298.15 K and 101325 Pa (kg/(m^3 Pa))
    neg_dH_solR : ndarray, size (nc)
        The negative of the enthalpies of solution / Ru (K).
    nu_bar : ndarray, size (nc)
        Partial molar volumes at infinite dilution (m^3/mol)
    K_salt : ndarray, size(nc)
        Setschenow constants (m^3/mol)
    Vc : ndarray, size(nc), default=None
        Specific volume at the critical point (m^3/mol)
    Tb : ndarray, size(nc), default=None
        Boiling point (K)
    Vb : ndarray, size(nc), default=None
        Specific volume at the boiling point (m^3/mol)
    B : ndarray, size (nc), default=None
        White and Houghton (1966) pre-exponential factor (m^2/s)
    dE : ndarray, size (nc), default=None
        Activation energy (J/mol)
    
    Returns
    -------
    data : dict
        A dictionary of chemical property data in the format expected by 
        the tamoc.dbm module objects.
    units : dict
        A dictionary of units for each chemical component in the mixture in
        the format expected by the tamoc.dbm module objects.
    
    """
    # Count the number of components in the oil mixture
    nc = len(composition)
    
    # Set flags to -9999. as expected by the tamoc.dbm module for any 
    # parameters that were not passed to this function 
    if not isinstance(Vc, np.ndarray):
        Vc = np.zeros(nc) - 9999.
    if not isinstance(Tb, np.ndarray):
        Tb = np.zeros(nc) - 9999.
    if not isinstance(Vb, np.ndarray):
        Vb = np.zeros(nc) - 9999.
    if not isinstance(B, np.ndarray):
        B = np.zeros(nc) - 9999.
    if not isinstance(dE, np.ndarray):
        dE = np.zeros(nc) - 9999.
    
    # Disable aqueous dissolution of insoluble components
    for i in range(nc):
        if kh_0[i] < 0.:
            kh_0[i] = 0.
    
    # Create an emmpty dictionary of chemical property data
    data = {}
    
    # Fill the dictionary with the properties for each chemical component
    for i in range(nc):
        # Add this chemical
        data[composition[i]] = {
            'M' : M[i],
            'Pc' : Pc[i],
            'Tc' : Tc[i],
            'omega' : omega[i],
            'kh_0' : kh_0[i],
            '-dH_solR' : neg_dH_solR[i],
            'nu_bar' : nu_bar[i],
            'K_salt' : K_salt[i],
            'Vc' : Vc[i],
            'Tb' : Tb[i],
            'Vb' : Vb[i], 
            'B' : B[i], 
            'dE' : dE[i]
        }
    
    # This function requires user to provide data in SI units suitable 
    # for TAMOC.  Assume this has been done.
    units = {
        'M' : '(kg/mol)',
        'Pc' : '(Pa)',
        'Tc' : '(K)',
        'omega' : '(--)',
        'kh_0' : '(kg/(m^3 Pa))',
        '-dH_solR' : '(K)',
        'nu_bar' : '(m^3/mol)',
        'K_salt' : '(m^3/mol)',
        'Vc' : '(m^3/mol)',
        'Tb' : '(K)',
        'Vb' : '(m^3/mol)', 
        'B' : '(m^2/s)', 
        'dE' : '(J/mol)'
    }
    
    # Return the two dictionaries
    return (data, units)


def mix_gas_for_gor(dead_composition, dead_mass_frac, user_data, delta, gor):
    """
    Create a live oil with a given gas to oil ratio (GOR)
    
    Mix natural gas into a dead oil composition until a live oil with a given
    gas to oil ratio (GOR) is achieved.  This function add gas compounds to 
    the oil mixture and adjusts the mass fractions of the whole mixture until
    the given GOR results from an equilibrium calculation at standard 
    conditions (15 deg C and atmospheric pressure).  This method return the
    new chemical composition list, mass fractions, and binary interaction
    coefficients.  The user_data describing the dead oil components is not
    changed.
    
    Parameters
    ----------
    dead_composition : list
        List of strings containing the names of the oil components in the 
        dead oil from the Adios database.
    dead_mass_frac : np.array
        An array of mass fractions for all compounds in the dead oil from 
        the Adios database (kg).
    user_data : dict
        A dictionary of chemical property data in the format expected by 
        the tamoc.dbm FluidMixture and FluidParticle module objects.
    delta : np.array (len M, len M)
        Array of binary interaction coefficients
    gor : float
        Gas to oil ratio desired for a given live-oil release.  We add light
        gas compounds to the dead oil composition and iterate the gas 
        fractions until the GOR at standard conditions is returned.  Here, 
        gor has the standard units ft^3 gas per stock barrel of oil at 
        surface conditions.  The gas composition is specified in the 
        function natural_gas().
    
    Returns
    -------
    composition : list
        List of strings containing the names of all oil components in the 
        live oil.
    mass_frac : np.array
        An array of mass fractions for all compounds in the live oil (kg/kg).
    delta : np.array (len M, len M)
        An updated array of binary interaction coefficients that includes the
        interactions among all compounds in composition.
    
    """
    # Get the composition of a natural gas
    gas_comp, gas_mf = natural_gas()
    
    # Get a list of all compounds in the live oil and gas mixture
    composition = gas_comp + dead_composition
    
    # Initialize arrays to contain the oil and gas mass fractions and fill
    # them with the pure gas and dead oil compositions
    mf_gas = np.zeros(len(composition))
    mf_oil = np.zeros(len(composition))
    mf_gas[0:len(gas_mf)] = gas_mf
    mf_oil[len(gas_mf):] = dead_mass_frac
    
    # Create a dbm.FluidMixture object with this composition, but not yet
    # knowing the binary interaction coefficients
    oil = dbm.FluidMixture(composition, user_data=user_data)
    
    # Use the Pedersen method to estimate the binary interaction parameters
    delta = pedersen(oil.M, composition)
    
    # Create a new dbm.FluidMixture object with the correct binary
    # interaction matrix
    oil = dbm.FluidMixture(composition, delta=delta, user_data=user_data)
    
    # Set up the GOR at atmospheric pressure and 15 deg C
    P = 101325.
    T = 273.15 + 15.
    p_gas = oil.density(mf_gas, T, P)[0,0]
    p_oil = oil.density(mf_oil, T, P)[1,0]
    
    # Get an estimate for the gas fraction from the GOR
    v_gas = gor * 0.0283168  # converts ft^3 = m^3
    v_oil = 0.158987         # converts bbl to m^3
    m_gas = p_gas * v_gas
    m_oil = p_oil * v_oil
    beta = m_gas / (m_gas + m_oil)
    
    # Iterate to converge on a correct gas fraction to have the desired
    # gas to oil ratio
    beta = fsolve(gas_fraction, beta, args=(gor, oil, mf_gas, mf_oil, T, P))
    
    # Use the final value of beta to get the composition of oil and gas
    mass_frac = beta * mf_gas + (1. - beta) * mf_oil
    
    # Return this petroleum fluid property data
    return (composition, mass_frac, delta)


def natural_gas():
    """
    Define the composition of a natural gas
    
    Returns
    -------
    gas_compounds : list
        List of the string names of the gaseous compounds included in this
        description of a natural gas.  These names are in the general 
        database of chemical properties provided with TAMOC.
    gas_fractions : np.array
        Array of the mass fractions of each gaseous compound in the pure
        gas (kg/kg)
    
    """
    gas_compounds = ['methane', 'ethane', 'propane', 'isobutane', 'n-butane']
    gas_fractions = np.array([93.9, 4.2, 1.84, 0.03, 0.03])
    
    return (gas_compounds, gas_fractions)


def gas_fraction(beta, gor_0, oil, mf_gas, mf_oil, T, P):
    """
    Compute the difference between the GOR of a mixture and a given GOR
    
    Computes the equilibrium gas-liquid partitioning for a given oil and 
    mixture composition and then reports the difference between the gas to 
    oil ratio (GOR) of this mixture to a desired value of the GOR, gor_0.  
    Equilibrium calculations are done at the input conditions of temperature
    and pressure.  GOR is computed in the units of ft^3 of gas to barrels 
    of liquid oil at the given T and P.
    
    Parameters
    ----------
    beta : float
        The faction of pure gas in the mixture (0 to 1)
    gor_0 : float
        The desired gas to oil ratio of the oil mixture (ft^3/bbl)
    oil : dbm.FluidMixture 
        A dbm.FluidMixture object for the present oil composition.
    mf_gas : np.array
        Array of the mass fractions of the gaseous compounds in the mixture.
        All of the numbers in this array sum to 1.
    mf_oil : np.array
        Array of the mass fractions of the dead oil compounds in the mixture.
        All of the numbers in this array sum to 1.
    T : float
        Temperature to compute gas to oil ratio (K)
    P : float
        Pressure to compute gas to oil ratio (Pa)
    
    Returns
    -------
    delta_gor : float
        The difference between the present GOR of this mixture and the input
        value of GOR, gor_0.
    
    """
    # Set up a mass fraction array containing all of the oil compounds
    mass_frac = beta * mf_gas + (1. - beta) * mf_oil
    
    # Compute the equilibrium conditions for this oil mixture
    m, xi, K = oil.equilibrium(mass_frac, T, P)
    
    # Compute the volume of gas and oil
    mf_gas = m[0,:]   # mass fractions of all compounds in the gas phase
    p_gas = oil.density(mf_gas, T, P)[0,0]
    mf_oil = m[1,:]   # mass fractions of all compounds in the liquid phase
    p_oil = oil.density(mf_oil, T, P)[1,0]
    v_gas = np.sum(mf_gas) / p_gas  # m^3
    v_oil = np.sum(mf_oil) / p_oil  # m^3
    
    # Report the gas to oil ratio in standard units
    v_gas = v_gas / 0.0283168   # m^3 to ft^3
    v_oil = v_oil / 0.158987    # m^3 to barrels
    gor = v_gas / v_oil
    
    # Return the deviation from the desired gor
    return gor - gor_0


def set_mass_fluxes(composition, mass_frac, user_data, delta, q_oil):
    """
    Compute the mass fluxes to achieve a desired oil flow rate
    
    Compute the mass fluxes of each gas and liquid oil component to achieve
    a given oil flow rate (q_oil) in barrels per day at standard conditions
    (15 deg C and atmospheric pressure).
    
    Parameters
    ----------
    composition : list
        List of strings containing the names of the oil components in the 
        live oil.
    mass_frac : np.array
        An array of mass fractions for all compounds in the live oil (kg/kg).
    user_data : dict
        A dictionary of chemical property data in the format expected by 
        the tamoc.dbm FluidMixture and FluidParticle module objects.
    delta : np.array (len M, len M)
        Array of binary interaction coefficients
    q_oil : float
        Flow rate of oil (bbl/d at standard conditions).
    
    Returns
    -------
    mass_flux : np.array
        An array of gas and liquid mass fluxes for each chemical component
        in the mixture (kg/s) required to achieve the desired flow rate of 
        dead oil at the surface, q_oil.
    
    
    """
    # Create a dbm.FluidMixture object
    oil = dbm.FluidMixture(composition, delta=delta, user_data=user_data)
    
    # Get the equilibrium at standard conditions
    P0 = 101325.
    T0 = 273.15 + 15.
    m0, xi0, K0 = oil.equilibrium(mass_frac, T0, P0)
    
    # Get the volume flow rate of liquid oil at standard conditions for a 
    # total petroleum fluid flow rate of 1 kg/s (e.g., using mass_flux equal
    # to mass_frac)
    p_oil = oil.density(m0[1,:], T0, P0)[1,0]
    v_oil = np.sum(m0[1,:]) / p_oil / 0.158987 # bbl
    
    # Adjust the masses to yield the desired flow rate of oil in bbl/d
    k_fac = (q_oil / 86400.) / v_oil
    mass_flux = mass_frac * k_fac
    
    return mass_flux


def print_chemdata(user_data, units, chems=None):
    """
    Print a table of the chemical property data
    
    Print a readable table of the chemical property data stored in the 
    user_data dictionary used by the TAMOC discrete bubble model (dbm 
    module).
    
    Parameters
    ----------
    user_data : dict
        A dictionary of chemical property data in the format expected by 
        the tamoc.dbm module objects.
    units : dict
        A dictionary of units for each of the chemical properties in the
        database
    chems : list, default=[]
        An optional list of chemicals to print. Each chemical must be listed
        in the user_data dictionary. If no input is given, then all chemicals
        in user_data will print.
    
    """
    # Put user_data in a list if it is not already a list
    if not isinstance(user_data, list):
        user_data = [user_data]
    
    # Decide how many chemical components to print
    if chems == None:
        names = user_data[0].keys()
    else:
        names = chems
    
    # Get the names of the variables stored in the chemical database
    variables = user_data[0][names[0]].keys()
    
    # Print the data by component
    print '\n----------'
    for name in names:
        print name
        for var in variables:
            line = ['    ' + var + ' = ']
            for k in range(len(user_data)):
                line.append('%g, ' % user_data[k][name][var]) 
            line.append(' ' + str(units[var]))
            print ''.join(line)
    print '----------\n'


def print_composition(composition, mass_frac):
    """
    Print a table of the chemical composition
    
    Print a readable table of the composition data (names of the pseudo-
    components and the mass fractions of each).
    
    Parameters
    ----------
    composition : list
        List of strings containing the names of the oil components in the 
        dead oil from the Adios database.
    mass_frac : np.array
        An array of mass fractions for all compounds in the dead oil from 
        the Adios database (kg).
    
    """
    # Put the composition data in a list if it is not already a list
    if not isinstance(composition[0], list):
        composition = [composition]

    # Put the mass fraction data in a list if it is not already a list
    if not isinstance(mass_frac, list):
        mass_frac = [mass_frac]
    
    # Print the data by component
    print '\n----------'
    for i in range(len(composition[0])):
        names = []
        line = ['    ']
        for j in range(len(mass_frac)):
            names.append(composition[j][i] + ', ')
            line.append('%g, ' % mass_frac[j][i])
        print ''.join(names)
        print ''.join(line)
    print '----------\n'


def print_petroleum_props(comp, mass_frac, data, delta, T, S, P, q_oil=None):
    """
    Compute the gas and liquid properties of a petroleum fluid
    
    Compute the gas and liquid properties of a petroleum fluid described by a
    dbm.FluidMixture object (oil) and a composition (mass_frac) at a given
    temperature (T), salinity (psu), and pressure (P). Print the results to
    the screen for review.
    
    Parameter
    ---------
    comp : list
        List of strings containing the names of the oil components in the 
        live oil mixture.
    mass_frac : np.array
        An array of mass fractions for all compounds in the live oil mixture
        (kg/kg).
    data : dict
        A dictionary of chemical property data in the format expected by 
        the tamoc.dbm FluidMixture and FluidParticle module objects.
    delta : np.array (len M, len M)
        Array of binary interaction coefficients
    T : float
        Temperature to compute gas-liquid equilibrium and viscosity (K)
    S : float
        Salinity to compute interfacial tension (psu)
    P : float
        Pressure to compute gas-liquid equilibrium and viscosity (Pa)
    q_oil : float, default=None
        Optional flow rate of oil (bbl/d at standard conditions).  This is 
        used to report the mass and volume flow rates of gas and liquid at
        the given T, P that are equivalent to this amount of oil.
    
    Returns
    -------
    mass_flux : np.array
        An array of the mass fluxes of each compound in the oil mixture
        (kg/s) so that the desired oil flow rate and GOR is achieved.  If
        q_oil is not specified, this mass flux array will be a total of
        1 kg/s such that it is equivalent to the mass_frac array (kg/kg).
    
    """
    # Create a dbm.FluidMixture object
    oil = dbm.FluidMixture(comp, delta=delta, user_data=data)
    
    # Compute the equilibrium composition
    print '\nComputing oil/gas equilibrium at:'
    print '---------------------------------'
    print '    T = %g (K)' % T
    print '    S = %g (psu)' % S
    print '    P = %g (Pa)' %P
    m, xi, K = oil.equilibrium(mass_frac, T, P)
    
    # Get the properties of gas
    mf_gas = m[0,:]   # mass fractions of all compounds in the gas phase
    rho = oil.density(mf_gas, T, P)[0,0]
    mu = oil.viscosity(mf_gas, T, P)[0,0]
    sigma = oil.interface_tension(mf_gas, T, S, P)[0,0]
    
    # Print a table to properties
    print '\nGas Properties:'
    print '---------------'
    print '    density (kg/m^3)        : ', rho
    print '    viscosity (Pa s)        : ', mu
    print '    interface tension (N/m) : ', sigma

    # Get the properties of the oil
    mf_oil = m[1,:]   # mass fractions of all compounds in the liquid phase
    rho = oil.density(mf_oil, T, P)[1,0]
    mu = oil.viscosity(mf_oil, T, P)[1,0]
    sigma = oil.interface_tension(mf_oil, T, S, P)[1,0]
    
    # Print a table to properties
    print '\nOil Properties:'
    print '---------------'
    print '    density (kg/m^3)        : ', rho
    print '    viscosity (Pa s)        : ', mu
    print '    interface tension (N/m) : ', sigma
    
    # Print seawater properties
    print '\nSeawater Properties:'
    print '--------------------'
    print '    density (kg/m^3) : ', seawater.density(T, S, P)
    print '    viscosity (Pa s) : ', seawater.mu(T, S, P)
    
    # Get the mass flux for the given oil flow rate
    if q_oil == None:
        mass_flux = mass_frac
    else:
        mass_flux = set_mass_fluxes(comp, mass_frac, data, delta, q_oil)
    
    # Compute the volume flow rates at the release
    m, xi, K = oil.equilibrium(mass_flux, T, P)
    p_gas = oil.density(m[0,:], T, P)[0,0]
    md_gas = np.sum(m[0,:])
    q_gas = md_gas / p_gas
    p_oil = oil.density(m[1,:], T, P)[1,0]
    md_oil = np.sum(m[1,:])
    q_oil = md_oil / p_oil
    
    # Print a table or properties
    print '\nIn Situ Volume Flow Rates:'
    print '--------------------------'
    print '    gas flow rate (m^3/s)   : ', q_gas
    print '    gas flow rate (ft^3/d)  : ', q_gas * 86400. / 0.0283168
    print '\n    oil flow rate (m^3/s)   : ', q_oil
    print '    oil flow rate (bbl/d)   : ', q_oil * 86400. / 0.158987
    print '\n    GOR (m^3/m^3)           : ', q_gas / q_oil
    print '    GOR (ft^3/bbl)          : ', (q_gas / 0.0283168) / \
                                           (q_oil / 0.158987)
    
    # Return the oil composition with the correct flow rates
    return mass_flux

