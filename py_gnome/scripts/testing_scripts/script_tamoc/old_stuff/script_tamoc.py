#!/usr/bin/env python
"""
Script to test GNOME with:

TAMOC - Texas A&M Oil spill Calculator

https://github.com/socolofs/tamoc

This is a very simple environment:

Simple map (no land) and simple current mover (steady uniform current)

Rise velocity and vertical diffusion

But it's enough to see if the coupling with TAMOC works.

"""

import gnome.scripting as gs
import gnome.tamoc.tamoc_spill as ts

from gnome.environment import (Water, Waves)
from gnome.weatherers import (Evaporation, Dissolution)
from gnome.spills.substance import GnomeOil

import os
import numpy as np


def set_directory_structure():
    """
    Set up the base and output directories for a GNOME simulation

    Defines the directories for the base model run and the output.  If an
    output directory does not exist, this function will create that
    directory.

    Returns
    -------
    base_dir : os.path
        Directory path to the present directory where this file is stored.
    images_dir : os.path
        Directory path to the ./Images directory where visual output for
        this simulation will be stored.
    outfiles_dir : os.path
        Directory path to the ./Output directory where NetCDF and other data
        output for this simulation will be stored.

    """
    # Define the present directory as the base directory
    base_dir = os.path.dirname(__file__)

    # Store output in directories directly under the base directory
    images_dir = os.path.join(base_dir, 'Images')
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)

    outfiles_dir = os.path.join(base_dir, 'Output')
    if not os.path.exists(outfiles_dir):
        os.mkdir(outfiles_dir)

    return (base_dir, images_dir, outfiles_dir)


def base_environment(water_temp=280.928,
                     salinity=34.5,
                     wind_speed=5.,
                     wind_dir=117.):
    """
    Create a minimalist ocean environment that allows for surface weathering

    Create a water, wind, and waves environment for a GNOME simulation that
    allows for surface weathering processes

    Parameters
    ----------
    water_temp : float
        Temperature of the surface water (K)
    salinity : float
        Salinity of the surface ocean water (psu)
    wind_speed : float
        Wind speed (kt)
    wind_dir : float
        Wind direction (deg from North).  Per atmospheric modeling
        convention, this points in the direction from which the wind is
        coming.

    Returns
    -------
    water : gnome.environment.Water
        GNOME environment object that contains the water temperature (K)
    wind : gnome.environment.constant_wind
        GNOME environment object that contains the local wind (speed in
        knots and direction in deg from North)
    waves : gnome.environment.Waves
        GNOME environment object that uses the wind and water objects to
        predict the wave conditions.

    """
    # Create an ocean environment using GNOME environment objects
    water = Water(temperature=water_temp, salinity=salinity)
    wind = gs.constant_wind(wind_speed,
                            wind_dir,
                            'knots')
    waves = Waves(wind, water)

    return (water, wind, waves)


def make_model():
    """
    Set up a GNOME simulation that uses TAMOC

    Set up a spill scenario in GNOME that uses TAMOC to simulate a subsurface
    blowout and then pass the TAMOC solution to GNOME for the far-field
    particle tracking

    """
    # Set up the directory structure for the model
    base_dir, images_dir, outfiles_dir = set_directory_structure()

    # Set up the modeling environment
    print('\n-- Initializing the Model         --')
    start_time = "2019-06-01T12:00"
    model = gs.Model(start_time=start_time,
                     duration=gs.days(3),
                     time_step=gs.minutes(30),
                     uncertain=False)

    # Add a map
    print('\n-- Adding a Map                   --')
    model.map = gs.GnomeMap()

    # Add image output
    print('\n-- Adding Image Outputters        --')
    renderer = gs.Renderer(output_dir=images_dir,
                           image_size=(1024, 768),
                           output_timestep=gs.hours(1),
                           viewport=((-0.15, -0.35), (0.15, 0.35)))
    model.outputters += renderer

    # Add NetCDF output
    print('\n-- Adding NetCDF Outputter        --')
    if not os.path.exists(outfiles_dir):
        os.mkdir(outfiles_dir)
    netcdf_file = os.path.join(outfiles_dir, 'script_tamoc.nc')
    gs.remove_netcdf(netcdf_file)
    file_writer = gs.NetCDFOutput(netcdf_file,
                                  which_data='all',
                                  output_timestep=gs.hours(2))
    model.outputters += file_writer
    oil_file = os.path.join(base_dir, 'light-louisianna-sweet-bp_AD01554.json')
    subs = GnomeOil(filename=oil_file)

    # Add a spill object
    print('\n-- Adding a Point Spill           --')
    end_release_time = model.start_time + gs.hours(12)
    point_source = ts.TamocSpill(num_elements=100,
                                 start_position=(0.0, 0.0, 1000.),
                                 release_duration=gs.hours(24),
                                 release_time=start_time,
                                 substance=subs,
                                 release_rate=20000.,
                                 units='bbl/day',
                                 gor=500.,
                                 d0=0.5,
                                 phi_0=-np.pi / 2.,
                                 theta_0=0.,
                                 windage_range=(0.01, 0.04),
                                 windage_persist=900,
                                 name='Oil Well Blowout')

    model.spills += point_source

    # Create an ocean environment
    water, wind, waves = base_environment(water_temp=273.15+21.,
                                          wind_speed=5.,
                                          wind_dir=225.)

    # Add a uniform current in the easterly direction
    print('\n-- Adding Currents                --')
    uniform_current = gs.SimpleMover(velocity=(0.1, 0.0, 0.))
    model.movers += uniform_current

    # Add a wind mover
    wind_mover = gs.PointWindMover(wind)
    model.movers += wind_mover

    # Add particle diffusion...note, units are in cm^2/s
    print('\n-- Adding Particle Diffusion      --')
    particle_diffusion = gs.RandomMover3D(
                         horizontal_diffusion_coef_above_ml=100000.,
                         horizontal_diffusion_coef_below_ml=10000.,
                         vertical_diffusion_coef_above_ml=100.,
                         vertical_diffusion_coef_below_ml=10.,
                         mixed_layer_depth=15.)
    model.movers += particle_diffusion

    # Add rise velocity for droplets
    print('\n-- Adding Particle Rise Velocity  --')
    # fixme: we do have code for rise velocity:
    #  gnome.movers.RiseVelocityMover
    #  let's test that later
    slip_velocity = gs.SimpleMover(velocity=(0.0, 0.0, -0.1))
    model.movers += slip_velocity


    # Add dissolution
    print('\n-- Adding Weathering              --')
    evaporation = Evaporation(water=water,
                              wind=wind)
    model.weatherers += evaporation

    dissolution = Dissolution(waves=waves,
                              wind=wind)
    model.weatherers += dissolution

    return model


def run_model(model):
    """
    Run a GNOME simulation

    Steps through all of the simulation time steps of a gnome.Model object
    simulation

    Parameters
    ----------
    model : gnome.Model
        A gnome.Model simulation object

    Returns
    -------
    model : gnome.Model
        A gnome.Model simulation object

    """
    print('\n-- RUNNING TAMOC-GNOME SIMULATION --')
    for step in model:
        print('   Step: %.4i' % (step['step_num']))

    return model


if  __name__ == '__main__':

    # Initialize the model
    model = make_model()

    # Run the model
    #model = run_model(model)
