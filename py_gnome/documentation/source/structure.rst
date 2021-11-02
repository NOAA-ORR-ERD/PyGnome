PyGNOME Structure
=================

PyGNOME is fundamentally a Lagrangian element (particle tracking) mode. The oil (or
other tracer) is represented as Lagrangian elements (LEs), or particles with their 
movement and properties tracked over time. Particles move under the influence of 
ocean currents and/or direct wind forcing. Particles also may undergo physical changes
over time (e.g. oil weathering).

Setting up a simulation in PyGNOME involves inputting or loading data to instantiate 
various objects in the ``gnome`` package wich create, move, or and/or modify the particles.

Primary Base Classes in ``gnome``
---------------------------------

**The Model Object** :mod:`gnome.model.Model`
    The Model Object is used to initialize a scenario and run the simulation. It contains various parameters
    including model start time, duration, and time step. All other objects are added to the Model.

**The Map Object** :mod:`gnome.map.GnomeMap`
    The Map Object defines the domain of the model and handles all collision-related 
    effects such as particle beaching. It can consist of domain boundaries,
    shoreline data (to define where land and water are), and properties of the
    shoreline. For 3-d modeling, it can also define the bathymetry.

**Environment Objects** :mod:`gnome.environment.Environment`
    The environmental conditions the particles interact with are determined using
    Environment Objects. These objects are designed to represent data flexibly from a 
    large variety of sources and file formats. Simple scalars, time series data, or full 4D
    environment data in netCDF format are some examples of what can be used to create 
    environment objects.

**Movers** :mod:`gnome.movers.Mover`
    Movers repsesent physical processes that move the particles. They may utilize environment objects for 
    determining conditions at a particle location or may be simpler parameterizations (e.g. spatially 
    constant diffusion). 

**Weatherers** :mod:`gnome.weatherers.Weatherer`
    Weatherers represent processes that change the mass of the floating oil or of oil droplets
    within the water column. These include processes that are traditionally described as
    "weathering" (e.g. evaporation, dispersion) and response options (e.g. skimming, 
    burning oil). Weatherers utilize oil chemistry data along with environment objects.

**Spill Objects** :mod:`gnome.spill.Spill`
    A Spill Object is a composition of two objects:
    
    * A Release Object contains information on where and when particles are released
    * A Substance Object contains information about what was spilled. If the simulation includes weathering processes the Substance must be a GnomeOil. Otherwise, the default substance is NonWeatheringSubstance (a passive tracer).

**Outputters** :mod:`gnome.outputters.Outputter`
    Outputter Objects handle all aspects of exporting data from the
    model.

    The Renderer outputter renders a base map, and a set of transparent pngs that plot the
    positions of the elements, etc. These can be composited to make a movie of the
    simulation

    Various formats containing the element information can also be outputted -- at 
    present these include NetCDF, KMZ, and ESRI shapefiles.
 

 
  










