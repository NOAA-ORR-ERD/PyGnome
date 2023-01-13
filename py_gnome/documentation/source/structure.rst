#################
PyGNOME Structure
#################

PyGNOME is essentially a Lagrangian element (particle tracking) modelling framework. It was developed specifically to support oil spill modeling, but can be used for any number of other particle tracking applications.

The substance (oil or other tracer) is represented as Lagrangian elements (LEs), or particles, with their movement and properties tracked over time. As PYGNOME was originally developed to support oil spill modeling (and is used as the primary operational tool for NOAA's support of oil spills in US Waters), it comes built in with many features for oil spill modeling.

Individual elements can move under the influence of ocean currents, direct wind forcing, diffusion, or any other custom "mover". Elements can be transformed over time, including undergoing chemical/physical changes over time (e.g. oil weathering).

Setting up a simulation in PyGNOME involves inputting or loading data to instantiate various objects in the ``gnome`` package wich create, move, or and/or modify the particles.

Primary Base Classes in ``gnome``
---------------------------------

**The Model Object** :mod:`gnome.model.Model`
    The Model Object is used to initialize a scenario and run the simulation. It contains various parameters
<<<<<<< HEAD
    including model start time, duration, and time step. All other objects are added to the Model. The primary point of the Model is to provide the framework to act on the particles over time -- the model itself does not include any assumptions about what the elements represent or what is acting on them.
=======
    including model start time, duration, and time step. All other objects are added to the Model. The primariy point of the model is to act on the particles over time -- the model itself does not include any assumptions about what the elements represent or what is acting on them.
>>>>>>> 62e446c936a154803b1839f278a952ae703e3d86

**The Map Object** :mod:`gnome.map.GnomeMap`
    The Map Object defines the domain of the model and handles interaction with the boundaries, such as particle beaching.
    It can consist of domain boundaries, shoreline data (to define where land and water are), and properties of the shoreline. For 3-d modeling, it can also define the bathymetry.

**Environment Objects** :mod:`gnome.environment.Environment`
    The environmental conditions (e.g. wind, currents) the particles interact with are defined by Environment Objects.
    These objects are designed to represent data flexibly from a large variety of sources and file formats.
    Simple scalars, time series data, or full 4D environment data in netCDF format are some examples of what can be used to create
    environment objects.

**Movers** :mod:`gnome.movers.Mover`
    Movers represent physical processes that move the particles.
    They may utilize environment objects for determining conditions at a particle location or may be simpler parameterizations (e.g. spatially constant diffusion).

**Weatherers** :mod:`gnome.weatherers.Weatherer`
    Weatherers represent processes that change the mass or properties of the floating oil or of oil droplets within the water column. These include processes that are traditionally described as "weathering" (e.g. evaporation, dispersion) and response options (e.g. skimming, burning oil). Weatherers utilize oil property data along with environment objects.

**Spill Objects** :mod:`gnome.spills.Spill`
    A Spill Object is a composition of two objects:
    
    * A **Release Object** contains information on where and when elements are released by the model. :mod:`gnome.spills.release.Release`  
    * A **Substance Object**  defines the properties of the particles that affect how they might move or change in the environment. PyGNOME includes a sophisticated ``GnomeOil`` substance to support full oil weathering modeling, as well as a simple ``NonWeatheringSubstance`` that is subject only to movement, and can be used to represent a number of "passive tracers". :mod:`gnome.spills.substance.Substance`
    
**Outputters** :mod:`gnome.outputters.Outputter`
    Outputter Objects handle all aspects of exporting results from the model.

    The ``Renderer`` outputter renders an animated GIF or a sepate image for each timestep.

    Other Outputters can provide the element information in various formats, including NetCDF, KMZ, and ESRI shapefiles.
 

 
  










