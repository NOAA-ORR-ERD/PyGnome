.. include:: ../links.rst

Spills
======

The Spill Class
---------------

Setting up the Spill object can be tricky because it requires both a Release object and a Substance Object. Some
helper functions are described in the next section that simplify this task. But first, we'll 
show some detail on creating a spill to help understand the components.
 
The :class:`gnome.spill.release.Release` Object specifies the details of the release (e.g. where, when, how many elements). 
Some of the subclasses of this include:

* :class:`gnome.spill.release.PointLineRelease` - a release of particles at a point or along a line, either instantaneously or over a time interval
* :class:`gnome.spill.release.SpatialRelease` - an instantaneous release of particles distributed randomly in a specified polygon 

The :class:`gnome.spill.substance.Substance` Object provides information on the type of substance spilled. Although, its possible to add multiple spills to the model, they must all use the same substance object. There are two subclasses that can be used to instantiate substances:

* :class:`gnome.spill.substance.GnomeOil` - used for creating a spill that will include oil weathering processes
* :class:`gnome.spill.substance.NonWeatheringSubstance` - used for running transport simulations with conservative particles (i.e. the particle properties do not change over time).

Here's an example setting up a non-weathering spill. This is the default Substance for a spill so we do not need to create or pass in a Substance object::

    import gnome.scripting as gs
    from datetime import datetime, timedelta
    start_time = datetime(2015, 1, 1, 0, 0)
    model = gs.Model(start_time=start_time,
                  duration=timedelta(days=3),
                  time_step=60 * 15, #seconds
                  )
    release = gs.PointLineRelease(release_time=start_time,start_position=(-144,48.5,0),num_elements=1000)
    spill = gs.Spill(release=release)
    model.spills += spill

.. admonition:: Creating spills with oil data from the ADIOS database

    Specific oils can be downloaded from the |adios_db|. The oil properties are stored in the JSON file format which can be read using any text editor. This file can then be used to instantiate a GnomeOil. In the following examples, we use an Alaska North Slope Crude downloaded from the database. That file can be accessed :download:`here <alaska-north-slope_AD00020.json>` to use in the following examples.
    
To model a spill of 5000 bbls using a specific oil downloaded from the ADIOS oil database (adios.orr.noaa.gov) we could instantiate the Spill oject like this::
    
    import gnome.scripting as gs
    from datetime import datetime, timedelta
    start_time = datetime(2015, 1, 1, 0, 0)
    model = gs.Model(start_time=start_time,
                  duration=timedelta(days=3),
                  time_step=60 * 15, #seconds
                  )
    release = gs.PointLineRelease(release_time=start_time,start_position=(-144,48.5,0),num_elements=1000)  
    substance = gs.GnomeOil(filename='alaska-north-slope_AD00020.json')
    spill = gs.Spill(release=release,substance=substance,amount=5000,units='bbls')
    model.spills += spill
 

.. admonition:: A note on "Windage"

    Floating objects experience a drift due to the wind. The default for substances is to have windage values set in the range 1-4% with a persistence of 15 minutes. More detail on the wind drift parameterization can be found in the |gnome_tech_manual|. 

Spatial releases
----------------

Documentation forthcoming (Nov 2021).
 
Using helper functions
----------------------

Rather than deal with the complexities of the Spill class directly, helper functions in the scripting package 
can be utilized for a lot of typical use cases. Some examples are include below.

Surface spill 
~~~~~~~~~~~~~

We use the :func:`gnome.scripting.surface_point_line_spill` helper function to inialize a release along a line 
that occurs over one day. The oil type is specified using the sample oil file provided above with a spill volume 
of 5000 barrels. Here we change the default windage range to be 1-2% with an infinite persistence (particles keep the same windage value for all time). The helper function creates both the Release and the Substance objects and uses them to create a Spill object.
::

    import gnome.scripting as gs
    from datetime import datetime, timedelta
    start_time = datetime(2015, 1, 1, 0, 0)
    model = gs.Model(start_time=start_time,
              duration=timedelta(days=3),
              time_step=60 * 15, #seconds
              )
    spill = gs.surface_point_line_spill(num_elements=1000,
                                 start_position=(-144,48.5, -1000.0),
                                 release_time=start_time,
                                 end_position=(-144,48.6, 0.0),
                                 end_release_time= start_time + timedelta(days=1),
                                 amount=5000,
                                 substance=gs.GnomeOil(filename='alaska-north-slope_AD00020.json'),
                                 units='bbl',
                                 windage_range=(0.01,0.02),
                                 windage_persist=-1,
                                 name='My spill')
    model.spills += spill
    
    # ... add movers/weatherers
    
    model.full_run()
    
.. _subsurface_plume:

Subsurface plume
~~~~~~~~~~~~~~~~

For initialization of a subsurface plume, we can use the :func:`gnome.scripting.subsurface_plume_spill` 
helper function.
Required parameters in this case also include a specification of the droplet size distribution 
or of the rise velocities. The :mod:`gnome.utilities.distributions` module includes methods for 
specifying different types of distributions. In this case we specify a uniform distribution of
droplets ranging from 10-300 microns::
    
    import gnome.scripting as gs
    from gnome.utilities.distributions import UniformDistribution
    from datetime import datetime, timedelta
    start_time = datetime(2015, 1, 1, 0, 0)
    model = gs.Model(start_time=start_time,
              duration=timedelta(days=3),
              time_step=60 * 15, #seconds
              )
    ud = UniformDistribution(10e-6,300e-6) #droplets in the range 10-300 microns
    spill = gs.subsurface_plume_spill(num_elements=1000,
                                   start_position=(-144,48.5, -100.0),
                                   release_time=start_time,
                                   distribution=ud,
                                   distribution_type='droplet_size',
                                   end_release_time= start_time + timedelta(days=1),
                                   amount=5000,
                                   substance=gs.GnomeOil(filename='alaska-north-slope_AD00020.json'),
                                   units='bbl',
                                   windage_range=(0.01,0.02),
                                   windage_persist=-1,
                                   name='My spill')
                                   
    model.spills += spill
    
    # ... add movers/weatherers
    
    model.full_run()

    
