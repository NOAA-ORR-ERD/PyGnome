.. include:: ../links.rst

.. _scripting_spills:

Spills
======

The Spill Class
---------------

The :class:`gnome.spills.Spill` class is used for creating a Spill object to add to the model. This can be tricky because it requires both a Release object and a Substance object.
Some helper functions are available that simplify this task. But first, we'll show some detail on creating a spill to help understand the components.

The :class:`gnome.spills.release.Release` class is used to create a Release object which specifies the details of the release (e.g. where, when, how many elements).
Some of the subclasses of this include:

* :class:`gnome.spills.release.PointLineRelease` - a release of elements at a point or along a line, either instantaneously or over a time interval

* :class:`gnome.spills.release.PolygonRelease` - an instantaneous release of elements distributed randomly in specified polygons.

The :class:`gnome.spills.substance.Substance` class is used to create a Substance object which provides information on the type of substance spilled. It's possible to add multiple spills to the model, they must all use the same Substance object. There are currently two classes that can be used to instantiate substances:

* :class:`gnome.spills.substance.GnomeOil` - used for creating a spill that will include oil weathering processes

* :class:`gnome.spills.substance.NonWeatheringSubstance` - used for running transport simulations with conservative elements (i.e. the element properties do not change over time).

All of these classes are imported into the scripting module for convenience.

Non-weathering Example
----------------------

Here's an example setting up a non-weathering spill. This is the default Substance for a spill so we do not need to create or pass in a Substance object:

.. code-block:: python

    import gnome.scripting as gs
    start_time = "2015-01-01T00:00"
    model = gs.Model(start_time=start_time,
                     duration=gs.days(3),
                     time_step=60 * 15,  #15 minutes in seconds
                     )
    release = gs.PointLineRelease(num_elements=1000,
                                  release_time=start_time,
                                  start_position=(-144, 48.5, 0),
                                  )
    spill = gs.Spill(release=release)
    model.spills += spill

.. admonition:: Creating spills with oil data from the ADIOS database

    Specific oils can be downloaded from the |adios_db|. The oil properties are stored in the JSON file format which can be read using any text editor. This file can then be used to instantiate a ``GnomeOil``. In the following examples, we use an Alaska North Slope Crude downloaded from the database. That file can be accessed :download:`here <alaska-north-slope_AD00020.json>` to use in the following examples.

A spill of 5000 bbls using a specific oil downloaded from the `ADIOS Oil Database <https://adios.orr.noaa.gov>`_ can be used with the Spill object like this:

.. code-block:: python

    import gnome.scripting as gs
    start_time = "2015-01-01T00:00"
    model = gs.Model(start_time=start_time,
                     duration=gs.days(3),
                     time_step=60 * 15, # 15 minutes in seconds
                     )
    release = gs.PointLineRelease(num_elements=1000,
                                  release_time=start_time,
                                  start_position=(-144, 48.5, 0),
                                  )
    substance = gs.GnomeOil(filename='alaska-north-slope_AD00020.json')
    spill = gs.Spill(release=release,
                     substance=substance,
                     amount=5000,
                     units='bbls')
    model.spills += spill

.. admonition:: A note on "Windage"

    Floating objects experience a drift due to the wind. The default for substances is to have windage values set in the range 1-4% with a persistence (``  windage_persist``) of 15 minutes (900 seconds).
    This means that each element gets a random value in the range specified, and that value gets reset to a new random value every 15 minutes.
    If the ``windage_persist`` is set to a value of -1, then the value is persisted infinitely long, i.e. never reset.
.. We should reference th new tech doc when it's published
    .. More detail on the wind drift parametrization can be found in the |gnome_tech_manual|.


Polygon Releases
----------------

The :class:`gnome.spills.release.PolygonRelease` Object releases particles distributed over a polygon or set
of polygons. The particles are distributed randomly over the polygons with a simple weighting by polygon area (i.e., geographically larger polygons will be seeded with more particles). The subclass :class:`gnome.spills.release.NESDISRelease` can be used specifically with the |nesdis_reports| operationally produced by NOAA's Office of Satellite and Product Operations.


Using Helper Functions
----------------------

Rather than deal with the complexities of the Spill class directly, helper functions in the scripting package can be utilized for a lot of typical use cases. Some examples are include below.

Surface spill
.............

We use the :func:`gnome.scripting.surface_point_line_spill` helper function to initialize a spill along a line that occurs over one day.
The oil type is specified using the sample oil file provided above with a spill volume of 5000 barrels.
The windage range is changed from the default to 1-2% with an infinite persistence (elements keep the same windage value for all time).
The helper function creates both the ``Release`` and the ``Substance`` objects and uses them to create a Spill object.

.. code-block:: python

    import gnome.scripting as gs
    start_time = gs.asdatetime("2015-01-01T00:00")
    model = gs.Model(start_time=start_time,
                     duration=gs.days(3),
                     time_step=60 * 15, # 15 minutes in seconds
                     )

    spill = gs.surface_point_line_spill(num_elements=1000,
                                        start_position=(-144, 48.5),
                                        release_time=start_time,
                                        end_position=(-144, 48.6),
                                        end_release_time= start_time + gs.days(1),
                                        amount=5000,
                                        substance=gs.GnomeOil(filename='alaska-north-slope_AD00020.json'),
                                        units='bbl',
                                        windage_range=(0.01, 0.02),
                                        windage_persist=-1,
                                        name='My spill')
    model.spills += spill

    # ... add movers/weatherers

    model.full_run()

.. _subsurface_plume:

Subsurface plume
................

**NOTE** the subsurface plume class is being rewritten (Oct 2022) -- it may not work as described here.


For initialization of a subsurface plume, we can use the :func:`gnome.scripting.subsurface_plume_spill` helper function.
Required parameters in this case also include a specification of the droplet size distribution or of the rise velocities.
The :mod:`gnome.utilities.distributions` module includes methods for
specifying different types of distributions.
In this case we specify a uniform distribution of droplets ranging from 10-300 microns:

.. code-block:: python

    import gnome.scripting as gs
    from gnome.utilities.distributions import UniformDistribution
    from datetime import datetime, timedelta
    start_time = datetime(2015, 1, 1, 0, 0)
    model = gs.Model(start_time=start_time,
              duration=timedelta(days=3),
              time_step=60 * 15, #seconds
              )
    ud = UniformDistribution(10e-6,300e-6) #droplets in the range 10-300 microns
    spill = gs.subsurface_spill(num_elements=1000,
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


